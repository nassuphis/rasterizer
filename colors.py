import math
import numpy as np
import colorsys
from scipy.ndimage import gaussian_filter, sobel, laplace, gaussian_laplace, convolve
import re
from collections import Counter
try:
    import color_dicts
except ModuleNotFoundError:
    # allow import when loaded via 'from rasterizer import colors' style path tricks
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import color_dicts

DEFAULT_GAMMA = 1

_COLOR_NAME_MAP = color_dicts.COLOR_NAME_MAP

COLOR_TRI_STRINGS = color_dicts.COLOR_TRI_STRINGS
COLOR_LONG_STRINGS = color_dicts.COLOR_LONG_STRINGS
COLOR_STRINGS = {}
COLOR_STRINGS.update(COLOR_TRI_STRINGS)
COLOR_STRINGS.update(COLOR_LONG_STRINGS)

def parse_color_spec(spec: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Parse a color spec into (R,G,B) in 0..255.

    Accepts:
        - "RRGGBB" hex
        - "#RRGGBB" hex
        - simple names: red, blue, yellow, ...
    """
    if not isinstance(spec, str):
        return default

    s = spec.strip()
    if not s:
        return default

    if s.startswith("#"):
        s = s[1:]

    # Name → hex mapping
    lower = s.lower()
    if lower in _COLOR_NAME_MAP:
        s = _COLOR_NAME_MAP[lower]

    if len(s) != 6:
        return default

    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except ValueError:
        return default
    return float(r), float(g), float(b)


# ---------------------------------------------------------------------------
# Histogram equalization
# ---------------------------------------------------------------------------

def _hist_equalize(values: np.ndarray, nbins: int = 256) -> np.ndarray:
    """
    Simple 1D histogram equalization on 'values', returning t in [0,1].

    We use a fixed number of bins and map each value to the CDF bin
    it falls into. This is O(N) and works well for large images.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros_like(values, dtype=np.float64)

    vmin = float(values.min())
    vmax = float(values.max())

    if (not math.isfinite(vmin)) or (not math.isfinite(vmax)) or vmax <= vmin:
        return np.zeros_like(values, dtype=np.float64)

    hist, bin_edges = np.histogram(values, bins=nbins, range=(vmin, vmax))
    cdf = hist.cumsum().astype(np.float64)
    if cdf[-1] <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    cdf /= cdf[-1]

    # For each value, find its bin and pick the CDF
    idx = np.searchsorted(bin_edges, values, side="right") - 1
    idx = np.clip(idx, 0, nbins - 1)
    t = cdf[idx]
    return t


# ---------------------------------------------------------------------------
# HSV helpers
# ---------------------------------------------------------------------------


def _rgb255_to_hsv01(rgb255: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = rgb255
    return colorsys.rgb_to_hsv(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0)

# IMPORTANT: do NOT replace with modulo-based shortest-arc logic.
# That reintroduces a branch-cut seam (see palette field artifacts).
def _interp_hue_circle(h0, h1, t):
    """
    Hue interpolation via complex unit circle blend.

    This avoids hard seams from wrap/branch-cut logic. For antipodal hues
    (180° apart) the blend magnitude can approach 0; in that case we fall
    back to h0 (hue is undefined there anyway).
    """
    h0 = np.asarray(h0, dtype=np.float64)
    h1 = np.asarray(h1, dtype=np.float64)
    t  = np.asarray(t,  dtype=np.float64)

    a0 = 2.0 * np.pi * h0
    a1 = 2.0 * np.pi * h1

    z0 = np.cos(a0) + 1j * np.sin(a0)
    z1 = np.cos(a1) + 1j * np.sin(a1)

    z = (1.0 - t) * z0 + t * z1
    mag = np.abs(z)

    # if mag ~ 0, hue is undefined; pick h0 deterministically
    eps = 1e-12
    z = np.where(mag < eps, z0, z)

    ang = np.angle(z)  # (-pi, pi]
    return (ang / (2.0 * np.pi)) % 1.0


def _hsv01_to_rgb255_batch(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Vectorized HSV(0..1) -> RGB(0..255). Returns float64 (...,3).
    """
    c = v * s
    hp = h * 6.0
    x = c * (1.0 - np.abs((hp % 2.0) - 1.0))
    m = v - c

    r_ = np.zeros_like(h)
    g_ = np.zeros_like(h)
    b_ = np.zeros_like(h)

    c0 = (0.0 <= hp) & (hp < 1.0)
    c1 = (1.0 <= hp) & (hp < 2.0)
    c2 = (2.0 <= hp) & (hp < 3.0)
    c3 = (3.0 <= hp) & (hp < 4.0)
    c4 = (4.0 <= hp) & (hp < 5.0)
    c5 = (5.0 <= hp) & (hp < 6.0)

    r_[c0], g_[c0], b_[c0] = c[c0], x[c0], 0.0
    r_[c1], g_[c1], b_[c1] = x[c1], c[c1], 0.0
    r_[c2], g_[c2], b_[c2] = 0.0, c[c2], x[c2]
    r_[c3], g_[c3], b_[c3] = 0.0, x[c3], c[c3]
    r_[c4], g_[c4], b_[c4] = x[c4], 0.0, c[c4]
    r_[c5], g_[c5], b_[c5] = c[c5], 0.0, x[c5]

    r = (r_ + m) * 255.0
    g = (g_ + m) * 255.0
    b = (b_ + m) * 255.0
    return np.stack([r, g, b], axis=-1)

# ---------------------------------------------------------------------------
# Field normalization
# ---------------------------------------------------------------------------

def _two_sided_t_and_masks(v: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        t     : float64 (H,W) in [0,1]
        finite: bool (H,W)
        neg   : bool (H,W) finite & v<0
        pos   : bool (H,W) finite & v>0

    params:
        norm  : "linear" | "eq"  (default "linear")
        gamma : float (default DEFAULT_GAMMA)
        nbins : int (for eq, default 256)
    """
    v = np.asarray(v, dtype=np.float64)
    finite = np.isfinite(v)
    neg = finite & (v < 0.0)
    pos = finite & (v > 0.0)

    t = np.zeros_like(v, dtype=np.float64)
    norm = params.get("norm", "linear")
    nbins = int(params.get("nbins", 256))

    if norm == "eq":
        if np.any(neg):
            t[neg] = _hist_equalize(np.abs(v[neg]), nbins=nbins)
        if np.any(pos):
            t[pos] = _hist_equalize(v[pos], nbins=nbins)
    elif norm == "linear":
        min_neg = float(v[neg].min()) if np.any(neg) else 0.0
        max_pos = float(v[pos].max()) if np.any(pos) else 0.0
        scale = max(abs(min_neg), abs(max_pos))
        scale = 1.0 if (not math.isfinite(scale)) or scale <= 0.0 else scale
        if np.any(neg):
            t[neg] = np.abs(v[neg]) / scale
        if np.any(pos):
            t[pos] = v[pos] / scale
    else:
        raise ValueError(f"Unknown norm {norm!r}. Use 'linear' or 'eq'.")

    t = np.clip(t, 0.0, 1.0)

    gamma = params.get("gamma", DEFAULT_GAMMA)
    gamma = 1.0 if float(gamma) <= 0.0 else float(gamma)
    if gamma != 1.0:
        t = t ** gamma

    return t, finite, neg, pos

def _parse_tri_colors(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve neg / zero / pos colors from params into RGB255 float arrays.

    params:
        neg_color  : str (default "FFFF00")
        zero_color : str (default "000000")
        pos_color  : str (default "FF0000")

    Returns:
        neg_rgb  : (3,) float64
        zero_rgb : (3,) float64
        pos_rgb  : (3,) float64
    """
    pos_spec  = params.get("pos_color",  "FF0000")
    zero_spec = params.get("zero_color", "000000")
    neg_spec  = params.get("neg_color",  "FFFF00")

    neg_rgb  = np.asarray(parse_color_spec(neg_spec,  (255.0, 255.0, 0.0)), dtype=np.float64)
    zero_rgb = np.asarray(parse_color_spec(zero_spec, (0.0,   0.0,   0.0)), dtype=np.float64)
    pos_rgb  = np.asarray(parse_color_spec(pos_spec,  (255.0, 0.0,   0.0)), dtype=np.float64)

    return neg_rgb, zero_rgb, pos_rgb

# ---------------------------------------------------------------------------
# RGB Colorizers
# ---------------------------------------------------------------------------

def rgb_scheme_mh(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Markus & Hess style (RGB interpolation, linear normalization).
    """
    v = np.asarray(lyap, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("lyap array must be 2D (H, W)")

    H, W = v.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    t, finite, neg, pos = _two_sided_t_and_masks(v, dict(params, norm="linear"))
    if not np.any(finite): return rgb

    neg_rgb, zero_rgb, pos_rgb = _parse_tri_colors(params)

    if np.any(neg):
        tn = t[neg][:, None]
        rgb[neg] = np.rint((1.0 - tn) * zero_rgb + tn * neg_rgb).astype(np.uint8)

    if np.any(pos):
        tp = t[pos][:, None]
        rgb[pos] = np.rint((1.0 - tp) * zero_rgb + tp * pos_rgb).astype(np.uint8)

    return rgb

def rgb_scheme_mh_eq(lyap: np.ndarray, params: dict) -> np.ndarray:

    v = np.asarray(lyap, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("lyap array must be 2D (H, W)")

    H, W = v.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Shared normalization + masks
    t, finite, neg, pos = _two_sided_t_and_masks(v, dict(params, norm="eq"))
    if not np.any(finite): return rgb

    # Resolve endpoint colors once
    neg_rgb, zero_rgb, pos_rgb = _parse_tri_colors(params)

    # λ < 0 : zero -> neg
    if np.any(neg):
        tn = t[neg][:, None]
        rgb[neg] = np.rint((1.0 - tn) * zero_rgb + tn * neg_rgb).astype(np.uint8)

    # λ > 0 : zero -> pos
    if np.any(pos):
        tp = t[pos][:, None]
        rgb[pos] = np.rint((1.0 - tp) * zero_rgb + tp * pos_rgb).astype(np.uint8)

    return rgb

# ---------------------------------------------------------------------------
# composite rgb colorizers
# ---------------------------------------------------------------------------

def rgb_scheme_palette_eq(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Colorize using a named palette from COLOR_TRI_STRINGS.
    """
    palette_name = params.get("palette")
    if palette_name is None:
        raise ValueError(
            "params['palette'] must be set to a key in COLOR_TRI_STRINGS"
        )

    if palette_name not in COLOR_TRI_STRINGS:
        raise KeyError(
            f"Unknown palette {palette_name!r}. "
            f"Available: {', '.join(sorted(COLOR_TRI_STRINGS.keys()))}"
        )

    palette_spec = COLOR_TRI_STRINGS[palette_name]
    try:      
        parts = palette_spec.split(":")
        neg_spec, zero_spec, pos_spec = parts[:3]
    except ValueError as exc:
        raise ValueError(
            f"Invalid COLOR_TRI_STRINGS entry for {palette_name!r}: {palette_spec!r} "
            "(expected 'NEG:ZERO:POS')"
        ) from exc

    # Clone params so we don't mutate the caller's dict
    sub_params = dict(params)
    sub_params["neg_color"] = neg_spec
    sub_params["zero_color"] = zero_spec
    sub_params["pos_color"] = pos_spec

    return rgb_scheme_mh_eq(lyap, sub_params)


def rgb_scheme_multipoint(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Colorize using N color stops between -1 and +1.

    params:
        palette : str (preferred)
            Name of a palette in COLOR_STRINGS. Its value is a colon-
            separated list "HEX:HEX:HEX:...". All colors are used as
            equidistant stops in [-1, +1].

        color_string : str (optional override)
            If provided, overrides 'palette'. Same format as above.

        gamma : float (optional)
            Gamma applied to normalized coordinate in [0, 1].
            gamma <= 0 is treated as 1 (no gamma).

    Values outside [-1, +1] are clamped before mapping.
    Non-finite entries are left black.
    """
    arr = np.asarray(lyap, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("lyap array must be 2D (H, W)")

    H, W = arr.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    finite = np.isfinite(arr)
    if not np.any(finite):
        return rgb

    # Choose source of color string
    color_string = params.get("color_string")
    if not color_string:
        palette_name = params.get("palette")
        if not palette_name:
            raise ValueError(
                "scheme_multipoint requires either params['palette'] "
                "or params['color_string']"
            )
        try:
            color_string = COLOR_STRINGS[palette_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown palette {palette_name!r} for scheme_multipoint"
            ) from exc

    # Parse into list of RGB triples
    specs = [s.strip() for s in color_string.split(":") if s.strip()]
    if len(specs) < 2:
        raise ValueError(
            "scheme_multipoint needs at least 2 colors "
            "in color_string / palette"
        )

    colors = []
    for spec in specs:
        r, g, b = parse_color_spec(spec, (0.0, 0.0, 0.0))
        colors.append((r, g, b))

    colors = np.asarray(colors, dtype=np.float64)
    N = colors.shape[0]

    # Map [-1, +1] -> [0, 1], clamp
    vals = arr[finite]
    t = (np.clip(vals, -1.0, 1.0) + 1.0) * 0.5  # in [0, 1]

    gamma = params.get("gamma", 1)
    gamma = 1.0 if gamma <= 0.0 else float(gamma)
    if gamma != 1.0:
        t = t ** gamma

    # N colors => N-1 segments in [0,1]
    segment_float = t * (N - 1)
    idx_low = np.floor(segment_float).astype(np.int64)
    idx_low = np.clip(idx_low, 0, N - 2)
    frac = segment_float - idx_low

    c0 = colors[idx_low]           # (M, 3)
    c1 = colors[idx_low + 1]       # (M, 3)
    frac = frac[:, np.newaxis]     # (M, 1) for broadcasting

    rgb_vals = np.rint((1.0 - frac) * c0 + frac * c1).astype(np.uint8)
    rgb[finite] = rgb_vals

    return rgb

# ---------------------------------------------------------------------------
# HSV Colorizers
# ---------------------------------------------------------------------------

def hsv_scheme_mh_eq(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Markus & Hess style with histogram equalization, interpolating in HSV
    (shortest-arc hue), using shared two-sided normalization.

      λ < 0 : zero_color -> neg_color
      λ = 0 : zero_color
      λ > 0 : zero_color -> pos_color

    Normalization:
      - Uses _two_sided_t_and_masks(..., norm="eq")
      - t ∈ [0,1] is hist-eq(|λ|) on neg side and hist-eq(λ) on pos side
      - gamma applied inside the helper

    params:
        norm       : must be "eq" (forced internally)
        gamma      : float (optional, default DEFAULT_GAMMA)
        nbins      : int   (optional, default 256)

        pos_color  : str   (optional, default "FF0000")
        zero_color : str   (optional, default "000000")
        neg_color  : str   (optional, default "FFFF00")
    """
    v = np.asarray(lyap, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("lyap array must be 2D (H, W)")

    H, W = v.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    # Shared normalization + masks
    t, finite, neg, pos = _two_sided_t_and_masks(v, dict(params, norm="eq"))
    if not np.any(finite):
        return out

    # Resolve endpoint colors once
    neg_rgb, zero_rgb, pos_rgb = _parse_tri_colors(params)

    # Convert endpoints to HSV once
    neg_h, neg_s, neg_v = _rgb255_to_hsv01(tuple(neg_rgb))
    zero_h, zero_s, zero_v = _rgb255_to_hsv01(tuple(zero_rgb))
    pos_h, pos_s, pos_v = _rgb255_to_hsv01(tuple(pos_rgb))

    # λ < 0 : zero -> neg
    if np.any(neg):
        tn = t[neg]
        h = _interp_hue_circle(zero_h, neg_h, tn)
        s = zero_s + tn * (neg_s - zero_s)
        v_ = zero_v + tn * (neg_v - zero_v)

        rgb = _hsv01_to_rgb255_batch(h, s, v_)
        out[neg] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    # λ > 0 : zero -> pos
    if np.any(pos):
        tp = t[pos]
        h = _interp_hue_circle(zero_h, pos_h, tp)
        s = zero_s + tp * (pos_s - zero_s)
        v_ = zero_v + tp * (pos_v - zero_v)

        rgb = _hsv01_to_rgb255_batch(h, s, v_)
        out[pos] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    return out

def hsv_scheme_palette_eq(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Equivalent to rgb_scheme_palette_eq, but routes to hsv_scheme_mh_eq
    so interpolation is done in HSV.
    """
    palette_name = params.get("palette")
    if palette_name is None:
        raise ValueError("params['palette'] must be set to a key in COLOR_TRI_STRINGS")

    if palette_name not in COLOR_TRI_STRINGS:
        raise KeyError(
            f"Unknown palette {palette_name!r}. "
            f"Available: {', '.join(sorted(COLOR_TRI_STRINGS.keys()))}"
        )

    palette_spec = COLOR_TRI_STRINGS[palette_name]
    try:
        parts = palette_spec.split(":")
        neg_spec, zero_spec, pos_spec = parts[:3]
    except ValueError as exc:
        raise ValueError(
            f"Invalid COLOR_TRI_STRINGS entry for {palette_name!r}: {palette_spec!r} "
            "(expected 'NEG:ZERO:POS')"
        ) from exc

    sub_params = dict(params)
    sub_params["neg_color"] = neg_spec
    sub_params["zero_color"] = zero_spec
    sub_params["pos_color"] = pos_spec

    return hsv_scheme_mh_eq(lyap, sub_params)


# ---------------------------------------------------------------------------
# Palette field colorizers (per-pixel palette via palette arithmetic)
# ---------------------------------------------------------------------------

def _norm01_percentile(x: np.ndarray, lo: float = 10.0, hi: float = 99.0) -> np.ndarray:
    """
    Robust normalize to [0,1] via percentiles (computed on finite entries).
    """
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float64)

    a = x[finite]
    q0, q1 = np.percentile(a, [float(lo), float(hi)])
    if (not math.isfinite(q0)) or (not math.isfinite(q1)) or (q1 <= q0):
        return np.zeros_like(x, dtype=np.float64)

    y = (x - q0) / (q1 - q0)
    return np.clip(y, 0.0, 1.0)

def _tri_palette_from_name(palette_name: str) -> np.ndarray:
    """
    Return tri-palette as float64 array shape (3,3) in RGB255:
        idx 0 = neg, 1 = zero, 2 = pos
    """
    if palette_name not in COLOR_TRI_STRINGS:
        raise KeyError(
            f"Unknown palette {palette_name!r}. "
            f"Available: {', '.join(sorted(COLOR_TRI_STRINGS.keys()))}"
        )
    palette_spec = COLOR_TRI_STRINGS[palette_name]
    parts = palette_spec.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid COLOR_TRI_STRINGS entry for {palette_name!r}: {palette_spec!r} "
            "(expected 'NEG:ZERO:POS')"
        )
    neg_spec, zero_spec, pos_spec = parts[:3]
    neg_rgb = parse_color_spec(neg_spec, (0.0, 0.0, 0.0))
    zer_rgb = parse_color_spec(zero_spec, (0.0, 0.0, 0.0))
    pos_rgb = parse_color_spec(pos_spec, (0.0, 0.0, 0.0))
    P = np.asarray([neg_rgb, zer_rgb, pos_rgb], dtype=np.float64)
    return P  # (3,3)

def _tri_palette_from_name_hsv(palette_name: str) -> np.ndarray:
    """
    Return tri-palette as float64 array shape (3,3) in HSV01:
        idx 0 = neg, 1 = zero, 2 = pos
        channels: H,S,V in [0,1]
    """
    P_rgb = _tri_palette_from_name(palette_name)  # (3,3) RGB255
    hsv = []
    for i in range(3):
        h, s, v = _rgb255_to_hsv01(tuple(P_rgb[i]))
        hsv.append((h, s, v))
    return np.asarray(hsv, dtype=np.float64)


def _blend_tri_palettes_rgb(P0: np.ndarray, P1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Blend two tri-palettes into a per-pixel tri-palette field.

    P0,P1: (3,3) float64 in RGB255
    w: (H,W) in [0,1]
    returns: (H,W,3,3)
    """
    w = np.asarray(w, dtype=np.float64)
    return (1.0 - w[..., None, None]) * P0[None, None, :, :] + w[..., None, None] * P1[None, None, :, :]

def _tri_colorize_rgb_perpixel(v: np.ndarray, P: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Tri-palette interpolation with per-pixel palette P.

    v: (H,W) in [-1,1]
    P: (H,W,3,3) RGB255, stops [neg, zero, pos]
    t: (H,W) in [0,1] interpolation coordinate (usually abs(v) or equalized abs(v))

    returns uint8 (H,W,3)
    """
    v = np.asarray(v, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    H, W = v.shape
    out = np.zeros((H, W, 3), dtype=np.float64)

    finite = np.isfinite(v)
    if not np.any(finite):
        return out.astype(np.uint8)

    neg = finite & (v < 0.0)
    pos = finite & (v > 0.0)

    # stops
    Cn = P[:, :, 0, :]  # (H,W,3)
    Cz = P[:, :, 1, :]
    Cp = P[:, :, 2, :]

    if np.any(neg):
        tn = t[neg][:, None]
        out[neg] = (1.0 - tn) * Cz[neg] + tn * Cn[neg]

    if np.any(pos):
        tp = t[pos][:, None]
        out[pos] = (1.0 - tp) * Cz[pos] + tp * Cp[pos]

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)

def _tri_colorize_hsv_perpixel(v: np.ndarray, P_rgb: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Same as _tri_colorize_rgb_perpixel, but interpolates in HSV with shortest-arc hue.
    Palette stops are given in RGB255; converted to HSV once per call.

    Key extra robustness:
      - Hue is ill-defined when saturation is near 0. To avoid visible bands/seams,
        we "pin" the hue of near-neutral endpoints to the other endpoint before
        shortest-arc interpolation.
    """
    v = np.asarray(v, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    H, W = v.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    finite = np.isfinite(v)
    if not np.any(finite):
        return out

    # P_rgb: (H,W,3stops,3rgb)
    Pr = np.clip(P_rgb[..., 0], 0, 255) / 255.0  # (H,W,3)
    Pg = np.clip(P_rgb[..., 1], 0, 255) / 255.0
    Pb = np.clip(P_rgb[..., 2], 0, 255) / 255.0

    cmax = np.maximum(np.maximum(Pr, Pg), Pb)
    cmin = np.minimum(np.minimum(Pr, Pg), Pb)
    delta = cmax - cmin

    # Val
    Vv = cmax

    # Sat
    Ss = np.zeros_like(cmax)
    nonzero = cmax > 1e-12
    Ss[nonzero] = delta[nonzero] / cmax[nonzero]

    # Hue (continuous; no max-channel sector branching)
    Hh = np.zeros_like(cmax)
    eps = 1e-12
    mask = delta > eps
    num = np.sqrt(3.0) * (Pg - Pb)
    den = 2.0 * Pr - Pg - Pb
    Hh[mask] = (np.arctan2(num[mask], den[mask]) / (2.0 * np.pi)) % 1.0

    # Masks for sides
    neg = finite & (v < 0.0)
    pos = finite & (v > 0.0)

    # Stops in HSV: idx 0=neg, 1=zero, 2=pos
    Hn, Hz, Hp = Hh[:, :, 0], Hh[:, :, 1], Hh[:, :, 2]
    Sn, Sz, Sp = Ss[:, :, 0], Ss[:, :, 1], Ss[:, :, 2]
    Vn, Vz, Vp = Vv[:, :, 0], Vv[:, :, 1], Vv[:, :, 2]

    # If either endpoint is near-neutral, hue is ill-defined; pin to avoid bands.
    sat_eps = 1e-3  # try 3e-3 or 1e-2 if you still see artifacts

    if np.any(neg):
        tn = t[neg]
        h0, h1 = Hz[neg], Hn[neg]
        s0, s1 = Sz[neg], Sn[neg]
        v0, v1 = Vz[neg], Vn[neg]

        h0p = _hue_pin_smooth(h0, s0, h1, sat_eps)
        h1p = _hue_pin_smooth(h1, s1, h0, sat_eps)

        h  = _interp_hue_circle(h0p, h1p, tn)
        s  = s0 + tn * (s1 - s0)
        vv = v0 + tn * (v1 - v0)

        rgb = _hsv01_to_rgb255_batch(h, s, vv)
        out[neg] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    if np.any(pos):
        tp = t[pos]
        h0, h1 = Hz[pos], Hp[pos]
        s0, s1 = Sz[pos], Sp[pos]
        v0, v1 = Vz[pos], Vp[pos]

        # pin both endpoints toward each other, but WITHOUT feedback
        h0p = _hue_pin_smooth(h0, s0, h1, sat_eps)
        h1p = _hue_pin_smooth(h1, s1, h0, sat_eps)

        h  = _interp_hue_circle(h0p, h1p, tp)
        s  = s0 + tp * (s1 - s0)
        vv = v0 + tp * (v1 - v0)

        rgb = _hsv01_to_rgb255_batch(h, s, vv)
        out[pos] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    return out



def _blend_tri_palettes_hsv(P0_hsv: np.ndarray, P1_hsv: np.ndarray, w: np.ndarray, *, sat_eps: float = 1e-3) -> np.ndarray:
    """
    Blend two tri-palettes in HSV, saturation-aware so hue doesn't misbehave near neutrals.

    P0_hsv,P1_hsv: (3,3) HSV01 [neg, zero, pos]
    w: (H,W) in [0,1]
    returns: (H,W,3stops,3channels) HSV01
    """
    print("USING _blend_tri_palettes_hsv @", __file__, _blend_tri_palettes_hsv.__code__.co_firstlineno)
    w = np.asarray(w, dtype=np.float64)

    # broadcast stops -> (H,W,3)
    H0 = np.broadcast_to(P0_hsv[None, None, :, 0], w.shape + (3,))
    S0 = np.broadcast_to(P0_hsv[None, None, :, 1], w.shape + (3,))
    V0 = np.broadcast_to(P0_hsv[None, None, :, 2], w.shape + (3,))

    H1 = np.broadcast_to(P1_hsv[None, None, :, 0], w.shape + (3,))
    S1 = np.broadcast_to(P1_hsv[None, None, :, 1], w.shape + (3,))
    V1 = np.broadcast_to(P1_hsv[None, None, :, 2], w.shape + (3,))

    # Smoothly "pin" hue toward the other endpoint when saturation is low.
    # a ~ 0 => use other hue, a ~ 1 => keep own hue
    a0 = S0 / (S0 + sat_eps)
    a1 = S1 / (S1 + sat_eps)

    H0p = _interp_hue_circle(H1, H0, a0)  # if S0 small -> H0p ~ H1
    H1p = _interp_hue_circle(H0, H1, a1)  # if S1 small -> H1p ~ H0

    # Now blend hue on shortest arc, using pinned hues
    Hblend = _interp_hue_circle(H0p, H1p, w[..., None])
    Sblend = (1.0 - w[..., None]) * S0 + w[..., None] * S1
    Vblend = (1.0 - w[..., None]) * V0 + w[..., None] * V1

    return np.stack([Hblend, Sblend, Vblend], axis=-1)


def _hue_pin_smooth(h_self, s_self, h_other, sat_eps=1e-3):
    """
    If s_self is small, blend hue toward h_other smoothly.
    sat_eps controls how fast the transition happens.
    """
    a = s_self / (s_self + sat_eps)   # a in [0,1], smooth
    return _interp_hue_circle(h_other, h_self, a)

def _tri_colorize_hsv_perpixel_from_hsv(v: np.ndarray, P_hsv: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Tri-palette interpolation with per-pixel palette in HSV.

    v: (H,W) in [-1,1]
    P_hsv: (H,W,3stops,3) HSV01
    t: (H,W) in [0,1]

    returns uint8 (H,W,3) RGB
    """
    v = np.asarray(v, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    H, W = v.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    finite = np.isfinite(v)
    if not np.any(finite):
        return out

    neg = finite & (v < 0.0)
    pos = finite & (v > 0.0)

    # stops: idx 0=neg, 1=zero, 2=pos
    Hn, Hz, Hp = P_hsv[:, :, 0, 0], P_hsv[:, :, 1, 0], P_hsv[:, :, 2, 0]
    Sn, Sz, Sp = P_hsv[:, :, 0, 1], P_hsv[:, :, 1, 1], P_hsv[:, :, 2, 1]
    Vn, Vz, Vp = P_hsv[:, :, 0, 2], P_hsv[:, :, 1, 2], P_hsv[:, :, 2, 2]

    sat_eps = 1e-3

    if np.any(neg):
        tn = t[neg]
        h0, h1 = Hz[neg], Hn[neg]
        s0, s1 = Sz[neg], Sn[neg]
        v0, v1 = Vz[neg], Vn[neg]

        # smooth hue pinning (no hard threshold => no vertical seam)
        h0 = _hue_pin_smooth(h0, s0, h1, sat_eps)
        h1 = _hue_pin_smooth(h1, s1, h0, sat_eps)

        h = _interp_hue_circle(h0, h1, tn)
        s = s0 + tn * (s1 - s0)
        vv = v0 + tn * (v1 - v0)

        rgb = _hsv01_to_rgb255_batch(h, s, vv)
        out[neg] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    if np.any(pos):
        tp = t[pos]
        h0, h1 = Hz[pos], Hp[pos]
        s0, s1 = Sz[pos], Sp[pos]
        v0, v1 = Vz[pos], Vp[pos]

        # smooth hue pinning (no hard threshold => no vertical seam)
        h0 = _hue_pin_smooth(h0, s0, h1, sat_eps)
        h1 = _hue_pin_smooth(h1, s1, h0, sat_eps)

        h = _interp_hue_circle(h0, h1, tp)
        s = s0 + tp * (s1 - s0)
        vv = v0 + tp * (v1 - v0)

        rgb = _hsv01_to_rgb255_batch(h, s, vv)
        out[pos] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    return out


# Field Features

def _grad_components_scipy(v: np.ndarray, sigma: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=np.float64)
    if sigma and sigma > 0.0:
        v = gaussian_filter(v, sigma=float(sigma), mode="nearest")
    # sobel returns derivative-like responses
    gx = sobel(v, axis=1, mode="nearest")  # x = columns
    gy = sobel(v, axis=0, mode="nearest")  # y = rows
    return gx, gy

def _gradmag_scipy(v: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    gx, gy = _grad_components_scipy(v, sigma=sigma)
    return np.sqrt(gx*gx + gy*gy)

def _gabor_kernels(
    sigma: float,
    freq: float,
    theta: float,
    *,
    gabor_gamma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (cos_kernel, sin_kernel) for a real quadrature Gabor at orientation theta.

    sigma: envelope std in pixels
    freq : cycles per pixel (e.g. 0.05..0.25)
    theta: radians (0 = x-direction)
    gamma: aspect ratio ( <1 elongates along y', >1 along x' )
    """
    sigma = float(sigma)
    freq = float(freq)
    gabor_gamma = float(gabor_gamma)

    # kernel half-size: ~3 sigmas is enough
    r = int(max(1, math.ceil(3.0 * sigma)))
    y, x = np.mgrid[-r:r+1, -r:r+1].astype(np.float64)

    ct = math.cos(theta)
    st = math.sin(theta)

    # rotate coords: x' is along theta
    xp =  ct * x + st * y
    yp = -st * x + ct * y

    # anisotropic Gaussian envelope
    env = np.exp(-(xp * xp + (gabor_gamma * gabor_gamma) * (yp * yp)) / (2.0 * sigma * sigma))

    phase = 2.0 * math.pi * freq * xp
    kcos = env * np.cos(phase)
    ksin = env * np.sin(phase)

    # remove DC to avoid bias (important for smooth fields)
    kcos = kcos - kcos.mean()
    ksin = ksin - ksin.mean()

    return kcos, ksin

def _gabor_energy(
    v: np.ndarray,
    *,
    sigma: float,
    freq: float,
    theta: float,
    gabor_gamma: float = 1.0,
) -> np.ndarray:
    """
    Phase-invariant Gabor energy at one orientation.
    """
    kcos, ksin = _gabor_kernels(sigma, freq, theta, gabor_gamma=gabor_gamma)
    rc = convolve(v, kcos, mode="nearest")
    rs = convolve(v, ksin, mode="nearest")
    return np.sqrt(rc * rc + rs * rs)


def _feature_gabor_max(v, params:dict) -> np.array:
    # params: sigma, freq, ntheta, theta0, theta1, gamma
    sigma = float(params.get("sigma", 3.0))
    freq  = float(params.get("freq", 0.12))
    ntheta = int(params.get("ntheta", 8))
    theta0 = float(params.get("theta0", 0.0))
    theta1 = float(params.get("theta1", math.pi))
    gabor_gamma = float(params.get("gabor_gamma", 1.0))

    # optional pre-blur of v to reduce pixel noise before filtering
    pre_sigma = float(params.get("pre_sigma", 0.0))
    vv = gaussian_filter(v, sigma=pre_sigma, mode="nearest") if pre_sigma > 0 else v

    thetas = np.linspace(theta0, theta1, num=max(1, ntheta), endpoint=False)
    f = np.zeros_like(vv, dtype=np.float64)
    for th in thetas:
        e = _gabor_energy(vv, sigma=sigma, freq=freq, theta=float(th), gabor_gamma=gabor_gamma)
        f = np.maximum(f, e)
    return f

def _feature_gabor_theta(v, params:dict) -> np.array:
    # params: sigma, freq, theta, gamma
    sigma = float(params.get("sigma", 3.0))
    freq  = float(params.get("freq", 0.12))
    theta = float(params.get("theta", 0.0))
    gabor_gamma = float(params.get("gabor_gamma", 1.0))

    pre_sigma = float(params.get("pre_sigma", 0.0))
    vv = gaussian_filter(v, sigma=pre_sigma, mode="nearest") if pre_sigma > 0 else v

    f = _gabor_energy(vv, sigma=sigma, freq=freq, theta=theta, gabor_gamma=gabor_gamma)
    return f


def _feature_grad_dir(v, params:dict) -> np.array:
    theta = float(params.get("theta", 0.0))
    sigma = float(params.get("sigma", 2.0))
    gx, gy = _grad_components_scipy(v, sigma=sigma)
    proj = np.cos(theta) * gx + np.sin(theta) * gy
    proj_abs = np.abs(proj)
    gmag = np.sqrt(gx * gx + gy * gy) + 1e-12
    mode = str(params.get("mode", "align")).lower()
    if mode == "align": f = proj_abs / gmag
    elif mode == "strength": f = proj_abs
    else: raise ValueError(f"Unknown mode {mode!r}. Use 'align', 'strength'.")
    return f

def _feature_struct_tensor_coherence(v: np.ndarray, params: dict) -> np.ndarray:
    # Gaussian pre-blur before gradients
    sigma_pre = float(params.get("sigma_pre", 1.0))
    gx, gy =  _grad_components_scipy(v, sigma=sigma_pre)
    # Tensor smoothing scale
    sigma_tensor = float(params.get("sigma_tensor", 3.0))
    Jxx = gaussian_filter(gx * gx, sigma=sigma_tensor, mode="nearest")
    Jxy = gaussian_filter(gx * gy, sigma=sigma_tensor, mode="nearest")
    Jyy = gaussian_filter(gy * gy, sigma=sigma_tensor, mode="nearest")
    # Eigenvalue-based coherence
    tr = Jxx + Jyy
    det = (Jxx - Jyy)**2 + 4.0 * (Jxy**2)
    s = np.sqrt(np.maximum(det, 0.0))
    l1 = 0.5 * (tr + s)
    l2 = 0.5 * (tr - s)
    eps = 1e-12
    return (l1 - l2) / (l1 + l2 + eps)  # ∈ [0,1]

def _feature_ms_ratio(v: np.ndarray, params: dict) -> np.ndarray:
    """
    Multiscale band-energy ratio:
        w = E1 / (E1 + E2 + eps)

    Bands:
        b1 = G(s1) - G(s2)    (fine)
        b2 = G(s2) - G(s3)    (coarse)

    Energies (optional smoothing):
        E = G(se) (b^2)

    params:
        s1, s2, s3 : sigmas (s1 < s2 < s3)
        se         : energy smoothing sigma
        pre_sigma  : pre-smooth v (optional)
        power      : optional power on energies (default 1)
        eps        : stability epsilon
    """
    s1 = float(params.get("s1", 1.0))
    s2 = float(params.get("s2", 3.0))
    s3 = float(params.get("s3", 9.0))
    if not (s1 < s2 < s3):
        # enforce ordering without being annoying
        s1, s2, s3 = sorted([s1, s2, s3])
        if s1 == s2: s2 = s1 + 1e-6
        if s2 == s3: s3 = s2 + 1e-6

    se = float(params.get("se", 1.0))
    pre = float(params.get("pre_sigma", 0.0))
    power = float(params.get("power", 1.0))
    eps = float(params.get("eps", 1e-12))

    vv = gaussian_filter(v, sigma=pre, mode="nearest") if pre > 0 else v

    g1 = gaussian_filter(vv, sigma=s1, mode="nearest")
    g2 = gaussian_filter(vv, sigma=s2, mode="nearest")
    g3 = gaussian_filter(vv, sigma=s3, mode="nearest")

    b1 = g1 - g2
    b2 = g2 - g3

    E1 = b1 * b1
    E2 = b2 * b2
    if se > 0:
        E1 = gaussian_filter(E1, sigma=se, mode="nearest")
        E2 = gaussian_filter(E2, sigma=se, mode="nearest")

    if power != 1.0:
        # emphasize stronger bands (power>1) or soften (power<1)
        E1 = np.maximum(E1, 0.0) ** power
        E2 = np.maximum(E2, 0.0) ** power

    return E1 / (E1 + E2 + eps)


def _palette_weight_from_feature(v: np.ndarray, params: dict) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)

    w_feature = str(params.get("w_feature", "grad")).lower()

    # ---- compute raw feature f ----
    if w_feature == "grad":
        sigma = float(params.get("sigma", 0.0))
        f = _gradmag_scipy(v, sigma=sigma)

    elif w_feature == "gradx":
        sigma = float(params.get("sigma", 0.0))
        gx, gy = _grad_components_scipy(v,sigma=sigma)
        f = np.abs(gx)

    elif w_feature == "grady":
        sigma = float(params.get("sigma", 0.0))
        gx, gy = _grad_components_scipy(v,sigma=sigma)
        f = np.abs(gy)

    elif w_feature == "grad_dir":  # oriented energy along angle (radians)
        f = _feature_grad_dir(v, params)
 
    elif w_feature == "lap":
        sigma = float(params.get("sigma", 2.0))
        f = np.abs(gaussian_laplace(v, sigma=sigma, mode="nearest"))
        
    elif w_feature == "lvar":
        # local variance via box blurs; radius controlled by iterations
        sigma = float(params.get("sigma", 0.0))
        m = gaussian_filter(v, sigma=sigma, mode="nearest")
        m2 = gaussian_filter(v*v, sigma=sigma, mode="nearest")
        f = np.maximum(m2 - m * m, 0.0)

    elif w_feature == "dog":
        # difference of box blurs (poor man's DoG)
        sigma1 = float(params.get("sigma1", 1.0))
        sigma2 = float(params.get("sigma2", 4.0))
        if sigma2 <= sigma1: sigma2 = sigma1 + 1e-6
        g1 = gaussian_filter(v, sigma=sigma1, mode="nearest")
        g2 = gaussian_filter(v, sigma=sigma2, mode="nearest")
        b = g1 - g2
        mode = str(params.get("mode", "abs")).lower()
        if mode == "energy":
            energy_sigma = float(params.get("energy_sigma",1.0))
            f = gaussian_filter(b * b, sigma=energy_sigma, mode="nearest")
        else:
            f = np.abs(b)

    elif w_feature == "sign_coh":
        s = np.sign(v)
        sigma = float(params.get("sigma", 0.0))
        sbar = gaussian_filter(s, sigma=sigma, mode="nearest")
        f = np.abs(sbar)

    elif w_feature =="st_coh":
        f = _feature_struct_tensor_coherence(v,params)

    elif w_feature == "gabor_max":
        f = _feature_gabor_max(v,params)

    elif w_feature == "gabor_theta":
        f = _feature_gabor_theta(v,params)

    elif w_feature == "ms_ratio":
        f = _feature_ms_ratio(v, params)

    else:
        raise ValueError(
            f"Unknown w_feature {w_feature!r}. Try: "
            "'grad','gradx','grady','grad_dir','lap','lvar','dog','sign_coh','st_coh', 'gabor_max','gabor_theta'."
        )

    # ---- normalize f -> w in [0,1] ----
    w_lo = float(params.get("w_lo", 10.0))
    w_hi = float(params.get("w_hi", 99.0))
    w = _norm01_percentile(f, lo=w_lo, hi=w_hi)

    # smooth weight (separate from feature blurs)
    w_sigma = float(params.get("w_sigma", 0.0))
    if w_sigma > 0.0:
        w = gaussian_filter(w, sigma=w_sigma, mode="nearest")

    # gamma shaping on weight
    w_gamma = float(params.get("w_gamma", 1.0))
    if w_gamma > 0.0 and w_gamma != 1.0:
        w = np.clip(w, 0.0, 1.0) ** w_gamma

    # strength LAST -> guarantees w_strength=0 => paletteA everywhere
    strength = float(params.get("w_strength", 1.0))
    w = np.clip(w * strength, 0.0, 1.0)

    return w

def rgb_scheme_palette_field(lyap: np.ndarray, params: dict) -> np.ndarray:
    """
    Per-pixel tri-palette blending between two named tri-palettes.

    Key property (by design):
        - If w_strength == 0, Pf == paletteA everywhere, and the result matches
          the single-palette scheme (rgb_scheme_palette_eq) under the same
          normalization settings (norm/gamma/nbins).

    Required params:
        paletteA : str  (name in COLOR_TRI_STRINGS)
        paletteB : str  (name in COLOR_TRI_STRINGS)

    Uses shared normalization:
        t, finite, neg, pos = _two_sided_t_and_masks(v, params)
        where params["norm"] is typically "eq" or "linear".
    """
    v = np.asarray(lyap, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("lyap array must be 2D (H, W)")

    # palettes
    palA = params.get("paletteA")
    palB = params.get("paletteB")
    if not palA or not palB:
        raise ValueError("rgb_scheme_palette_field requires params['paletteA'] and params['paletteB']")

    P0 = _tri_palette_from_name(str(palA))
    P1 = _tri_palette_from_name(str(palB))

    # weight field (already applies w_strength internally)
    w = _palette_weight_from_feature(v, params)

    # per-pixel palette (H,W,3,3)
    Pf = _blend_tri_palettes_rgb(P0, P1, w)

    # interpolation coordinate t in [0,1] with shared semantics (norm + gamma)
    t, finite, neg, pos = _two_sided_t_and_masks(v, params)

    if not np.any(finite):
        return np.zeros((v.shape[0], v.shape[1], 3), dtype=np.uint8)

    return _tri_colorize_rgb_perpixel(v, Pf, t)




def hsv_scheme_palette_field(lyap: np.ndarray, params: dict) -> np.ndarray:
    v = np.asarray(lyap, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("lyap array must be 2D (H, W)")

    palA = params.get("paletteA")
    palB = params.get("paletteB")
    if not palA or not palB:
        raise ValueError("hsv_scheme_palette_field requires params['paletteA'] and params['paletteB']")

    # palettes in HSV (3 stops)
    P0_hsv = _tri_palette_from_name_hsv(str(palA))  # (3,3) HSV01
    P1_hsv = _tri_palette_from_name_hsv(str(palB))  # (3,3) HSV01

    # weight field (applies w_strength internally)
    w = _palette_weight_from_feature(v, params)      # (H,W)

    # per-pixel palette field in HSV
    Pf_hsv = _blend_tri_palettes_hsv(P0_hsv, P1_hsv, w)  # (H,W,3stops,3)

    # shared normalization (handles norm + gamma + nbins)
    t, finite, neg, pos = _two_sided_t_and_masks(v, params)
    if not np.any(finite):
        return np.zeros((v.shape[0], v.shape[1], 3), dtype=np.uint8)

    # colorize using HSV palette stops directly
    return _tri_colorize_hsv_perpixel_from_hsv(v, Pf_hsv, t)


def _tri_palette_from_any(pal) -> np.ndarray:
    """
    Accept:
      - palette name in COLOR_TRI_STRINGS
      - tri string "NEG:ZERO:POS" (each part name or hex)
      - array-like shape (3,3) in RGB255
    Return: (3,3) float64 RGB255 [neg, zero, pos]
    """
    if isinstance(pal, np.ndarray):
        P = np.asarray(pal, dtype=np.float64)
        if P.shape != (3, 3):
            raise ValueError(f"tri palette array must have shape (3,3), got {P.shape}")
        return P

    if isinstance(pal, (list, tuple)):
        P = np.asarray(pal, dtype=np.float64)
        if P.shape != (3, 3):
            raise ValueError(f"tri palette list/tuple must have shape (3,3), got {P.shape}")
        return P

    if not isinstance(pal, str):
        raise TypeError("palette must be a name, a 'NEG:ZERO:POS' string, or a (3,3) array")

    s = pal.strip()
    if not s:
        raise ValueError("empty palette spec")

    # name -> tri
    if s in COLOR_TRI_STRINGS:
        return _tri_palette_from_name(s)

    # "NEG:ZERO:POS" -> tri
    parts = [p.strip() for p in s.split(":") if p.strip()]
    if len(parts) < 3:
        raise ValueError(
            f"Invalid tri palette spec {pal!r}. Expected a name in COLOR_TRI_STRINGS "
            "or a string like 'NEG:ZERO:POS'."
        )
    neg_spec, zero_spec, pos_spec = parts[:3]
    neg_rgb = parse_color_spec(neg_spec, (0.0, 0.0, 0.0))
    zer_rgb = parse_color_spec(zero_spec, (0.0, 0.0, 0.0))
    pos_rgb = parse_color_spec(pos_spec, (0.0, 0.0, 0.0))
    return np.asarray([neg_rgb, zer_rgb, pos_rgb], dtype=np.float64)


def create_palette_field(
    paletteA,
    paletteB,
    *,
    pix: int = 512,
    interp: str = "rgb",
    gamma: float = 1.0,
    y_dir: str = "neg_to_pos",
) -> np.ndarray:
    """
    Visualize the 2D "palette field" space for tri-palettes.

    Axes:
      - x (columns): blend weight w in [0,1] from paletteA -> paletteB
      - y (rows):    v in [-1,+1] selecting color from the blended tri-palette:
                       v<0:  zero -> neg
                       v=0:  zero
                       v>0:  zero -> pos

    Args:
      paletteA, paletteB:
        - name in COLOR_TRI_STRINGS, or
        - tri string 'NEG:ZERO:POS', or
        - (3,3) RGB255 array-like
      pix:
        output image size: (pix, pix, 3)
      interp:
        "rgb" or "hsv" (HSV uses shortest-arc hue; matches your field HSV behavior)
      gamma:
        applied to t = |v| in [0,1] (gamma<=0 treated as 1)
      y_dir:
        "neg_to_pos" (top=-1, bottom=+1) or "pos_to_neg"

    Returns:
      uint8 RGB image (pix, pix, 3)
    """
    pix = int(pix)
    if pix <= 1:
        raise ValueError("pix must be >= 2")

    interp = str(interp).lower()
    if interp not in ("rgb", "hsv"):
        raise ValueError("interp must be 'rgb' or 'hsv'")

    gamma = float(gamma)
    if gamma <= 0.0:
        gamma = 1.0

    # resolve palettes (3,3) RGB255
    P0 = _tri_palette_from_any(paletteA)
    P1 = _tri_palette_from_any(paletteB)

    # x-axis weight field w in [0,1]
    w_col = np.linspace(0.0, 1.0, num=pix, endpoint=True, dtype=np.float64)
    w = np.broadcast_to(w_col[None, :], (pix, pix))  # (H,W)

    # y-axis value v in [-1,1]
    if str(y_dir).lower() == "neg_to_pos":
        v_row = np.linspace(-1.0, 1.0, num=pix, endpoint=True, dtype=np.float64)
    elif str(y_dir).lower() == "pos_to_neg":
        v_row = np.linspace(1.0, -1.0, num=pix, endpoint=True, dtype=np.float64)
    else:
        raise ValueError("y_dir must be 'neg_to_pos' or 'pos_to_neg'")

    v = np.broadcast_to(v_row[:, None], (pix, pix))  # (H,W)

    # per-pixel tri-palette field
    Pf_rgb = _blend_tri_palettes_rgb(P0, P1, w)  # (H,W,3stops,3rgb)

    # interpolation coordinate t in [0,1]
    t = np.abs(v)
    if gamma != 1.0:
        t = t ** gamma

    if interp == "rgb":
        Pf_rgb = _blend_tri_palettes_rgb(P0, P1, w)
        return _tri_colorize_rgb_perpixel(v, Pf_rgb, t)

    # interp == "hsv" (ALL-HSV)
    P0_hsv = np.asarray([_rgb255_to_hsv01(tuple(P0[i])) for i in range(3)], dtype=np.float64)  # (3,3)
    P1_hsv = np.asarray([_rgb255_to_hsv01(tuple(P1[i])) for i in range(3)], dtype=np.float64)  # (3,3)

    Pf_hsv = _blend_tri_palettes_hsv(P0_hsv, P1_hsv, w)  # (H,W,3,3)
    return _tri_colorize_hsv_perpixel_from_hsv(v, Pf_hsv, t)

def write_jpeg_rgb(rgb: np.ndarray, out_path: str, *, quality: int = 95) -> None:
    """
    Write uint8 RGB (H,W,3) to JPEG.
    Requires Pillow: pip install pillow
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H,W,3), got {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    try:
        from PIL import Image
    except Exception as exc:
        raise ImportError("write_jpeg_rgb requires Pillow (pip install pillow)") from exc

    Image.fromarray(rgb, mode="RGB").save(out_path, format="JPEG", quality=int(quality), optimize=True)


def _rgb_to_hsv_deg(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    return (h * 360.0) % 360.0, s, v

def _classify_color(spec: str,
                    neutral_sat: float = 0.12,
                    neutral_val_low: float = 0.10,
                    neutral_val_high: float = 0.98):
    r, g, b = parse_color_spec(spec, (0.0, 0.0, 0.0))
    h, s, v = _rgb_to_hsv_deg(r, g, b)

    if s < neutral_sat or v <= neutral_val_low or v >= neutral_val_high:
        return "neutral"

    # warm: reds/oranges/yellows
    return "warm" if (h < 70.0 or h >= 330.0) else "cool"

def _classify_tri(tri_spec: str):
    parts = [p.strip() for p in tri_spec.split(":")[:3]]
    if len(parts) != 3:
        return None

    labels = [_classify_color(p) for p in parts]
    c = Counter(labels)
    warm_n, cool_n, neutral_n = c["warm"], c["cool"], c["neutral"]

    if warm_n >= 2:
        pal = "warm"
    elif cool_n >= 2:
        pal = "cool"
    else:
        pal = "mixed"

    score = warm_n - cool_n  # neutrals ignored
    return pal, score, neutral_n, warm_n, cool_n

def filter_tri_palettes(d: dict,
                        *,
                        only: str | None = None,
                        name_regex: str | None = None,
                        min_score: int | None = None,
                        max_neutral: int | None = None,
                        min_warm: int | None = None,
                        min_cool: int | None = None):
    rx = re.compile(name_regex) if name_regex else None
    out = {}

    for name, tri in d.items():
        if rx and not rx.search(name):
            continue

        info = _classify_tri(tri)
        if info is None:
            continue
        pal, score, neutral_n, warm_n, cool_n = info

        if only and pal != only:
            continue
        if min_score is not None and score < min_score:
            continue
        if max_neutral is not None and neutral_n > max_neutral:
            continue
        if min_warm is not None and warm_n < min_warm:
            continue
        if min_cool is not None and cool_n < min_cool:
            continue

        out[name] = tri

    return out


def _cli_export_dict(d: dict, *, fmt: str, values: bool) -> str:
    keys = sorted(d.keys())

    if fmt == "lines":
        if values:
            # key<TAB>value
            return "\n".join(f"{k}\t{d[k]}" for k in keys) + ("\n" if keys else "")
        else:
            return "\n".join(keys) + ("\n" if keys else "")

    if fmt == "tsv":
        # header + key/value rows
        if values:
            out = ["name\tvalue"]
            out += [f"{k}\t{d[k]}" for k in keys]
        else:
            out = ["name"]
            out += keys
        return "\n".join(out) + "\n"

    if fmt == "json":
        import json
        if values:
            return json.dumps({k: d[k] for k in keys}, indent=2, sort_keys=True) + "\n"
        else:
            return json.dumps(keys, indent=2) + "\n"

    raise ValueError(f"Unknown format {fmt!r} (use: lines|tsv|json)")


def _main(argv=None) -> int:
    import argparse
    import sys
    from pathlib import Path

    p = argparse.ArgumentParser(prog="color.py")

    # NEW: optional positional "mode" + spec
    # Usage:
    #   color.py field PA,PB[,gamma[,pix[,interp]]] -o out.jpg
    p.add_argument(
        "mode",
        nargs="?",
        help="Optional mode. Use: field",
    )
    p.add_argument(
        "field",
        nargs="?",
        help="Field spec: PA,PB[,gamma[,pix[,interp]]]. Example: rg,gc,1.2,1024,hsv",
    )

    p.add_argument(
        "--export",
        choices=["color", "tri", "long", "all", "field"],  # <-- add field
        help="Export palette/name sets (to stdout).",
    )
    p.add_argument(
        "--format",
        default="lines",
        choices=["lines", "tsv", "json"],
        help="Output format (default: lines).",
    )
    p.add_argument(
        "--values",
        action="store_true",
        help="Include values (e.g. name<TAB>hex or JSON mapping).",
    )

    # existing tri filters...
    p.add_argument("--only", choices=["warm", "cool", "mixed"], help="Filter tri palettes by warm/cool/mixed.")
    p.add_argument("--name-regex", help="Regex filter on palette name (tri only).")
    p.add_argument("--min-score", type=int, help="Min warm-minus-cool score (tri only).")
    p.add_argument("--max-neutral", type=int, help="Max neutral colors allowed (0..3) (tri only).")
    p.add_argument("--min-warm", type=int, help="Require at least this many warm colors (0..3) (tri only).")
    p.add_argument("--min-cool", type=int, help="Require at least this many cool colors (0..3) (tri only).")

    # NEW: field output controls
    p.add_argument("-o", "--out", help="Output JPEG path for field render.")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95).")

    args = p.parse_args(argv)

    # ------------------------------------------------------------------
    # NEW: "field" mode takes priority if provided
    # ------------------------------------------------------------------
    if args.mode == "field":
        if not args.field:
            p.error("field mode requires FIELD spec: PA,PB[,gamma[,pix[,interp]]]")

        # parse PA,PB[,gamma[,pix[,interp]]]
        parts = [x.strip() for x in args.field.split(",") if x.strip()]
        if len(parts) < 2:
            p.error("FIELD must provide at least PA,PB (comma-separated)")

        palA = parts[0]
        palB = parts[1]
        gamma = float(parts[2]) if len(parts) >= 3 else 1.0
        pix   = int(parts[3]) if len(parts) >= 4 else 512
        interp = parts[4].lower() if len(parts) >= 5 else "rgb"

        out_path = args.out
        if not out_path:
            # sensible default filename
            safeA = palA.replace(":", "_")
            safeB = palB.replace(":", "_")
            out_path = f"field_{safeA}__{safeB}__g{gamma:g}__{pix}_{interp}.jpg"

        rgb = create_palette_field(
            palA,
            palB,
            pix=pix,
            gamma=gamma,
            interp=interp,
            y_dir="neg_to_pos",
        )
        write_jpeg_rgb(rgb, out_path, quality=args.quality)
        print(out_path)
        return 0

    # ------------------------------------------------------------------
    # Existing export behavior (unchanged), with NEW --export field
    # ------------------------------------------------------------------
    if not args.export:
        p.print_help()
        return 2

    if args.export == "field":
        # just print a short usage hint
        sys.stdout.write(
            "Field render mode (writes a JPEG):\n"
            "  color.py field PA,PB[,gamma[,pix[,interp]]] -o out.jpg\n"
            "Examples:\n"
            "  color.py field rg,gc,1.2,1024,hsv -o field.jpg\n"
            "  color.py field firebrick:black:sunset,seagreen:black:copper,1.0,768,rgb\n"
        )
        return 0

    if args.export == "color":
        d = _COLOR_NAME_MAP
        sys.stdout.write(_cli_export_dict(d, fmt=args.format, values=args.values))
        return 0

    if args.export == "tri":
        d = COLOR_TRI_STRINGS
        if any([
            args.only, args.name_regex, args.min_score is not None,
            args.max_neutral is not None, args.min_warm is not None, args.min_cool is not None
        ]):
            d = filter_tri_palettes(
                d,
                only=args.only,
                name_regex=args.name_regex,
                min_score=args.min_score,
                max_neutral=args.max_neutral,
                min_warm=args.min_warm,
                min_cool=args.min_cool,
            )
        sys.stdout.write(_cli_export_dict(d, fmt=args.format, values=args.values))
        return 0

    if args.export == "long":
        d = COLOR_LONG_STRINGS
        sys.stdout.write(_cli_export_dict(d, fmt=args.format, values=args.values))
        return 0

    if args.export == "all":
        # Names only by default; with --values we keep separate sections in TSV/lines,
        # and in JSON we emit an object with 3 fields.
        if args.format == "json":
            import json
            if args.values:
                obj = {
                    "color": dict(sorted(_COLOR_NAME_MAP.items())),
                    "tri": dict(sorted(COLOR_TRI_STRINGS.items())),
                    "long": dict(sorted(COLOR_LONG_STRINGS.items())),
                }
            else:
                obj = {
                    "color": sorted(_COLOR_NAME_MAP.keys()),
                    "tri": sorted(COLOR_TRI_STRINGS.keys()),
                    "long": sorted(COLOR_LONG_STRINGS.keys()),
                }
            sys.stdout.write(json.dumps(obj, indent=2) + "\n")
            return 0

        # lines/tsv: emit section headers
        def section(title: str, d: dict) -> str:
            header = f"# {title}"
            body = _cli_export_dict(d, fmt=("lines" if args.format == "lines" else "tsv"), values=args.values)
            return header + "\n" + body

        sys.stdout.write(section("color", _COLOR_NAME_MAP))
        sys.stdout.write(section("tri", COLOR_TRI_STRINGS))
        sys.stdout.write(section("long", COLOR_LONG_STRINGS))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(_main())



