import argparse
from pathlib import Path
import math
import numpy as np
import colorsys
from scipy.ndimage import gaussian_filter, sobel, laplace, gaussian_laplace, convolve
import numpy as np
import pyvips 


def hist_u8_band(im_band: pyvips.Image) -> np.ndarray:
    h = im_band.hist_find()
    mem = h.write_to_memory()
    arr = np.frombuffer(mem, dtype=np.uint32)
    if arr.size != 256:
        w = h.width
        if w > 0 and arr.size % w == 0:
            arr = arr.reshape((-1, w)).sum(axis=0)
        else:
            raise RuntimeError(f"Unexpected histogram size: {arr.size} (expected 256)")
    return arr.astype(np.int64)


def quantile_from_hist(hist: np.ndarray, q: float) -> float:
    if not (0.0 <= q <= 100.0):
        raise ValueError("Quantile must be between 0 and 100.")
    total = hist.sum()
    if total <= 0:
        return float("nan")
    target = (q / 100.0) * total
    cdf = np.cumsum(hist)
    v = int(np.searchsorted(cdf, target, side="left"))
    return float(min(255, max(0, v)))

def winsorize_per_channel(
        im: pyvips.Image, 
        hists: list[np.ndarray], 
        q_lo: float, q_hi: float
    ) -> tuple[pyvips.Image, list[float], list[float]]:
    
    qlo_rgb = [float(quantile_from_hist(h, q_lo)) for h in hists]
    qhi_rgb = [float(quantile_from_hist(h, q_hi)) for h in hists]

    r = im[0].clamp(min=qlo_rgb[0], max=qhi_rgb[0])
    g = im[1].clamp(min=qlo_rgb[1], max=qhi_rgb[1])
    b = im[2].clamp(min=qlo_rgb[2], max=qhi_rgb[2])
    return r.bandjoin([g, b]), qlo_rgb, qhi_rgb

# ---------------------------------------------------------------------------
# Histogram equalization
# ---------------------------------------------------------------------------

def hist_equalize(values: np.ndarray, nbins: int = 256) -> np.ndarray:
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


def rgb255_to_hsv01(rgb255: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = rgb255
    return colorsys.rgb_to_hsv(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0)

# IMPORTANT: do NOT replace with modulo-based shortest-arc logic.
# That reintroduces a branch-cut seam (see palette field artifacts).
def interp_hue_circle(h0, h1, t):
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


def hsv01_to_rgb255_batch(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
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




