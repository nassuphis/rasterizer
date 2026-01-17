#!/usr/bin/env python
"""
preview_autolevels_rgb_vips.py

A pyvips implementation of the "Preview-like Auto Levels" pipeline you derived:

1) Compute pooled RGB histogram on INPUT (bins=256 by default):
     - Build per-channel PDFs (each normalized to sum=1)
     - pooled_pdf = (pdf_r + pdf_g + pdf_b) / 3

2) Optional peak limiting on pooled_pdf (spike suppression), then choose endpoints:
     black = quantile(clip_low%)
     white = quantile(1 - clip_high%)

3) Apply the same per-channel mapping to RGB:
     v = clamp((rgb - black) / (white - black), 0..1)
     v = v ** gamma
     optional sigmoid (endpoint-normalized)
     (alpha channel is preserved unchanged)

   Gamma can be auto-picked from the median of pooled_pdf to hit a target midtone.

4) Optional vibrance (approximation; NOT CoreImage's exact CIVibrance).

5) FINAL STEP (your key discovery):
     pooled RGB stretch on the OUTPUT to quantiles q and 1-q:
       lo = quantile(q), hi = quantile(1-q)
       rgb = clamp((rgb - lo) / (hi - lo), 0..1)

This script is both:
- a CLI tool
- an importable module with modular steps + a Pipeline builder

Dependencies:
  pip install pyvips numpy

System dependency:
  libvips must be installed (e.g. apt-get install libvips42, yum install vips, etc.)

Notes on "exactness":
- The tone/levels logic matches your Swift structure.
- JPEG decoding and color management will differ between Apple frameworks and libvips.
- Vibrance is an approximation (if you need exact CIVibrance behavior, keep vibrance=0).

"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pyvips


# ----------------------------
# Low-level helpers
# ----------------------------

VIPS_TO_NUMPY_DTYPE: Dict[str, Any] = {
    "uchar": np.uint8,
    "char": np.int8,
    "ushort": np.uint16,
    "short": np.int16,
    "uint": np.uint32,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
    # complex formats omitted (not needed here)
}

# JPG meta-data

MetaDict = Dict[str, Tuple[int, Any]]  # name -> (gtype, value)

def set_vips_threads(threads: Optional[int]) -> None:
    """
    Set libvips concurrency (threads used per operation) for this process.
    """
    if threads is None:
        return
    try:
        pyvips.concurrency_set(int(threads))
    except Exception:
        # Older pyvips may not expose this; user can set VIPS_CONCURRENCY env var instead.
        pass


def _autorot(img: pyvips.Image) -> pyvips.Image:
    """
    Apply EXIF orientation if present (Preview-style display orientation).
    pyvips autorot may return either Image or (Image, ...) depending on version.
    """
    try:
        out = img.autorot()
        if isinstance(out, tuple):
            return out[0]
        return out
    except Exception:
        return img


def _ensure_rgb_or_rgba_u8(img: pyvips.Image) -> pyvips.Image:
    """
    Ensure image is u8 RGB or RGBA.
    - Grayscale -> replicate into RGB
    - CMYK etc -> try to convert to sRGB via colourspace if possible, else use first 3 bands
    """
    # If it's not 8-bit, cast to uchar (libvips will scale? It won't; it clips).
    # We prefer to keep original 8-bit JPEG decode, which is already uchar.
    if img.format != "uchar":
        img = img.cast("uchar")

    b = img.bands
    if b == 1:
        img = img.bandjoin([img, img])  # now 3 bands
        return img
    if b == 2:
        # gray + alpha -> replicate gray to RGB and keep alpha
        g = img.extract_band(0)
        a = img.extract_band(1)
        rgb = g.bandjoin([g, g])
        return rgb.bandjoin(a)
    if b >= 3:
        # Best effort: if interpretation suggests non-sRGB, try conversion.
        # This is optional; many JPEGs are already sRGB.
        try:
            # Only attempt if libvips knows interpretation; otherwise it's no-op or error.
            # 'srgb' is the common string accepted by libvips colourspace.
            img = img.colourspace("srgb")
        except Exception:
            pass

        # Keep first 3 channels and optional alpha (4th)
        if b == 3:
            return img
        # 4+ bands: keep RGBA (first 4)
        return img.extract_band(0, n=4)

    return img


def load_image(path: Union[str, Path], autorotate: bool = True) -> pyvips.Image:
    """
    Load an image file as float RGB(A) in [0, 1].
    """
    p = str(path)
    img = pyvips.Image.new_from_file(p, access="random")
    if autorotate:
        img = _autorot(img)
    img = _ensure_rgb_or_rgba_u8(img)
    # Convert to float range 0..1
    img_f = img.cast("float") / 255.0
    return img_f


def save_image(
    path: Union[str, Path],
    img_f01: pyvips.Image,
    *,
    quality: int = 95,
    strip: bool = False,
    meta_from: Optional[pyvips.Image] = None,
    extra_meta: Optional[MetaDict] = None,
    jpeg_subsample_mode: str = "on",
    jpeg_optimize_coding: bool = False,
    jpeg_interlace: bool = False,
) -> None:
    """
    Save float RGB(A) [0,1] image to path.

    - Preserves metadata by default (strip=False).
    - If meta_from is given, we *explicitly* copy metadata from that image
      right before saving (useful if some ops dropped metadata).
    - extra_meta lets you add/override metadata items (name -> (gtype, value))
      or use set_xmp_and_usercomment() for convenience.

    JPEG notes (important for matching macOS Preview / CoreImage output):
    - libvips will *disable* chroma subsampling automatically for Q >= 90 in 'auto' mode.
      That makes edges look "crisper" at high zoom.
      Preview's output often looks more "smoothed" because chroma subsampling is still used.
      Set jpeg_subsample_mode="on" to force subsampling at any quality.
      See libvips jpegsave docs for subsample_mode. 
    """
    out_path = str(path)
    ext = Path(out_path).suffix.lower()

    img_u8 = float01_to_u8(img_f01)

    if meta_from is not None:
        img_u8 = copy_metadata(meta_from, img_u8)

    if extra_meta:
        img_u8 = apply_metadata(img_u8, extra_meta, overwrite=True)

    if ext in [".jpg", ".jpeg"]:
        # JPEG can't store alpha
        if img_u8.bands == 4:
            img_u8 = img_u8.extract_band(0, n=3)

        # These are libvips jpegsave options.
        # subsample_mode values: "auto", "on", "off"
        img_u8.write_to_file(
            out_path,
            Q=int(quality),
            strip=strip,
            subsample_mode=jpeg_subsample_mode,
            optimize_coding=jpeg_optimize_coding,
            interlace=jpeg_interlace,
        )
    else:
        img_u8.write_to_file(out_path, strip=strip)



def extract_metadata(
    src: pyvips.Image,
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> MetaDict:
    """
    Extract metadata from `src` into a dict of name -> (gtype, value).
    Uses src.get_fields(), src.get_typeof(), src.get().
    """
    names = list(include) if include is not None else list(src.get_fields())
    ex = set(exclude or [])
    meta: MetaDict = {}
    for name in names:
        if name in ex:
            continue
        try:
            gtype = int(src.get_typeof(name))
            if gtype == 0:
                continue
            value = src.get(name)
            meta[name] = (gtype, value)
        except Exception:
            # Some fields may not be retrievable in some loaders/savers; skip.
            continue
    return meta


def apply_metadata(dst: pyvips.Image, meta: MetaDict, *, overwrite: bool = True) -> pyvips.Image:
    """
    Return a copy of `dst` with metadata applied via set_type().

    Note: libvips caches and shares images, so you must copy() before setting metadata.
    """
    out = dst.copy()
    for name, (gtype, value) in meta.items():
        if not overwrite:
            try:
                if int(out.get_typeof(name)) != 0:
                    continue
            except Exception:
                pass
        try:
            out.set_type(int(gtype), name, value)
        except Exception:
            # Fall back to set() if the field already exists.
            try:
                if int(out.get_typeof(name)) != 0:
                    out.set(name, value)
            except Exception:
                pass
    return out


def copy_metadata(
    src: pyvips.Image,
    dst: pyvips.Image,
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    overwrite: bool = True,
) -> pyvips.Image:
    """
    Copy metadata fields from src -> dst.
    """
    meta = extract_metadata(src, include=include, exclude=exclude)
    return apply_metadata(dst, meta, overwrite=overwrite)


def set_xmp_and_usercomment(
    img: pyvips.Image,
    *,
    xmp_packet: Optional[bytes] = None,
    user_comment: Optional[str] = None,
) -> pyvips.Image:
    """
    Convenience for the exact two tags you mentioned.

    Equivalent to:
      base.set_type(vips.GValue.blob_type, "xmp-data", ...)
      base.set_type(vips.GValue.gstr_type, "exif-ifd0-UserComment", ...)
    """
    out = img.copy()
    if xmp_packet is not None:
        out.set_type(pyvips.GValue.blob_type, "xmp-data", xmp_packet)
    if user_comment is not None:
        out.set_type(pyvips.GValue.gstr_type, "exif-ifd0-UserComment", user_comment)
    return out


def encode_image(
    img_f01: pyvips.Image,
    suffix: str,
    *,
    quality: int = 95,
    strip: bool = False,
    meta_from: Optional[pyvips.Image] = None,
    extra_meta: Optional[MetaDict] = None,
    jpeg_subsample_mode: str = "on",
    jpeg_optimize_coding: bool = False,
    jpeg_interlace: bool = False,
) -> bytes:
    """
    Encode an image to bytes WITHOUT saving to disk.

    Example:
        jpg_bytes = encode_image(out_img, ".jpg", quality=95, meta_from=in_img)

    Under the hood this uses libvips write_to_buffer(). Options can also be put in
    the suffix like ".jpg[Q=95]". 

    JPEG notes:
      - libvips disables chroma subsampling automatically for Q>=90 in 'auto' mode.
        We default to jpeg_subsample_mode="on" to better match Preview/CoreImage output.
        See jpegsave docs. 
    """
    ext = suffix.lower()
    img_u8 = float01_to_u8(img_f01)

    # Force-metadata copy just before writing (belt + suspenders).
    if meta_from is not None:
        img_u8 = copy_metadata(meta_from, img_u8)

    if extra_meta:
        img_u8 = apply_metadata(img_u8, extra_meta, overwrite=True)

    if ext in [".jpg", ".jpeg"] and img_u8.bands == 4:
        img_u8 = img_u8.extract_band(0, n=3)

    if ext in [".jpg", ".jpeg"]:
        return img_u8.write_to_buffer(
            ext,
            Q=int(quality),
            strip=strip,
            subsample_mode=jpeg_subsample_mode,
            optimize_coding=jpeg_optimize_coding,
            interlace=jpeg_interlace,
        )
    else:
        return img_u8.write_to_buffer(ext, strip=strip)

def split_alpha(img_f01: pyvips.Image) -> Tuple[pyvips.Image, Optional[pyvips.Image]]:
    if img_f01.bands == 4:
        rgb = img_f01.extract_band(0, n=3)
        a = img_f01.extract_band(3)
        return rgb, a
    return img_f01, None


def merge_alpha(rgb_f01: pyvips.Image, alpha: Optional[pyvips.Image]) -> pyvips.Image:
    if alpha is None:
        return rgb_f01
    return rgb_f01.bandjoin(alpha)


def clamp01(img: pyvips.Image) -> pyvips.Image:
    """
    Clamp image to [0, 1] elementwise using ifthenelse (portable across pyvips versions).
    """
    img = (img < 0.0).ifthenelse(0.0, img)
    img = (img > 1.0).ifthenelse(1.0, img)
    return img


def float01_to_u8(img_f01: pyvips.Image) -> pyvips.Image:
    """
    Convert float [0,1] to uchar [0,255] with rounding.
    """
    img = clamp01(img_f01)
    img255 = img * 255.0
    # round: add 0.5 then floor by cast
    img255 = img255 + 0.5
    # Clip just in case rounding makes 255.5
    img255 = (img255 < 0.0).ifthenelse(0.0, img255)
    img255 = (img255 > 255.0).ifthenelse(255.0, img255)
    return img255.cast("uchar")


def hist_1d_u8(band_u8: pyvips.Image, bins: int = 256) -> np.ndarray:
    """
    Compute 1D histogram for a single-band uchar image.
    Returns counts array of length `bins`.
    """
    if band_u8.format != "uchar":
        band_u8 = band_u8.cast("uchar")

    h = band_u8.hist_find()  # typically width=bins, height=1, bands=1
    dtype = VIPS_TO_NUMPY_DTYPE.get(h.format, np.uint32)

    mem = h.write_to_memory()
    arr = np.frombuffer(mem, dtype=dtype)

    # Histogram image might be (height * width * bands). Expect width=bins, height=1, bands=1.
    # Fallback: reshape using reported dimensions.
    expected = h.width * h.height * h.bands
    if arr.size != expected:
        # last resort: trim
        arr = arr[:expected]

    arr = arr.reshape((h.height, h.width, h.bands))
    # take first row, first band
    counts = arr[0, :, 0].astype(np.float64)

    if counts.size != bins:
        # If the histogram width differs (rare), resample by simple interpolation.
        x_old = np.linspace(0, 1, counts.size)
        x_new = np.linspace(0, 1, bins)
        counts = np.interp(x_new, x_old, counts)

    return counts


def pooled_rgb_pdf_from_image(img_f01: pyvips.Image, bins: int = 256) -> np.ndarray:
    """
    Compute pooled RGB PDF exactly like the Swift:
      - per-channel PDF = hist(channel) / sum(hist(channel))
      - pooled PDF = (pdf_r + pdf_g + pdf_b) / 3
      - renormalize
    """
    rgb, _ = split_alpha(img_f01)
    rgb_u8 = float01_to_u8(rgb)
    r = rgb_u8.extract_band(0)
    g = rgb_u8.extract_band(1)
    b = rgb_u8.extract_band(2)

    hr = hist_1d_u8(r, bins=bins)
    hg = hist_1d_u8(g, bins=bins)
    hb = hist_1d_u8(b, bins=bins)

    r_tot = hr.sum()
    g_tot = hg.sum()
    b_tot = hb.sum()
    if r_tot <= 0 or g_tot <= 0 or b_tot <= 0:
        raise ValueError("Histogram totals were zero.")

    pr = hr / r_tot
    pg = hg / g_tot
    pb = hb / b_tot

    pooled = (pr + pg + pb) / 3.0
    s = pooled.sum()
    if s > 0:
        pooled /= s
    return pooled


def peak_limit_pdf(pdf: np.ndarray, factor: float) -> np.ndarray:
    """
    Peak-limit a PDF exactly like Swift:
      avg = 1/bins, limit = factor*avg
      cap bins above limit, redistribute excess uniformly, renormalize
    """
    if factor <= 0:
        return pdf.copy()

    bins = pdf.size
    avg = 1.0 / float(bins)
    limit = factor * avg

    out = pdf.copy()
    excess = np.maximum(out - limit, 0.0).sum()
    out = np.minimum(out, limit)

    out += excess / float(bins)
    s = out.sum()
    if s > 0:
        out /= s
    return out


def cdf_from_pdf(pdf: np.ndarray) -> np.ndarray:
    return np.cumsum(pdf, dtype=np.float64)


def find_bin(cdf: np.ndarray, target: float) -> int:
    # first index where cdf >= target
    idx = int(np.searchsorted(cdf, target, side="left"))
    if idx < 0:
        return 0
    if idx >= cdf.size:
        return cdf.size - 1
    return idx


def endpoints_from_pdf(pdf: np.ndarray, clip_low_percent: float, clip_high_percent: float) -> Tuple[float, float, int, int]:
    """
    Compute black/white in [0,1] from percent clipping.
    Mirrors the Swift logic.
    """
    cdf = cdf_from_pdf(pdf)
    low_target = float(np.clip(clip_low_percent / 100.0, 0.0, 1.0))
    high_target = float(np.clip(1.0 - (clip_high_percent / 100.0), 0.0, 1.0))

    bbin = find_bin(cdf, low_target)
    wbin = find_bin(cdf, high_target)

    bins = pdf.size
    black = bbin / float(bins - 1)
    white = wbin / float(bins - 1)
    if white <= black:
        white = black + 1e-6
    return black, white, bbin, wbin


def gamma_from_median(pdf: np.ndarray, black: float, white: float, target: float) -> Tuple[float, int]:
    """
    Compute gamma so that the median bin maps to `target` after linear stretch.
    Mirrors Swift computeGammaFromMedian().
    """
    cdf = cdf_from_pdf(pdf)
    med_bin = find_bin(cdf, 0.5)
    bins = pdf.size
    x = med_bin / float(bins - 1)

    denom = max(white - black, 1e-9)
    s = (x - black) / denom
    s = float(np.clip(s, 1e-6, 1.0 - 1e-6))

    t = float(np.clip(target, 1e-6, 1.0 - 1e-6))

    g = math.log(t) / math.log(s)
    if not math.isfinite(g):
        g = 1.0
    g = float(np.clip(g, 0.5, 2.0))
    return g, med_bin


def apply_rgb_curve(
    img_f01: pyvips.Image,
    black: float,
    white: float,
    gamma: float,
    sigmoid_strength: float = 0.0,
    sigmoid_mid: float = 0.5,
) -> pyvips.Image:
    """
    Apply the Swift kernel logic to each RGB channel:
      v = clamp((rgb-black)/(white-black), 0..1)
      v = v^gamma
      optional sigmoid with endpoint normalization
    """
    rgb, a = split_alpha(img_f01)

    denom = max(white - black, 1e-6)
    v = (rgb - black) / denom
    v = clamp01(v)

    # gamma
    if abs(gamma - 1.0) > 1e-12:
        v = v.math2_const('pow', [gamma])

    # sigmoid (endpoint normalized)
    if sigmoid_strength > 0:
        s0 = 1.0 / (1.0 + math.exp(sigmoid_strength * (sigmoid_mid - 0.0)))
        s1 = 1.0 / (1.0 + math.exp(sigmoid_strength * (sigmoid_mid - 1.0)))
        denom_s = max(s1 - s0, 1e-6)

        ss = 1.0 / (1.0 + (sigmoid_strength * (sigmoid_mid - v)).math('exp'))
        v = (ss - s0) / denom_s
        v = clamp01(v)

    out = merge_alpha(v, a)
    return out


def apply_final_pooled_rgb_stretch(
    img_f01: pyvips.Image,
    q: float,
    bins: int = 256,
) -> Tuple[pyvips.Image, Dict[str, Any]]:
    """
    FINAL STEP:
      compute pooled RGB PDF on current image, find lo/hi bins at q and 1-q,
      then linearly stretch each channel with clamp.
    """
    if q <= 0:
        return img_f01, {}

    pooled = pooled_rgb_pdf_from_image(img_f01, bins=bins)
    cdf = cdf_from_pdf(pooled)

    lo_bin = find_bin(cdf, q)
    hi_bin = find_bin(cdf, 1.0 - q)

    lo = lo_bin / float(bins - 1)
    hi = hi_bin / float(bins - 1)
    if hi <= lo:
        hi = lo + 1e-6

    rgb, a = split_alpha(img_f01)
    v = (rgb - lo) / max(hi - lo, 1e-6)
    v = clamp01(v)
    out = merge_alpha(v, a)
    dbg = {
        "pooled_rgb_q": q, 
        "final_lo_bin": lo_bin, 
        "final_hi_bin": hi_bin, 
        "final_lo": lo, 
        "final_hi": hi
    }
    return out, dbg


def apply_vibrance_approx(img_f01: pyvips.Image, amount: float) -> pyvips.Image:
    """
    Approximate vibrance:
      - compute saturation proxy sat = (max-min)/(max+eps)
      - boost factor = 1 + amount*(1 - sat)  (boost low-sat more)
      - push RGB away from mean by boost factor

    This is NOT Core Image's exact CIVibrance.
    Keep amount small (0.02..0.10).
    """
    if abs(amount) < 1e-12:
        return img_f01

    rgb, a = split_alpha(img_f01)
    r = rgb.extract_band(0)
    g = rgb.extract_band(1)
    b = rgb.extract_band(2)

    # per-pixel max/min via ifthenelse
    max_rg = (r > g).ifthenelse(r, g)
    max_rgb = (max_rg > b).ifthenelse(max_rg, b)

    min_rg = (r < g).ifthenelse(r, g)
    min_rgb = (min_rg < b).ifthenelse(min_rg, b)

    delta = max_rgb - min_rgb
    eps = 1e-6
    sat = delta / (max_rgb + eps)
    sat = clamp01(sat)

    boost = 1.0 + amount * (1.0 - sat)

    mean = (r + g + b) / 3.0

    r2 = mean + (r - mean) * boost
    g2 = mean + (g - mean) * boost
    b2 = mean + (b - mean) * boost

    out_rgb = r2.bandjoin([g2, b2])
    out_rgb = clamp01(out_rgb)
    return merge_alpha(out_rgb, a)


# ----------------------------
# Modular pipeline
# ----------------------------

Step = Callable[[pyvips.Image, Dict[str, Any]], pyvips.Image]


class Pipeline:
    def __init__(self) -> None:
        self.steps: List[Step] = []

    def add(self, step: Step) -> "Pipeline":
        self.steps.append(step)
        return self

    def run(self, img: pyvips.Image, return_debug: bool = False) -> Union[pyvips.Image, Tuple[pyvips.Image, Dict[str, Any]]]:
        dbg: Dict[str, Any] = {}
        out = img
        for s in self.steps:
            out = s(out, dbg)
        return (out, dbg) if return_debug else out


def step_pooled_rgb_levels(
    *,
    bins: int = 256,
    clip_low: float = 0.0,
    clip_high: float = 1.0,
    peak_factor: float = 0.0,
    gamma: float = 1.0,
    auto_gamma: str = "none",  # "none" or "median"
    target: float = 0.5,
    sigmoid_strength: float = 0.0,
    sigmoid_mid: float = 0.5,
) -> Step:
    """
    One step that:
      - computes pooled RGB PDF on current image
      - applies optional peak limiting
      - finds black/white from clip_low/clip_high
      - optionally auto-chooses gamma from median to hit target
      - applies curve per channel

    Mirrors Swift's Step 1 + Step 2 combined (since it depends on input histogram).
    """
    auto_gamma = auto_gamma.lower().strip()

    def _step(img: pyvips.Image, dbg: Dict[str, Any]) -> pyvips.Image:
        pooled = pooled_rgb_pdf_from_image(img, bins=bins)
        pooled2 = peak_limit_pdf(pooled, peak_factor)

        black, white, bbin, wbin = endpoints_from_pdf(pooled2, clip_low_percent=clip_low, clip_high_percent=clip_high)

        gamma_use = gamma
        med_bin = None
        if auto_gamma == "median":
            gamma_use, med_bin = gamma_from_median(pooled2, black, white, target=target)

        if dbg is not None:
            dbg.update({
                "bins": bins,
                "clip_low_percent": clip_low,
                "clip_high_percent": clip_high,
                "peak_factor": peak_factor,
                "black_bin": bbin,
                "white_bin": wbin,
                "black": black,
                "white": white,
                "gamma": gamma_use,
                "auto_gamma": auto_gamma,
                "median_bin": med_bin,
                "target": target,
                "sigmoid_strength": sigmoid_strength,
                "sigmoid_mid": sigmoid_mid,
            })

        out = apply_rgb_curve(
            img,
            black=black,
            white=white,
            gamma=gamma_use,
            sigmoid_strength=sigmoid_strength,
            sigmoid_mid=sigmoid_mid,
        )
        return out

    return _step


def step_vibrance(amount: float) -> Step:
    def _step(img: pyvips.Image, dbg: Dict[str, Any]) -> pyvips.Image:
        if dbg is not None:
            dbg["vibrance"] = float(amount)
        return apply_vibrance_approx(img, amount=float(amount))
    return _step


def step_final_pooled_rgb_stretch(q: float, *, bins: int = 256) -> Step:
    """
    FINAL pooled RGB stretch to quantiles q and 1-q.
    `q` can be:
      - fraction (0.01)
      - percent (1 -> 1%)  (for convenience, like your Swift)
    """
    q = float(q)
    if q > 0.5 and q <= 50.0:
        q = q / 100.0

    def _step(img: pyvips.Image, dbg: Dict[str, Any]) -> pyvips.Image:
        out, info = apply_final_pooled_rgb_stretch(img, q=q, bins=bins)
        if dbg is not None and info:
            dbg.update(info)
        return out

    return _step


# ----------------------------
# High-level processing
# ----------------------------

@dataclass
class AutoLevelsRGBConfig:
    bins: int = 256
    clip_low: float = 0.0
    clip_high: float = 1.0
    peak_factor: float = 0.0
    gamma: float = 1.0
    auto_gamma: str = "none"  # "none" or "median"
    target: float = 0.5
    sigmoid_strength: float = 0.0
    sigmoid_mid: float = 0.5
    vibrance: float = 0.0
    pooled_rgb: Optional[float] = None  # q (fraction or percent)
    threads: Optional[int] = None
    quality: int = 95

    # JPEG encode tuning (only affects .jpg/.jpeg output)
    jpeg_subsample_mode: str = "on"  # "auto", "on", "off"
    jpeg_optimize_coding: bool = False
    jpeg_interlace: bool = False

    def build_pipeline(self) -> Pipeline:
        p = Pipeline()
        p.add(step_pooled_rgb_levels(
            bins=self.bins,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
            peak_factor=self.peak_factor,
            gamma=self.gamma,
            auto_gamma=self.auto_gamma,
            target=self.target,
            sigmoid_strength=self.sigmoid_strength,
            sigmoid_mid=self.sigmoid_mid,
        ))
        if abs(self.vibrance) > 1e-12:
            p.add(step_vibrance(self.vibrance))
        if self.pooled_rgb is not None and float(self.pooled_rgb) != 0.0:
            p.add(step_final_pooled_rgb_stretch(self.pooled_rgb, bins=self.bins))
        return p


def process_image(
    img_f01: pyvips.Image,
    cfg: AutoLevelsRGBConfig,
    *,
    return_debug: bool = False
) -> Union[pyvips.Image, Tuple[pyvips.Image, Dict[str, Any]]]:
    pipe = cfg.build_pipeline()
    return pipe.run(img_f01, return_debug=return_debug)


def process_file(
    in_path: Union[str, Path],
    out_path: Union[str, Path],
    cfg: AutoLevelsRGBConfig,
    *,
    dump: bool = False,
) -> None:
    set_vips_threads(cfg.threads)
    img = load_image(in_path, autorotate=True)
    out, dbg = process_image(img, cfg, return_debug=True)
    if dump:
        print(f"== {in_path} ==")
        for k in sorted(dbg.keys()):
            print(f"{k}: {dbg[k]}")
    save_image(
        out_path, 
        out, 
        quality=cfg.quality, 
        meta_from=img, 
        jpeg_subsample_mode=cfg.jpeg_subsample_mode, 
        jpeg_optimize_coding=cfg.jpeg_optimize_coding, 
        jpeg_interlace=cfg.jpeg_interlace
    )


# ----------------------------
# CLI
# ----------------------------

def _parse_sigmoid(s: str) -> Tuple[float, float]:
    parts = s.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("sigmoid must be SxM, e.g. 3x0.5")
    try:
        strength = float(parts[0])
        mid = float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError("sigmoid must be SxM with numeric values")
    return strength, mid


def _expand_inputs(inputs: List[str]) -> List[str]:
    # Allow shell globs to be expanded by shell; still accept @filelist.
    out: List[str] = []
    for p in inputs:
        if p.startswith("@"):
            lst = Path(p[1:])
            if not lst.exists():
                raise FileNotFoundError(f"List file not found: {lst}")
            for line in lst.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                out.append(line)
        else:
            out.append(p)
    return out


def _resolve_output_paths(inputs: List[str], output: str) -> List[str]:
    outp = Path(output)
    if len(inputs) == 1 and (not outp.exists() or outp.is_file()) and output.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")):
        return [output]

    # Otherwise treat as directory
    outp.mkdir(parents=True, exist_ok=True)
    res: List[str] = []
    for inp in inputs:
        stem = Path(inp).name
        res.append(str(outp / stem))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview-like Auto Levels (pyvips).")
    ap.add_argument("inputs", nargs="+", help="Input image(s). You can also pass @list.txt")
    ap.add_argument("output", help="Output file (single input) or output directory (multiple inputs).")

    ap.add_argument("--dump", "-d", action="store_true", help="Print chosen parameters per file.")

    ap.add_argument("--bins", type=int, default=256)
    ap.add_argument("--clip-low", type=float, default=0.0, help="Percent clipped in shadows.")
    ap.add_argument("--clip-high", type=float, default=1.0, help="Percent clipped in highlights.")
    ap.add_argument("--peak-factor", type=float, default=0.0, help="Peak limiting factor (0 disables).")

    ap.add_argument("--gamma", type=float, default=1.0, help="Fixed gamma (disables auto-gamma).")
    ap.add_argument("--auto-gamma", choices=["none", "median"], default="none")
    ap.add_argument("--target", type=float, default=0.5, help="Target median after stretch (auto-gamma).")

    ap.add_argument("--sigmoid", type=_parse_sigmoid, default=(0.0, 0.5), help="SxM, e.g. 3x0.5")
    ap.add_argument("--vibrance", type=float, default=0.0, help="Approx vibrance, small values recommended (0.02..0.10).")

    ap.add_argument("--pooled_rgb", "--pooled-rgb", type=float, default=0.0,
                    help="FINAL pooled RGB stretch to q and 1-q. q can be fraction (0.01) or percent (1=1%).")

    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (if saving .jpg).")
    ap.add_argument("--jpeg-subsample", choices=["auto", "on", "off"], default="on",
                    help="JPEG chroma subsampling mode. libvips disables subsampling for Q>=90 in auto mode; "
                         "set 'on' to force subsampling (often matches Preview/CoreImage look better).")
    ap.add_argument("--jpeg-optimize-coding", action="store_true",
                    help="JPEG: optimize Huffman tables (smaller files, slightly slower).")
    ap.add_argument("--jpeg-interlace", action="store_true",
                    help="JPEG: progressive/interlaced output (does not change pixel values).")


    ap.add_argument("--threads", type=int, default=None,
                    help="libvips threads per process (sets VIPS_CONCURRENCY).")

    ap.add_argument("--jobs", type=int, default=1,
                    help="Number of processes for file-level parallelism. "
                         "If >1, consider also setting --threads to avoid oversubscription.")

    args = ap.parse_args()

    inputs = _expand_inputs(args.inputs)
    outputs = _resolve_output_paths(inputs, args.output)

    sig_s, sig_m = args.sigmoid

    cfg = AutoLevelsRGBConfig(
        bins=args.bins,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        peak_factor=args.peak_factor,
        gamma=args.gamma,
        auto_gamma=args.auto_gamma,
        target=args.target,
        sigmoid_strength=float(sig_s),
        sigmoid_mid=float(sig_m),
        vibrance=args.vibrance,
        pooled_rgb=(args.pooled_rgb if args.pooled_rgb != 0.0 else None),
        threads=args.threads,
        quality=args.quality,
        jpeg_subsample_mode=args.jpeg_subsample,
        jpeg_optimize_coding=args.jpeg_optimize_coding,
        jpeg_interlace=args.jpeg_interlace,
    )

    if args.jobs <= 1 or len(inputs) == 1:
        for inp, outp in zip(inputs, outputs):
            process_file(inp, outp, cfg, dump=args.dump)
    else:
        # Multiprocessing: each process runs its own libvips pipeline.
        # Avoid oversubscription: set --threads to something reasonable.
        def _worker(pair: Tuple[str, str]) -> Tuple[str, Optional[str]]:
            inp, outp = pair
            try:
                process_file(inp, outp, cfg, dump=args.dump)
                return inp, None
            except Exception as e:
                return inp, str(e)

        pairs = list(zip(inputs, outputs))
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_worker, p) for p in pairs]
            errors: List[str] = []
            for f in cf.as_completed(futs):
                inp, err = f.result()
                if err:
                    errors.append(f"{inp}: {err}")
            if errors:
                raise SystemExit("Some files failed:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()