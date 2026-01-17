# canvas.py
"""
Low-level rasterization and point stamping.

This module contains numba-compiled functions for:
- Disc/dot stamp generation
- Point stamping onto canvas
- Radius bucketing (sequential and parallel)
- Complex coordinate projection to pixel space
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


# ========================================
# dot stamping
# ========================================

def make_disc_offsets(r: int):
    """Produce a circular stamp of radius r."""
    r = int(max(1, r))
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    mask = (xx*xx + yy*yy) < r*r
    return yy[mask].astype(np.int32), xx[mask].astype(np.int32)


def build_disc_offset_cache_from_rpx(r_px: np.ndarray, rmin: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Build a cache {radius -> (dy, dx)} for all radii that will be stamped.
    Uses absolute radii and ignores those < rmin.
    """
    if r_px.size == 0:
        return {}
    r_abs = np.abs(r_px).astype(np.int32, copy=False)
    m = r_abs >= rmin
    if not np.any(m):
        return {}
    radii = np.unique(r_abs[m])
    return {int(r): make_disc_offsets(int(r)) for r in radii}


@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def stamp_points(canvas, ys, xs, dy, dx, value:np.int8=255):
    """Stamp points onto canvas.

    Args:
        canvas (np.zeros((H, W), np.uint8)): pixels are stamped here
        ys (int32): sorted y pixel coordinate
        xs (int32): sorted x pixel coordinate
        dy (int32): stamp pixel y coordinate
        dx (int32): stamp pixel x coordinate
        value (np.int8, optional): value to stamp. Defaults to 255.
    """
    H, W = canvas.shape
    n = ys.size; k = dy.size
    for i in prange(n):
        y0 = ys[i]; x0 = xs[i]
        for j in range(k):
            y = y0 + dy[j]; x = x0 + dx[j]
            if 0 <= y < H and 0 <= x < W:
                canvas[y, x] = value  # 255 draw, 0 erase


# ========================================
# sort points by dot radius
# ========================================

@njit(cache=True, nogil=True)
def bucket_by_radius(r_px: np.ndarray, r_min: int, r_max: int):
    n = r_px.size
    if n == 0 or r_min > r_max:
        return (np.empty(0, np.int64),
                np.empty(0, np.int32),
                np.empty(0, np.int64),
                np.empty(0, np.int64))
    size = r_max + 1
    counts_full = np.zeros(size, np.int64)
    kept = 0
    for i in range(n):
        r = r_px[i]
        if r_min <= r <= r_max:
            counts_full[r] += 1; kept += 1
    if kept == 0:
        return (np.empty(0, np.int64),
                np.empty(0, np.int32),
                np.empty(0, np.int64),
                np.empty(0, np.int64))
    starts_full = np.zeros(size, np.int64)
    s = 0
    for r in range(r_min, r_max + 1):
        c = counts_full[r]; starts_full[r] = s; s += c
    order = np.empty(kept, np.int64)
    write_ptr = starts_full.copy()
    for i in range(n):
        r = r_px[i]
        if r_min <= r <= r_max:
            p = write_ptr[r]; order[p] = i; write_ptr[r] = p + 1
    k = 0
    for r in range(r_min, r_max + 1):
        if counts_full[r] > 0: k += 1
    r_vals  = np.empty(k, np.int32)
    starts  = np.empty(k, np.int64)
    counts  = np.empty(k, np.int64)
    pos = 0
    for r in range(r_min, r_max + 1):
        c = counts_full[r]
        if c > 0:
            r_vals[pos] = np.int32(r)
            starts[pos] = starts_full[r]
            counts[pos] = c
            pos += 1
    return order, r_vals, starts, counts


@njit(cache=True, nogil=True, parallel=True)
def bucket_by_radius_parallel(r_px: np.ndarray, r_min: int, r_max: int):
    n = r_px.size
    if n == 0 or r_min > r_max:
        return (np.empty(0, np.int64),
                np.empty(0, np.int32),
                np.empty(0, np.int64),
                np.empty(0, np.int64))

    R = r_max - r_min + 1
    if R <= 0:
        return (np.empty(0, np.int64),
                np.empty(0, np.int32),
                np.empty(0, np.int64),
                np.empty(0, np.int64))

    # choose #blocks; ~64k elems per block, cap 1024
    B = (n + 65535) // 65536
    if B < 1: B = 1
    if B > 1024: B = 1024

    local = np.zeros((B, R), np.int64)

    # per-block histograms
    for b in prange(B):
        lo = (n * b) // B
        hi = (n * (b + 1)) // B
        row = local[b]
        for i in range(lo, hi):
            r = r_px[i]
            if r_min <= r <= r_max:
                row[r - r_min] += 1

    # reduce to global counts
    counts = np.zeros(R, np.int64)
    for r in range(R):
        s = 0
        for b in range(B):
            s += local[b, r]
        counts[r] = s

    kept = 0
    for r in range(R):
        kept += counts[r]
    if kept == 0:
        return (np.empty(0, np.int64),
                np.empty(0, np.int32),
                np.empty(0, np.int64),
                np.empty(0, np.int64))

    # global exclusive starts
    starts_full = np.empty(R, np.int64)
    s = 0
    for r in range(R):
        starts_full[r] = s
        s += counts[r]

    # per-block starts per radius
    block_starts = np.empty((B, R), np.int64)
    for r in range(R):
        off = starts_full[r]
        for b in range(B):
            block_starts[b, r] = off
            off += local[b, r]

    # parallel stable scatter: each block writes its own slice
    order = np.empty(kept, np.int64)
    for b in prange(B):
        lo = (n * b) // B
        hi = (n * (b + 1)) // B
        wp = block_starts[b].copy()
        for i in range(lo, hi):
            r = r_px[i]
            if r_min <= r <= r_max:
                rr = r - r_min
                p = wp[rr]
                order[p] = i
                wp[rr] = p + 1

    # compact to (r_vals, starts, counts) in ascending radius
    k = 0
    for r in range(R):
        if counts[r] > 0:
            k += 1

    r_vals = np.empty(k, np.int32)
    starts = np.empty(k, np.int64)
    cnts   = np.empty(k, np.int64)

    pos = 0
    for r in range(R):
        c = counts[r]
        if c > 0:
            r_vals[pos] = np.int32(r_min + r)
            starts[pos] = starts_full[r]
            cnts[pos]   = c
            pos += 1

    return order, r_vals, starts, cnts


# ========================================
# coordinate projection
# ========================================

def project_to_canvas(z: np.ndarray, pix: int, margin_frac: float):
    """Project complex coordinates to pixel coordinates."""
    if z.size<1: return np.empty(0,dtype=np.int32), np.empty(0,dtype=np.int32)
    half  = np.max(np.abs(z)) * (1.0 + 2.0 * margin_frac)
    #half = (0.5*max(np.ptp(z.real),np.ptp(z.imag))) * (1.0 + 2.0 * margin_frac)
    span  = 2.0 * half
    if span<1e-10: span=1
    px_per = (int(pix) - 1) / span
    px = np.rint((z.real + half) * px_per).astype(np.int32)
    py = np.rint((half - z.imag) * px_per).astype(np.int32)
    px = np.clip(px, 0, int(pix)-1)
    py = np.clip(py, 0, int(pix)-1)
    return px, py


def render_to_canvas(z: np.ndarray, pix: int, margin_frac: float):
    """Render complex points directly to a canvas."""
    canvas = np.zeros((int(pix), int(pix)), np.uint8)
    px, py = project_to_canvas(z,pix,margin_frac)
    canvas[px,py] = 255
    return canvas


# ========================================
# warmup
# ========================================

def warmup_raster_kernels():
    """Warmup numba JIT-compiled functions."""
    try:
        dummy = np.zeros((8, 8), np.uint8)
        ys = np.arange(4, dtype=np.int32)
        xs = np.arange(4, dtype=np.int32)
        dy = np.array([-1,0,1], np.int32)
        dx = np.array([-1,0,1], np.int32)
        stamp_points(dummy, ys, xs, dy, dx)
        r = np.array([1,2,1,3,2], np.int32)
        bucket_by_radius(r, 1, 3)
        bucket_by_radius_parallel(r, 1, 3)
    except Exception as e:
        print(f"[jit] raster warmup skipped: {e}")
