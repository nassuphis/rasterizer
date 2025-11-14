# raster.py
import math
import numpy as np
from numba import njit, prange
import pyvips as vips

# ========================================
# dot stamping
# ========================================

# produce a stamp
def make_disc_offsets(r: int):
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
    """_summary_
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


def project_to_canvas(z: np.ndarray, pix: int, margin_frac: float):
    if z.size<1: return np.empty(0,dtype=np.int32), np.empty(0,dtype=np.int32)
    #half  = np.max(np.abs(z)) * (1.0 + 2.0 * margin_frac)
    half = (0.5*max(np.ptp(z.real),np.ptp(z.imag))) * (1.0 + 2.0 * margin_frac)
    span  = 2.0 * half
    if span<1e-10: span=1
    px_per = (int(pix) - 1) / span
    px = np.rint((z.real + half) * px_per).astype(np.int32)
    py = np.rint((half - z.imag) * px_per).astype(np.int32)
    px = np.clip(px, 0, int(pix)-1)
    py = np.clip(py, 0, int(pix)-1)
    return px, py

def render_to_canvas(z: np.ndarray, pix: int, margin_frac: float):
    canvas = np.zeros((int(pix), int(pix)), np.uint8)
    px, py = project_to_canvas(z,pix,margin_frac)
    canvas[px,py] = 255
    return canvas

# ========================================
# image output
# ========================================

def np_to_vips_gray_u8(arr: np.ndarray) -> vips.Image:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    H, W = arr.shape
    return vips.Image.new_from_memory(arr.data, W, H, 1, "uchar")

def add_footer_label(
    base: vips.Image,
    text: str,
    *,
    footer_frac: float = 0.02,   # ≈ target glyph height vs H
    pad_lr_px: int = 40,
    dpi: int = 300,
    align: str = "centre",
    invert: bool = False,
    font_family: str = "Courier New",
    font_weight: str = "Bold",
    min_px: int = 10,
    max_px_frac: float = 0.05,
    max_retries: int = 8,
) -> vips.Image:
    H, W = base.height, base.width
    if H <= 0 or W <= 0 or not text:
        return base

    bottom_margin_px = max(2, H // 40)
    box_w = max(1, W - 2 * pad_lr_px)

    target_px = int(max(min_px, min(H * footer_frac, H * max_px_frac)))
    pt = max(6, int(round(target_px * 72.0 / dpi)))
    pt = min(pt, 512)

    tokens = [tok.strip() for tok in text.split(",")]
    tokens = [t for t in tokens if t]

    def wrap_lines(font_str: str) -> list[str]:
        lines: list[str] = []
        line = ""
        for tok in tokens:
            piece = tok if not line else f"{line}, {tok}"
            test = vips.Image.text(piece, dpi=dpi, font=font_str, align=align)
            if test.width <= box_w or not line:
                line = piece
            else:
                lines.append(line)
                line = tok
        if line:
            lines.append(line)
        return lines

    for _ in range(max_retries):
        font_str = f"{font_family} {font_weight} {pt}"
        try:
            lines = wrap_lines(font_str)
            glyph = vips.Image.text("\n".join(lines), dpi=dpi, font=font_str, align=align)
        except vips.Error:
            pt = max(6, int(pt * 0.85))
            continue

        glyph = (glyph > 0).ifthenelse(255, 0, blend=False)
        if glyph.height > int(H * max_px_frac * 1.1) or glyph.width > (box_w * 1.02):
            pt = max(6, int(pt * 0.9))
            continue

        gx = pad_lr_px + max(0, (box_w - glyph.width) // 2)
        gy = max(0, H - glyph.height - bottom_margin_px)
        glyph_full = vips.Image.black(W, H).insert(glyph, gx, gy)
        return base | glyph_full if not invert else base & (255 - glyph_full)

    return base  # fallback: unchanged



def add_rounded_passepartout_bilevel_pct(
    img: vips.Image,
    margin_frac: float = 0.10,   # 10% of width
    radius_frac: float = 0.04,   # 4% of width
    auto_white_bg: bool = True,
    mat_value: int | None = None,
):
    if img.bands != 1:
        raise ValueError("Expected a single-band (1-channel) image.")
    base = img if img.format == "uchar" else img.cast("uchar")

    H, W = base.height, base.width
    if H != W:
        raise ValueError("Expected a square image.")

    # px from fractions
    margin_px = max(0, int(round(W * float(margin_frac))))
    radius_px = max(0, int(round(W * float(radius_frac))))
    Wc, Hc = W + 2 * margin_px, H + 2 * margin_px

    # --- decide mat color (0 or 255) ---
    if auto_white_bg:
        b = max(2, int(round(0.005 * W)))
        strip = vips.Image.arrayjoin(
            [
                base.crop(0, 0, W, b),
                base.crop(0, H - b, W, b),
                base.crop(0, 0, b, H),
                base.crop(W - b, 0, b, H),
            ],
            across=2,
        )
        mean_val = float(strip.avg())
        mat = 255 if mean_val < 96 else 0
    else:
        if mat_value is None:
            mat_value = 255
        mat = 255 if mat_value > 127 else 0

    # canvas with mat color, paste the image
    canvas  = vips.Image.black(Wc, Hc).new_from_image(mat)
    composed = canvas.insert(base, margin_px, margin_px)

    # if no rounding, just return bilevel
    if radius_px <= 0:
        return (composed > 127).ifthenelse(255, 0)

    # --- build INNER rounded-rectangle mask ---
    # inner window position & size
    x0, y0 = margin_px, margin_px
    wi, hi = W, H
    # clamp radius to inner window
    radius_px = min(radius_px, wi // 2, hi // 2)

    # mask = 255 inside the rounded inner window, 0 elsewhere
    mask = vips.Image.black(Wc, Hc).new_from_image(0)
    # straight parts
    mask = mask.draw_rect(255, x0 + radius_px, y0,            wi - 2 * radius_px, hi,              fill=True)
    mask = mask.draw_rect(255, x0,             y0 + radius_px, wi,                hi - 2 * radius_px, fill=True)
    # four quarter-circles (centers on the inner window corners)
    mask = mask.draw_circle(255, x0 + radius_px,         y0 + radius_px,         radius_px, fill=True)  # TL
    mask = mask.draw_circle(255, x0 + wi - 1 - radius_px, y0 + radius_px,         radius_px, fill=True)  # TR
    mask = mask.draw_circle(255, x0 + radius_px,         y0 + hi - 1 - radius_px, radius_px, fill=True)  # BL
    mask = mask.draw_circle(255, x0 + wi - 1 - radius_px, y0 + hi - 1 - radius_px, radius_px, fill=True)  # BR

    # composite: show the pasted image only inside the INNER rounded window; elsewhere use mat
    mat_img = vips.Image.black(Wc, Hc).new_from_image(mat)
    out = mask.ifthenelse(composed, mat_img)

    # enforce bilevel (safe even if already 0/255)
    return (out > 127).ifthenelse(255, 0)

def save_png_bilevel(
    canvas: np.ndarray,
    out_path: str,
    invert: bool,
    footer_text: str | None = None,
    *,
    footer_pad_lr_px: int = 48,
    footer_dpi: int = 300,
    passepartout = False
):
    """
    Save a bilevel (0/255) PNG from a numpy array, optionally adding a footer title.
    """
    if canvas.dtype != np.uint8:
        canvas = canvas.astype(np.uint8, copy=False)
    if invert:
        canvas = 255 - canvas

    H, W = canvas.shape
    base = vips.Image.new_from_memory(canvas.data, W, H, 1, "uchar")

    if footer_text:
        base = add_footer_label(
            base,
            footer_text,
            pad_lr_px=footer_pad_lr_px,
            dpi=footer_dpi,
            align="centre",
            invert=invert,
        )

    if passepartout:
        base = add_rounded_passepartout_bilevel_pct(
            base,
            margin_frac = 0.01,   # e.g. 0.10 = 10% of width
            radius_frac = 0.1,   # e.g. 0.04 = 4% of width
            auto_white_bg = True,
            mat_value = None,   # 255 or 0 if you want to override
        )


    base.write_to_file(
        out_path,
        compression=1,
        effort=1,
        filter="none",
        interlace=False,
        strip=True,
        bitdepth=1,
    )


def pad_to_square(im: vips.Image, px: int) -> vips.Image:
    dx = max(0, (px - im.width) // 2)
    dy = max(0, (px - im.height) // 2)
    canvas = vips.Image.black(px, px)
    return canvas.insert(im, dx, dy)


def save_mosaic_png_bilevel(
    tiles: list[np.ndarray],
    titles: list[str] | None,
    *,
    cols: int,
    gap: int,
    out_path: str,
    invert: bool,
    footer_pad_lr_px: int = 48,
    footer_dpi: int = 300,
    thumbnail: int = None
) -> None:
    """
    Compose a mosaic from numpy tiles (uint8, 0/255). If 'titles' is provided,
    draw a footer on *each* tile before compositing. Saves as 1-bit PNG.
    """
    if not tiles:
        raise ValueError("No tiles provided")

    # Normalize titles: repeat single title, or pass through
    if titles is None:
        titles = [None] * len(tiles)
    elif len(titles) == 1 and len(tiles) > 1:
        titles = titles * len(tiles)
    elif len(titles) != len(tiles):
        raise ValueError("Length of 'titles' must be 1 or match number of tiles")

    # Convert first tile to VIPS to get tile size
    t0 = np_to_vips_gray_u8(tiles[0])
    tile_h, tile_w = t0.height, t0.width

    # Optionally pad any mismatched tiles to match the first tile size
    vtiles: list[vips.Image] = []
    for idx, (arr, title) in enumerate(zip(tiles, titles)):
        vt = np_to_vips_gray_u8(arr)
        # pad to match first tile size if needed
        if vt.width != tile_w or vt.height != tile_h:
            # center-pad to the larger of (tile_w, vt.width), (tile_h, vt.height)
            tw = max(tile_w, vt.width)
            th = max(tile_h, vt.height)
            vt = pad_to_square(vt, max(tw, th))
            tile_w = vt.width
            tile_h = vt.height
        # Add footer *before* global invert; draw text with invert=False
        if title:
            vt = add_footer_label(
                vt,
                title,
                pad_lr_px=footer_pad_lr_px,
                dpi=footer_dpi,
                align="centre",
                invert=False,  # global invert happens after composing
            )
        vtiles.append(vt)

    n = len(vtiles)
    rows = math.ceil(n / cols)
    W = cols * tile_w + (cols - 1) * gap
    H = rows * tile_h + (rows - 1) * gap
    base = vips.Image.black(W, H)

    # Composite tiles row-major
    for i, vt in enumerate(vtiles):
        r, c = divmod(i, cols)
        x = c * (tile_w + gap)
        y = r * (tile_h + gap)
        base = base.draw_image(vt, x, y)

    # Ensure bilevel and handle global invert
    base = (base > 0).ifthenelse(255, 0)
    if invert:
        base = base ^ 255

    if thumbnail:
        base_thumbnail = base.thumbnail_image(thumbnail)
        base_thumbnail.write_to_file(out_path)
    else:
        base.write_to_file(
            out_path,
            compression=1, effort=1, filter="none",
            interlace=False, strip=True, bitdepth=1,
        )


# ---------- warmup ----------

def warmup_raster_kernels():
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


