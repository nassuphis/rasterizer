# compose.py
"""
Image composition and visual effects.

This module handles:
- Footer/label text rendering and placement
- Rounded passepartout (mat frame) generation
- Image padding and alignment
- Bilevel image processing
"""
from __future__ import annotations

import pyvips as vips


def add_footer_label(
    base: vips.Image,
    text: str,
    *,
    footer_frac: float = 0.02,   # ≈ target glyph height vs H
    pad_lr_px: int = 48,
    dpi: int = 300,
    align: str = "centre",
    invert: bool = False,
    font_family: str = "PT Mono",
    font_weight: str = "Regular",
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


def pad_to_square(im: vips.Image, px: int) -> vips.Image:
    dx = max(0, (px - im.width) // 2)
    dy = max(0, (px - im.height) // 2)
    canvas = vips.Image.black(px, px)
    return canvas.insert(im, dx, dy)
