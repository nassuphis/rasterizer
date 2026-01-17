# export.py
"""
Image file output and mosaic creation.

This module handles:
- JPEG/PNG/VIPS file saving
- Auto-levels configuration and application
- Metadata/spec attachment
- Mosaic/grid composition
- Thumbnail generation
"""
from __future__ import annotations

import os
import math

import numpy as np
import pyvips as vips

from . import autolevels
from . import footer
from . import image2spec
from .formats import np_to_vips_rgb_u8, np_to_vips_gray_u8
from .compose import add_footer_label, add_rounded_passepartout_bilevel_pct, pad_to_square


def add_suffix_number(path: str, n: int, width: int = 5) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}_{n:0{width}d}{ext}"


autolvlcfg = autolevels.AutoLevelsRGBConfig(
    bins=256,
    clip_low=0.0,
    clip_high=1.0,
    peak_factor=8.0,
    gamma=1.0,
    auto_gamma="median",
    target=0.55,
    sigmoid_strength=3,
    sigmoid_mid=0.5,
    vibrance=0.05,
    pooled_rgb=1,
    threads=None,
    quality=95,
    jpeg_subsample_mode="on",
    jpeg_optimize_coding=True,
    jpeg_interlace=True,
)


def save_jpg_rgb(
    rgb: np.ndarray,
    out_path: str,
    footer_text: str | None = None,
    *,
    footer_pad_lr_px: int = 48,
    footer_dpi: int = 300,
    quality: int = 95,
    spec: str | None = None,
    autolvl: bool = True,
    resize: int | None = None
) -> None:

    base0 = np_to_vips_rgb_u8(rgb)

    if resize:
        w, h = base0.width, base0.height
        if w <= 0 or h <= 0:
            raise ValueError(f"Bad image dimensions: {w}x{h}")
        scale = max(resize / w, resize / h)
        base = base0.resize(scale, kernel="mitchell")
    else:
        base = base0

    if autolvl:
        base_f01 = base.cast("float") / 255.0
        out_f01 = autolevels.process_image(base_f01, autolvlcfg)
        base1 = autolevels.float01_to_u8(out_f01)
    else:
        base1 = base

    if footer_text:
        glyph = footer.text2glyph(footer_text, 40, 1.0)
        base2 = footer.fade_glyph(base1, glyph, 0.1, 0.05)
    else:
        base2 = base1

    # Attach metadata BEFORE write
    if spec is not None:
        base3 = image2spec.spec2image(base2, spec)
    else:
        base3 = base2

    base3.write_to_file(
        out_path,
        Q=int(quality),
        strip=False,
        interlace=True,
    )


def save_vips_rgb(
    rgb: np.ndarray,
    out_path: str,
    invert: bool = False,
    footer_text: str | None = None,
    *,
    footer_pad_lr_px: int = 48,
    footer_dpi: int = 300,
) -> None:
    base = np_to_vips_rgb_u8(rgb)
    if invert: base = base ^ 255
    if footer_text:
        base = add_footer_label(
            base,
            footer_text,
            pad_lr_px=footer_pad_lr_px,
            dpi=footer_dpi,
            align="centre",
            invert=False,
        )
    root, ext = os.path.splitext(out_path)
    if ext.lower() not in {".v", ".vips"}:
        out_path = root + ".v"
    base.write_to_file(out_path)


def save_png_rgb(
    rgb: np.ndarray,
    out_path: str,
    invert: bool = False,
    footer_text: str | None = None,
    *,
    footer_pad_lr_px: int = 48,
    footer_dpi: int = 300,
) -> None:
    base = np_to_vips_rgb_u8(rgb)

    if invert: base = base ^ 255
    if footer_text:
        base = add_footer_label(
            base,
            footer_text,
            pad_lr_px=footer_pad_lr_px,
            dpi=footer_dpi,
            align="centre",
            invert=False,
        )
    base.write_to_file(
        out_path,
        compression=1,
        effort=1,
        interlace=False,
        strip=True,
    )


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
    if canvas.dtype != np.uint8:
        canvas = canvas.astype(np.uint8, copy=False)
    if invert: canvas = canvas ^ 255
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
            margin_frac = 0.01,
            radius_frac = 0.1,
            auto_white_bg = True,
            mat_value = None,
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


def save_mosaic_png_rgb(
    tiles: list[np.ndarray],
    titles: list[str] | None,
    *,
    cols: int,
    gap: int,
    out_path: str,
    invert: bool = False,
    footer_pad_lr_px: int = 48,
    footer_dpi: int = 300,
    thumbnail: int | None = None,
) -> None:
    """
    Compose a mosaic from RGB numpy tiles (H x W x 3 or 3 x H x W each) and
    save as 8-bit RGB PNG.

    All tiles must have the same height/width/channels. Titles (if provided)
    are drawn as footers per tile, like in save_mosaic_png_bilevel.
    """
    if not tiles:
        raise ValueError("No tiles provided")

    # Normalize titles (same logic as bilevel)
    if titles is None:
        titles = [None] * len(tiles)
    elif len(titles) == 1 and len(tiles) > 1:
        titles = titles * len(tiles)
    elif len(titles) != len(tiles):
        raise ValueError("Length of 'titles' must be 1 or match number of tiles")

    # Convert first tile to VIPS, check size/bands
    first = np_to_vips_rgb_u8(tiles[0])
    tile_h, tile_w, bands = first.height, first.width, first.bands

    vtiles: list[vips.Image] = []
    for arr, title in zip(tiles, titles):
        vt = np_to_vips_rgb_u8(arr)

        if vt.width != tile_w or vt.height != tile_h or vt.bands != bands:
            raise ValueError(
                f"All RGB tiles must have the same size/bands; "
                f"expected ({tile_w}x{tile_h},{bands}), "
                f"got ({vt.width}x{vt.height},{vt.bands})"
            )

        if title:
            vt = add_footer_label(
                vt,
                title,
                pad_lr_px=footer_pad_lr_px,
                dpi=footer_dpi,
                align="centre",
                invert=False,
            )
        vtiles.append(vt)

    n = len(vtiles)
    rows = math.ceil(n / cols)
    W = cols * tile_w + (cols - 1) * gap
    H = rows * tile_h + (rows - 1) * gap

    # Black RGB canvas
    base = vips.Image.black(W, H).new_from_image([0] * bands)

    # Composite tiles row-major
    for i, vt in enumerate(vtiles):
        r, c = divmod(i, cols)
        x = c * (tile_w + gap)
        y = r * (tile_h + gap)
        base = base.draw_image(vt, x, y)

    if invert:
        base = 255 - base

    if thumbnail:
        base_thumb = base.thumbnail_image(thumbnail)
        base_thumb.write_to_file(out_path)
    else:
        base.write_to_file(
            out_path,
            compression=1,
            effort=1,
            interlace=False,
            strip=True,
        )


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
                invert=False,
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
            interlace=False, strip=False, bitdepth=1,
        )
