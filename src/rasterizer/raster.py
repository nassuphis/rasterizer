# raster.py
"""
Main rasterizer module - re-exports from submodules for backwards compatibility.

Submodules:
- canvas: Point stamping, radius bucketing, coordinate projection
- formats: NumPy <-> PyVIPS conversions
- compose: Image composition and effects
- export: File output and mosaic creation
"""
from __future__ import annotations

import argparse

# Re-export from canvas
from .canvas import (
    make_disc_offsets,
    build_disc_offset_cache_from_rpx,
    stamp_points,
    bucket_by_radius,
    bucket_by_radius_parallel,
    project_to_canvas,
    render_to_canvas,
    warmup_raster_kernels,
)

# Re-export from formats
from .formats import (
    np_to_vips_rgb_u8,
    np_to_vips_gray_u8,
)

# Re-export from compose
from .compose import (
    add_footer_label,
    add_rounded_passepartout_bilevel_pct,
    pad_to_square,
)

# Re-export from export
from .export import (
    add_suffix_number,
    autolvlcfg,
    save_jpg_rgb,
    save_vips_rgb,
    save_png_rgb,
    save_png_bilevel,
    save_mosaic_png_rgb,
    save_mosaic_png_bilevel,
)


def _main_cli() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    ap.print_help()


if __name__ == "__main__":
    _main_cli()
