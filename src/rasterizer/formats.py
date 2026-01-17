# formats.py
"""
Image format conversions between NumPy arrays and PyVIPS images.

This module handles:
- NumPy array to PyVIPS Image conversion (RGB and grayscale)
- Memory layout handling (HxWx3 vs 3xHxW)
- Data type normalization
"""
from __future__ import annotations

import numpy as np
import pyvips as vips


def np_to_vips_rgb_u8(arr: np.ndarray) -> vips.Image:
    """
    Convert an RGB numpy array to a pyvips Image.

    Accepts:
      - H x W x 3  (preferred)
      - 3 x H x W  (will be transposed to H x W x 3)
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D RGB array, got shape {arr.shape}")

    # Accept either HxWx3 or 3xHxW
    if arr.shape[2] == 3:
        H, W, C = arr.shape
        arr_rgb = arr
    elif arr.shape[0] == 3:
        C, H, W = arr.shape
        arr_rgb = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Last/first dimension must be 3 channels, got shape {arr.shape}")

    if C != 3:
        raise ValueError(f"Expected 3 channels, got {C}")

    if arr_rgb.dtype != np.uint8:
        arr_rgb = np.clip(arr_rgb, 0, 255).astype(np.uint8, copy=False)

    # Ensure contiguous memory for pyvips
    arr_rgb = np.ascontiguousarray(arr_rgb)

    # pyvips expects interleaved RGB, bands=3
    return vips.Image.new_from_memory(arr_rgb.data, W, H, 3, "uchar")


def np_to_vips_gray_u8(arr: np.ndarray) -> vips.Image:
    """Convert a 2D grayscale numpy array to a pyvips Image."""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    H, W = arr.shape
    return vips.Image.new_from_memory(arr.data, W, H, 1, "uchar")
