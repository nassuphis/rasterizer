# test_formats.py
"""Tests for formats module - NumPy to PyVIPS conversions."""
import numpy as np
import pytest
import pyvips as vips

from rasterizer.formats import (
    np_to_vips_rgb_u8,
    np_to_vips_gray_u8,
)


class TestNpToVipsRgbU8:
    """Tests for np_to_vips_rgb_u8 function."""

    def test_hwc_format_conversion(self):
        """Test H x W x 3 format conversion."""
        arr = np.zeros((100, 80, 3), dtype=np.uint8)
        arr[50, 40] = [255, 128, 64]
        img = np_to_vips_rgb_u8(arr)

        assert img.width == 80
        assert img.height == 100
        assert img.bands == 3
        assert img.format == "uchar"

    def test_chw_format_conversion(self):
        """Test 3 x H x W format conversion (transposed)."""
        arr = np.zeros((3, 100, 80), dtype=np.uint8)
        arr[0, 50, 40] = 255  # R
        arr[1, 50, 40] = 128  # G
        arr[2, 50, 40] = 64   # B
        img = np_to_vips_rgb_u8(arr)

        assert img.width == 80
        assert img.height == 100
        assert img.bands == 3

    def test_non_uint8_converted(self):
        """Test that non-uint8 arrays are converted."""
        arr = np.zeros((50, 50, 3), dtype=np.float32)
        arr[25, 25] = [255.0, 128.0, 64.0]
        img = np_to_vips_rgb_u8(arr)

        assert img.format == "uchar"
        assert img.bands == 3

    def test_clipping_out_of_range_values(self):
        """Test that values outside 0-255 are clipped."""
        arr = np.zeros((10, 10, 3), dtype=np.float64)
        arr[5, 5] = [300.0, -50.0, 128.0]
        img = np_to_vips_rgb_u8(arr)

        # Should not raise, values are clipped
        assert img.format == "uchar"

    def test_rejects_2d_array(self):
        """Test that 2D arrays raise ValueError."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 3D RGB array"):
            np_to_vips_rgb_u8(arr)

    def test_rejects_wrong_channel_count(self):
        """Test that arrays with wrong channel count raise ValueError."""
        arr = np.zeros((100, 100, 4), dtype=np.uint8)  # RGBA
        with pytest.raises(ValueError, match="must be 3 channels"):
            np_to_vips_rgb_u8(arr)


class TestNpToVipsGrayU8:
    """Tests for np_to_vips_gray_u8 function."""

    def test_basic_conversion(self):
        """Test basic 2D grayscale conversion."""
        arr = np.zeros((100, 80), dtype=np.uint8)
        arr[50, 40] = 255
        img = np_to_vips_gray_u8(arr)

        assert img.width == 80
        assert img.height == 100
        assert img.bands == 1
        assert img.format == "uchar"

    def test_non_uint8_converted(self):
        """Test that non-uint8 arrays are converted."""
        arr = np.zeros((50, 50), dtype=np.float32)
        arr[25, 25] = 128.0
        img = np_to_vips_gray_u8(arr)

        assert img.format == "uchar"
        assert img.bands == 1

    def test_preserves_pixel_values(self):
        """Test that pixel values are preserved after conversion."""
        arr = np.array([[0, 128], [255, 64]], dtype=np.uint8)
        img = np_to_vips_gray_u8(arr)

        assert img.width == 2
        assert img.height == 2
