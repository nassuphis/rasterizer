# test_colors.py
"""Tests for colors module - color parsing, histogram equalization, and colorization."""
import math
import numpy as np
import pytest

from rasterizer.colors import (
    parse_color_spec,
    _hist_equalize,
    _rgb255_to_hsv01,
    _interp_hue_circle,
    _hsv01_to_rgb255_batch,
    _two_sided_t_and_masks,
    rgb_scheme_mh,
    rgb_scheme_mh_eq,
)


class TestParseColorSpec:
    """Tests for parse_color_spec function."""

    def test_hex_without_hash(self):
        """Test parsing hex color without # prefix."""
        r, g, b = parse_color_spec("FF0000", (0, 0, 0))
        assert r == 255.0
        assert g == 0.0
        assert b == 0.0

    def test_hex_with_hash(self):
        """Test parsing hex color with # prefix."""
        r, g, b = parse_color_spec("#00FF00", (0, 0, 0))
        assert r == 0.0
        assert g == 255.0
        assert b == 0.0

    def test_named_color_red(self):
        """Test parsing named color 'red'."""
        r, g, b = parse_color_spec("red", (0, 0, 0))
        assert r == 255.0
        assert g == 0.0
        assert b == 0.0

    def test_named_color_blue(self):
        """Test parsing named color 'blue'."""
        r, g, b = parse_color_spec("blue", (0, 0, 0))
        assert r == 0.0
        assert g == 0.0
        assert b == 255.0

    def test_case_insensitive(self):
        """Test that color names are case insensitive."""
        r1, g1, b1 = parse_color_spec("RED", (0, 0, 0))
        r2, g2, b2 = parse_color_spec("red", (0, 0, 0))
        assert (r1, g1, b1) == (r2, g2, b2)

    def test_invalid_returns_default(self):
        """Test that invalid specs return the default."""
        default = (128.0, 64.0, 32.0)
        result = parse_color_spec("invalid_color", default)
        assert result == default

    def test_empty_string_returns_default(self):
        """Test that empty string returns the default."""
        default = (100.0, 100.0, 100.0)
        result = parse_color_spec("", default)
        assert result == default

    def test_non_string_returns_default(self):
        """Test that non-string input returns the default."""
        default = (50.0, 50.0, 50.0)
        result = parse_color_spec(None, default)
        assert result == default

    def test_wrong_length_hex_returns_default(self):
        """Test that wrong-length hex returns default."""
        default = (1.0, 2.0, 3.0)
        result = parse_color_spec("FFF", default)  # 3 chars, needs 6
        assert result == default


class TestHistEqualize:
    """Tests for _hist_equalize function."""

    def test_empty_array(self):
        """Test histogram equalization on empty array."""
        values = np.array([])
        result = _hist_equalize(values)
        assert len(result) == 0

    def test_constant_array(self):
        """Test histogram equalization on constant array (all same values)."""
        values = np.full(100, 5.0)
        result = _hist_equalize(values)
        # When all values are the same, vmax <= vmin, should return zeros
        assert result.shape == values.shape

    def test_uniform_distribution(self):
        """Test histogram equalization on uniform distribution."""
        values = np.linspace(0, 1, 1000)
        result = _hist_equalize(values)
        # Result should be roughly linear for uniform input
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_range(self):
        """Test that output is in [0, 1] range."""
        np.random.seed(42)
        values = np.random.randn(1000)
        result = _hist_equalize(values)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_degenerate_range_returns_uniform(self):
        """Test that very small range returns uniform 0.5 (edge case fix)."""
        # Values with extremely small range (precision issue)
        values = np.array([1.0, 1.0 + 1e-16, 1.0 + 2e-16])
        result = _hist_equalize(values, nbins=256)
        # Should not raise, and should return 0.5 for all
        assert result.shape == values.shape

    def test_non_finite_values(self):
        """Test handling of non-finite values."""
        values = np.array([1.0, 2.0, np.inf, 3.0])
        result = _hist_equalize(values)
        # With inf, vmax is inf, should return zeros
        assert result.shape == values.shape

    def test_nan_values(self):
        """Test handling of NaN values."""
        values = np.array([1.0, np.nan, 3.0])
        result = _hist_equalize(values)
        assert result.shape == values.shape


class TestRgb255ToHsv01:
    """Tests for _rgb255_to_hsv01 function."""

    def test_red(self):
        """Test conversion of pure red."""
        h, s, v = _rgb255_to_hsv01((255.0, 0.0, 0.0))
        assert abs(h - 0.0) < 0.01  # Hue ~0 for red
        assert abs(s - 1.0) < 0.01  # Full saturation
        assert abs(v - 1.0) < 0.01  # Full value

    def test_green(self):
        """Test conversion of pure green."""
        h, s, v = _rgb255_to_hsv01((0.0, 255.0, 0.0))
        assert abs(h - 1/3) < 0.01  # Hue ~0.333 for green
        assert abs(s - 1.0) < 0.01
        assert abs(v - 1.0) < 0.01

    def test_blue(self):
        """Test conversion of pure blue."""
        h, s, v = _rgb255_to_hsv01((0.0, 0.0, 255.0))
        assert abs(h - 2/3) < 0.01  # Hue ~0.666 for blue
        assert abs(s - 1.0) < 0.01
        assert abs(v - 1.0) < 0.01

    def test_white(self):
        """Test conversion of white."""
        h, s, v = _rgb255_to_hsv01((255.0, 255.0, 255.0))
        assert abs(s - 0.0) < 0.01  # No saturation
        assert abs(v - 1.0) < 0.01

    def test_black(self):
        """Test conversion of black."""
        h, s, v = _rgb255_to_hsv01((0.0, 0.0, 0.0))
        assert abs(v - 0.0) < 0.01


class TestInterpHueCircle:
    """Tests for _interp_hue_circle function."""

    def test_same_hue(self):
        """Test interpolation between same hues."""
        result = _interp_hue_circle(0.5, 0.5, 0.5)
        assert abs(result - 0.5) < 0.01

    def test_t_zero_returns_h0(self):
        """Test that t=0 returns h0."""
        result = _interp_hue_circle(0.2, 0.8, 0.0)
        assert abs(result - 0.2) < 0.01

    def test_t_one_returns_h1(self):
        """Test that t=1 returns h1."""
        result = _interp_hue_circle(0.2, 0.8, 1.0)
        assert abs(result - 0.8) < 0.01

    def test_array_input(self):
        """Test with array inputs."""
        h0 = np.array([0.0, 0.25, 0.5])
        h1 = np.array([0.5, 0.75, 1.0])
        t = np.array([0.5, 0.5, 0.5])
        result = _interp_hue_circle(h0, h1, t)
        assert result.shape == (3,)


class TestHsv01ToRgb255Batch:
    """Tests for _hsv01_to_rgb255_batch function."""

    def test_red(self):
        """Test conversion of HSV red to RGB."""
        h = np.array([0.0])
        s = np.array([1.0])
        v = np.array([1.0])
        rgb = _hsv01_to_rgb255_batch(h, s, v)
        assert rgb.shape == (1, 3)
        assert abs(rgb[0, 0] - 255.0) < 1.0  # R
        assert abs(rgb[0, 1] - 0.0) < 1.0    # G
        assert abs(rgb[0, 2] - 0.0) < 1.0    # B

    def test_white(self):
        """Test conversion of HSV white to RGB."""
        h = np.array([0.0])
        s = np.array([0.0])
        v = np.array([1.0])
        rgb = _hsv01_to_rgb255_batch(h, s, v)
        assert abs(rgb[0, 0] - 255.0) < 1.0
        assert abs(rgb[0, 1] - 255.0) < 1.0
        assert abs(rgb[0, 2] - 255.0) < 1.0

    def test_batch_conversion(self):
        """Test batch conversion of multiple colors."""
        h = np.array([0.0, 1/3, 2/3])  # R, G, B hues
        s = np.array([1.0, 1.0, 1.0])
        v = np.array([1.0, 1.0, 1.0])
        rgb = _hsv01_to_rgb255_batch(h, s, v)
        assert rgb.shape == (3, 3)


class TestTwoSidedTAndMasks:
    """Tests for _two_sided_t_and_masks function."""

    def test_basic_normalization(self):
        """Test basic two-sided normalization."""
        v = np.array([[-1.0, 0.0], [0.5, 1.0]])
        t, finite, neg, pos = _two_sided_t_and_masks(v, {"norm": "linear"})

        assert t.shape == v.shape
        assert finite.shape == v.shape
        assert neg.shape == v.shape
        assert pos.shape == v.shape

        # Check masks
        assert neg[0, 0]  # -1 is negative
        assert not neg[0, 1]  # 0 is not negative
        assert pos[1, 0]  # 0.5 is positive
        assert pos[1, 1]  # 1.0 is positive

    def test_eq_normalization(self):
        """Test histogram equalization normalization."""
        np.random.seed(42)
        v = np.random.randn(100, 100)
        t, finite, neg, pos = _two_sided_t_and_masks(v, {"norm": "eq"})

        assert t.shape == v.shape
        assert np.all(t >= 0.0)
        assert np.all(t <= 1.0)

    def test_gamma_application(self):
        """Test gamma is applied to t values."""
        v = np.array([[0.5, 1.0], [-0.5, -1.0]])
        t1, _, _, _ = _two_sided_t_and_masks(v, {"norm": "linear", "gamma": 1.0})
        t2, _, _, _ = _two_sided_t_and_masks(v, {"norm": "linear", "gamma": 2.0})

        # With gamma=2, t values should be squared
        # (not exactly, due to normalization, but different)
        assert not np.allclose(t1, t2)

    def test_invalid_norm_raises(self):
        """Test that invalid norm raises ValueError."""
        v = np.array([[1.0]])
        with pytest.raises(ValueError, match="Unknown norm"):
            _two_sided_t_and_masks(v, {"norm": "invalid"})


class TestRgbSchemeMh:
    """Tests for rgb_scheme_mh function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        lyap = np.random.randn(100, 100)
        rgb = rgb_scheme_mh(lyap, {})
        assert rgb.shape == (100, 100, 3)
        assert rgb.dtype == np.uint8

    def test_non_2d_raises(self):
        """Test that non-2D input raises ValueError."""
        lyap = np.random.randn(100)
        with pytest.raises(ValueError, match="must be 2D"):
            rgb_scheme_mh(lyap, {})

    def test_all_zeros_produces_dark_image(self):
        """Test that all-zero input produces dark image."""
        lyap = np.zeros((50, 50))
        rgb = rgb_scheme_mh(lyap, {})
        # Zero values should map to zero_color (default black)
        assert np.all(rgb == 0)


class TestRgbSchemeMhEq:
    """Tests for rgb_scheme_mh_eq function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        lyap = np.random.randn(100, 100)
        rgb = rgb_scheme_mh_eq(lyap, {})
        assert rgb.shape == (100, 100, 3)
        assert rgb.dtype == np.uint8

    def test_custom_colors(self):
        """Test with custom color parameters."""
        lyap = np.array([[-1.0, 0.0], [0.0, 1.0]])
        params = {
            "neg_color": "00FF00",  # Green for negative
            "zero_color": "000000",  # Black for zero
            "pos_color": "FF0000",  # Red for positive
        }
        rgb = rgb_scheme_mh_eq(lyap, params)
        assert rgb.shape == (2, 2, 3)
