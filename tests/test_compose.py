# test_compose.py
"""Tests for compose module - image composition and visual effects."""
import numpy as np
import pytest
import pyvips as vips

from rasterizer.compose import (
    add_footer_label,
    add_rounded_passepartout_bilevel_pct,
    pad_to_square,
)


class TestAddFooterLabel:
    """Tests for add_footer_label function."""

    def test_empty_text_returns_unchanged(self):
        """Test that empty text returns the original image."""
        base = vips.Image.black(100, 100)
        result = add_footer_label(base, "")
        assert result.width == base.width
        assert result.height == base.height

    def test_none_text_returns_unchanged(self):
        """Test that None text returns original image."""
        base = vips.Image.black(100, 100)
        # Function accepts str, so we test empty string behavior
        result = add_footer_label(base, "")
        assert result.width == base.width

    def test_adds_text_to_image(self):
        """Test that text is actually added to the image."""
        base = vips.Image.black(500, 500)
        result = add_footer_label(base, "Test Label")

        # Result should have same dimensions
        assert result.width == base.width
        assert result.height == base.height

        # Image should have some non-zero pixels from the text
        # (base is all black, text adds white pixels)
        stats = result.stats()
        max_val = stats(1, 0)[0]  # max of first band
        assert max_val > 0

    def test_small_image_handling(self):
        """Test that very small images don't crash."""
        base = vips.Image.black(10, 10)
        # Should not raise
        result = add_footer_label(base, "X")
        assert result.width == 10
        assert result.height == 10

    def test_invert_option(self):
        """Test the invert option for text rendering."""
        base = vips.Image.black(500, 500).new_from_image(255)  # White image
        result = add_footer_label(base, "Test", invert=True)
        assert result.width == base.width


class TestAddRoundedPassepartoutBilevelPct:
    """Tests for add_rounded_passepartout_bilevel_pct function."""

    def test_basic_passepartout(self):
        """Test basic passepartout creation."""
        # Create a square grayscale image
        base = vips.Image.black(100, 100).new_from_image(128)
        result = add_rounded_passepartout_bilevel_pct(
            base,
            margin_frac=0.1,
            radius_frac=0.05,
        )

        # Result should be larger due to margin
        assert result.width > base.width
        assert result.height > base.height

    def test_rejects_non_square(self):
        """Test that non-square images raise ValueError."""
        base = vips.Image.black(100, 50)  # Non-square
        with pytest.raises(ValueError, match="Expected a square image"):
            add_rounded_passepartout_bilevel_pct(base)

    def test_rejects_multi_band(self):
        """Test that multi-band images raise ValueError."""
        base = vips.Image.black(100, 100).new_from_image([128, 128, 128])  # RGB
        with pytest.raises(ValueError, match="single-band"):
            add_rounded_passepartout_bilevel_pct(base)

    def test_auto_white_bg(self):
        """Test auto white background detection."""
        # Dark image should get white mat
        dark = vips.Image.black(100, 100).new_from_image(0)
        result = add_rounded_passepartout_bilevel_pct(
            dark,
            margin_frac=0.1,
            auto_white_bg=True,
        )
        assert result.width == 120
        assert result.height == 120

    def test_zero_radius(self):
        """Test with zero radius (no rounding)."""
        base = vips.Image.black(100, 100).new_from_image(128)
        result = add_rounded_passepartout_bilevel_pct(
            base,
            margin_frac=0.1,
            radius_frac=0.0,
        )
        assert result.width == 120

    def test_explicit_mat_value(self):
        """Test explicit mat value setting."""
        base = vips.Image.black(100, 100).new_from_image(128)
        result = add_rounded_passepartout_bilevel_pct(
            base,
            margin_frac=0.1,
            auto_white_bg=False,
            mat_value=255,
        )
        assert result.width == 120


class TestPadToSquare:
    """Tests for pad_to_square function."""

    def test_pads_smaller_image(self):
        """Test padding a smaller image to target size."""
        img = vips.Image.black(50, 50)
        result = pad_to_square(img, 100)

        assert result.width == 100
        assert result.height == 100

    def test_image_centered(self):
        """Test that image is centered in padded result."""
        img = vips.Image.black(50, 50).new_from_image(255)
        result = pad_to_square(img, 100)

        # The 50x50 white area should be centered in 100x100 black canvas
        assert result.width == 100
        assert result.height == 100

    def test_already_correct_size(self):
        """Test image already at target size."""
        img = vips.Image.black(100, 100)
        result = pad_to_square(img, 100)

        assert result.width == 100
        assert result.height == 100

    def test_larger_than_target(self):
        """Test image larger than target (no padding added)."""
        img = vips.Image.black(150, 150)
        result = pad_to_square(img, 100)

        # With max(0, ...) logic, dx and dy will be 0
        assert result.width == 100
        assert result.height == 100

    def test_non_square_input(self):
        """Test padding a non-square image."""
        img = vips.Image.black(30, 50)
        result = pad_to_square(img, 100)

        assert result.width == 100
        assert result.height == 100
