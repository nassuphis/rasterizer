# test_export.py
"""Tests for export module - file saving and mosaic creation."""
import os
import tempfile
import numpy as np
import pytest

from rasterizer.export import (
    add_suffix_number,
    save_png_rgb,
    save_png_bilevel,
    save_mosaic_png_rgb,
    save_mosaic_png_bilevel,
)


class TestAddSuffixNumber:
    """Tests for add_suffix_number function."""

    def test_basic_suffix(self):
        """Test basic suffix addition."""
        result = add_suffix_number("/path/to/file.png", 42)
        assert result == "/path/to/file_00042.png"

    def test_custom_width(self):
        """Test custom width for suffix."""
        result = add_suffix_number("image.jpg", 7, width=3)
        assert result == "image_007.jpg"

    def test_large_number(self):
        """Test with large number."""
        result = add_suffix_number("test.png", 99999, width=5)
        assert result == "test_99999.png"

    def test_preserves_extension(self):
        """Test that file extension is preserved."""
        result = add_suffix_number("photo.jpeg", 1)
        assert result.endswith(".jpeg")

    def test_no_extension(self):
        """Test file without extension."""
        result = add_suffix_number("filename", 5)
        assert result == "filename_00005"


class TestSavePngRgb:
    """Tests for save_png_rgb function."""

    def test_saves_rgb_array(self):
        """Test saving an RGB numpy array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.zeros((100, 100, 3), dtype=np.uint8)
            arr[50, 50] = [255, 128, 64]
            out_path = os.path.join(tmpdir, "test.png")

            save_png_rgb(arr, out_path)

            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0

    def test_with_invert(self):
        """Test saving with invert option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.zeros((50, 50, 3), dtype=np.uint8)
            out_path = os.path.join(tmpdir, "inverted.png")

            save_png_rgb(arr, out_path, invert=True)

            assert os.path.exists(out_path)


class TestSavePngBilevel:
    """Tests for save_png_bilevel function."""

    def test_saves_bilevel_array(self):
        """Test saving a bilevel numpy array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.zeros((100, 100), dtype=np.uint8)
            arr[25:75, 25:75] = 255
            out_path = os.path.join(tmpdir, "bilevel.png")

            save_png_bilevel(arr, out_path, invert=False)

            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0

    def test_with_invert(self):
        """Test saving with invert option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.zeros((50, 50), dtype=np.uint8)
            out_path = os.path.join(tmpdir, "inverted.png")

            save_png_bilevel(arr, out_path, invert=True)

            assert os.path.exists(out_path)

    def test_non_uint8_converted(self):
        """Test that non-uint8 arrays are converted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.zeros((50, 50), dtype=np.float32)
            arr[20:30, 20:30] = 255.0
            out_path = os.path.join(tmpdir, "float_bilevel.png")

            save_png_bilevel(arr, out_path, invert=False)

            assert os.path.exists(out_path)


class TestSaveMosaicPngRgb:
    """Tests for save_mosaic_png_rgb function."""

    def test_creates_mosaic(self):
        """Test creating a basic RGB mosaic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tiles = [
                np.full((50, 50, 3), [255, 0, 0], dtype=np.uint8),
                np.full((50, 50, 3), [0, 255, 0], dtype=np.uint8),
                np.full((50, 50, 3), [0, 0, 255], dtype=np.uint8),
                np.full((50, 50, 3), [255, 255, 0], dtype=np.uint8),
            ]
            out_path = os.path.join(tmpdir, "mosaic.png")

            save_mosaic_png_rgb(
                tiles,
                titles=None,
                cols=2,
                gap=5,
                out_path=out_path,
            )

            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0

    def test_empty_tiles_raises(self):
        """Test that empty tile list raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "empty.png")
            with pytest.raises(ValueError, match="No tiles"):
                save_mosaic_png_rgb([], titles=None, cols=2, gap=5, out_path=out_path)

    def test_with_titles(self):
        """Test mosaic with titles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tiles = [
                np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8),
                np.full((100, 100, 3), [0, 255, 0], dtype=np.uint8),
            ]
            out_path = os.path.join(tmpdir, "titled_mosaic.png")

            save_mosaic_png_rgb(
                tiles,
                titles=["Red", "Green"],
                cols=2,
                gap=5,
                out_path=out_path,
            )

            assert os.path.exists(out_path)

    def test_single_title_repeated(self):
        """Test that single title is repeated for all tiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tiles = [
                np.full((50, 50, 3), 128, dtype=np.uint8),
                np.full((50, 50, 3), 128, dtype=np.uint8),
            ]
            out_path = os.path.join(tmpdir, "single_title.png")

            save_mosaic_png_rgb(
                tiles,
                titles=["Same Title"],
                cols=2,
                gap=5,
                out_path=out_path,
            )

            assert os.path.exists(out_path)

    def test_mismatched_titles_raises(self):
        """Test that mismatched title count raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tiles = [
                np.full((50, 50, 3), 128, dtype=np.uint8),
                np.full((50, 50, 3), 128, dtype=np.uint8),
                np.full((50, 50, 3), 128, dtype=np.uint8),
            ]
            out_path = os.path.join(tmpdir, "mismatch.png")

            with pytest.raises(ValueError, match="titles"):
                save_mosaic_png_rgb(
                    tiles,
                    titles=["A", "B"],  # 2 titles for 3 tiles
                    cols=2,
                    gap=5,
                    out_path=out_path,
                )


class TestSaveMosaicPngBilevel:
    """Tests for save_mosaic_png_bilevel function."""

    def test_creates_bilevel_mosaic(self):
        """Test creating a basic bilevel mosaic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tiles = [
                np.full((50, 50), 0, dtype=np.uint8),
                np.full((50, 50), 255, dtype=np.uint8),
            ]
            out_path = os.path.join(tmpdir, "bilevel_mosaic.png")

            save_mosaic_png_bilevel(
                tiles,
                titles=None,
                cols=2,
                gap=5,
                out_path=out_path,
                invert=False,
            )

            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0

    def test_empty_tiles_raises(self):
        """Test that empty tile list raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "empty.png")
            with pytest.raises(ValueError, match="No tiles"):
                save_mosaic_png_bilevel(
                    [],
                    titles=None,
                    cols=2,
                    gap=5,
                    out_path=out_path,
                    invert=False,
                )

    def test_with_invert(self):
        """Test bilevel mosaic with invert option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tiles = [
                np.full((50, 50), 255, dtype=np.uint8),
            ]
            out_path = os.path.join(tmpdir, "inverted_bilevel.png")

            save_mosaic_png_bilevel(
                tiles,
                titles=None,
                cols=1,
                gap=0,
                out_path=out_path,
                invert=True,
            )

            assert os.path.exists(out_path)

    def test_different_tile_sizes_padded(self):
        """Test that different tile sizes are handled (padded)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tiles = [
                np.full((50, 50), 255, dtype=np.uint8),
                np.full((40, 40), 255, dtype=np.uint8),  # Different size
            ]
            out_path = os.path.join(tmpdir, "mixed_sizes.png")

            # Should not raise - tiles are padded to match
            save_mosaic_png_bilevel(
                tiles,
                titles=None,
                cols=2,
                gap=5,
                out_path=out_path,
                invert=False,
            )

            assert os.path.exists(out_path)
