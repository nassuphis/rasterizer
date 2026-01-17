# test_canvas.py
"""Tests for canvas module - stamping, bucketing, and coordinate projection."""
import numpy as np
import pytest

from rasterizer.canvas import (
    make_disc_offsets,
    build_disc_offset_cache_from_rpx,
    stamp_points,
    bucket_by_radius,
    bucket_by_radius_parallel,
    project_to_canvas,
    render_to_canvas,
    warmup_raster_kernels,
)


class TestMakeDiscOffsets:
    """Tests for make_disc_offsets function."""

    def test_radius_1_produces_small_disc(self):
        dy, dx = make_disc_offsets(1)
        assert dy.dtype == np.int32
        assert dx.dtype == np.int32
        assert len(dy) == len(dx)
        assert len(dy) > 0

    def test_radius_0_treated_as_1(self):
        dy, dx = make_disc_offsets(0)
        assert len(dy) > 0

    def test_negative_radius_treated_as_1(self):
        dy, dx = make_disc_offsets(-5)
        assert len(dy) > 0

    def test_larger_radius_produces_more_pixels(self):
        dy1, dx1 = make_disc_offsets(2)
        dy2, dx2 = make_disc_offsets(5)
        assert len(dy2) > len(dy1)

    def test_offsets_are_within_radius(self):
        r = 5
        dy, dx = make_disc_offsets(r)
        distances_sq = dy * dy + dx * dx
        assert np.all(distances_sq < r * r)


class TestBuildDiscOffsetCache:
    """Tests for build_disc_offset_cache_from_rpx function."""

    def test_empty_array_returns_empty_cache(self):
        r_px = np.array([], dtype=np.int32)
        cache = build_disc_offset_cache_from_rpx(r_px, rmin=1)
        assert cache == {}

    def test_all_below_rmin_returns_empty_cache(self):
        r_px = np.array([1, 2, 3], dtype=np.int32)
        cache = build_disc_offset_cache_from_rpx(r_px, rmin=5)
        assert cache == {}

    def test_cache_contains_unique_radii(self):
        r_px = np.array([3, 5, 3, 7, 5], dtype=np.int32)
        cache = build_disc_offset_cache_from_rpx(r_px, rmin=1)
        assert set(cache.keys()) == {3, 5, 7}

    def test_negative_radii_converted_to_absolute(self):
        r_px = np.array([-3, -5, 3], dtype=np.int32)
        cache = build_disc_offset_cache_from_rpx(r_px, rmin=1)
        assert 3 in cache
        assert 5 in cache


class TestStampPoints:
    """Tests for stamp_points function."""

    def test_stamps_single_point(self):
        canvas = np.zeros((10, 10), dtype=np.uint8)
        ys = np.array([5], dtype=np.int32)
        xs = np.array([5], dtype=np.int32)
        dy = np.array([0], dtype=np.int32)
        dx = np.array([0], dtype=np.int32)
        stamp_points(canvas, ys, xs, dy, dx, 255)
        assert canvas[5, 5] == 255

    def test_stamps_multiple_points(self):
        canvas = np.zeros((10, 10), dtype=np.uint8)
        ys = np.array([2, 7], dtype=np.int32)
        xs = np.array([2, 7], dtype=np.int32)
        dy = np.array([0], dtype=np.int32)
        dx = np.array([0], dtype=np.int32)
        stamp_points(canvas, ys, xs, dy, dx, 255)
        assert canvas[2, 2] == 255
        assert canvas[7, 7] == 255

    def test_stamps_with_disc_offsets(self):
        canvas = np.zeros((10, 10), dtype=np.uint8)
        ys = np.array([5], dtype=np.int32)
        xs = np.array([5], dtype=np.int32)
        dy = np.array([-1, 0, 1], dtype=np.int32)
        dx = np.array([0, 0, 0], dtype=np.int32)
        stamp_points(canvas, ys, xs, dy, dx, 255)
        assert canvas[4, 5] == 255
        assert canvas[5, 5] == 255
        assert canvas[6, 5] == 255

    def test_clips_to_canvas_bounds(self):
        canvas = np.zeros((10, 10), dtype=np.uint8)
        ys = np.array([0], dtype=np.int32)
        xs = np.array([0], dtype=np.int32)
        dy = np.array([-1, 0, 1], dtype=np.int32)
        dx = np.array([-1, 0, 1], dtype=np.int32)
        # Should not raise, out-of-bounds are silently ignored
        stamp_points(canvas, ys, xs, dy, dx, 255)
        assert canvas[0, 0] == 255
        assert canvas[1, 1] == 255

    def test_custom_stamp_value(self):
        canvas = np.zeros((10, 10), dtype=np.uint8)
        ys = np.array([5], dtype=np.int32)
        xs = np.array([5], dtype=np.int32)
        dy = np.array([0], dtype=np.int32)
        dx = np.array([0], dtype=np.int32)
        stamp_points(canvas, ys, xs, dy, dx, 128)
        assert canvas[5, 5] == 128


class TestBucketByRadius:
    """Tests for bucket_by_radius function."""

    def test_empty_array(self):
        r_px = np.array([], dtype=np.int32)
        order, r_vals, starts, counts = bucket_by_radius(r_px, 1, 10)
        assert len(order) == 0
        assert len(r_vals) == 0

    def test_all_outside_range(self):
        r_px = np.array([1, 2, 3], dtype=np.int32)
        order, r_vals, starts, counts = bucket_by_radius(r_px, 10, 20)
        assert len(order) == 0

    def test_basic_bucketing(self):
        r_px = np.array([1, 2, 1, 3, 2], dtype=np.int32)
        order, r_vals, starts, counts = bucket_by_radius(r_px, 1, 3)

        # Should have 3 unique radii
        assert len(r_vals) == 3
        assert list(r_vals) == [1, 2, 3]

        # Counts should match
        assert counts[0] == 2  # two 1s
        assert counts[1] == 2  # two 2s
        assert counts[2] == 1  # one 3

    def test_order_contains_original_indices(self):
        r_px = np.array([3, 1, 2], dtype=np.int32)
        order, r_vals, starts, counts = bucket_by_radius(r_px, 1, 3)
        assert set(order) == {0, 1, 2}


class TestBucketByRadiusParallel:
    """Tests for bucket_by_radius_parallel function."""

    def test_empty_array(self):
        r_px = np.array([], dtype=np.int32)
        order, r_vals, starts, counts = bucket_by_radius_parallel(r_px, 1, 10)
        assert len(order) == 0

    def test_matches_sequential_version(self):
        np.random.seed(42)
        r_px = np.random.randint(1, 10, size=1000, dtype=np.int32)

        order1, r_vals1, starts1, counts1 = bucket_by_radius(r_px, 1, 9)
        order2, r_vals2, starts2, counts2 = bucket_by_radius_parallel(r_px, 1, 9)

        np.testing.assert_array_equal(r_vals1, r_vals2)
        np.testing.assert_array_equal(counts1, counts2)
        np.testing.assert_array_equal(starts1, starts2)


class TestProjectToCanvas:
    """Tests for project_to_canvas function."""

    def test_empty_array(self):
        z = np.array([], dtype=np.complex128)
        px, py = project_to_canvas(z, pix=100, margin_frac=0.1)
        assert len(px) == 0
        assert len(py) == 0

    def test_center_point_maps_consistently(self):
        """Test that a single point at origin maps to a valid pixel."""
        z = np.array([0 + 0j])
        px, py = project_to_canvas(z, pix=100, margin_frac=0.0)
        # With a single point at origin and no margin, half=0, span=1 (fallback)
        # The point maps to (0,0) due to the formula with zero extent
        assert 0 <= px[0] < 100
        assert 0 <= py[0] < 100

    def test_symmetric_points_span_canvas(self):
        """Test that symmetric points around origin span the canvas."""
        z = np.array([-1 + 0j, 1 + 0j])  # Points at -1 and +1
        px, py = project_to_canvas(z, pix=100, margin_frac=0.0)
        # These should map to opposite edges of the canvas
        assert px[0] == 0 or px[0] == 99
        assert px[1] == 0 or px[1] == 99

    def test_output_within_canvas_bounds(self):
        z = np.array([1 + 1j, -1 - 1j, 0.5 - 0.5j])
        pix = 100
        px, py = project_to_canvas(z, pix=pix, margin_frac=0.1)
        assert np.all(px >= 0)
        assert np.all(px < pix)
        assert np.all(py >= 0)
        assert np.all(py < pix)

    def test_output_dtype_is_int32(self):
        z = np.array([0.5 + 0.5j])
        px, py = project_to_canvas(z, pix=100, margin_frac=0.1)
        assert px.dtype == np.int32
        assert py.dtype == np.int32


class TestRenderToCanvas:
    """Tests for render_to_canvas function."""

    def test_renders_points(self):
        z = np.array([0 + 0j])
        canvas = render_to_canvas(z, pix=100, margin_frac=0.1)
        assert canvas.shape == (100, 100)
        assert canvas.dtype == np.uint8
        assert np.sum(canvas) > 0  # At least one pixel should be set

    def test_empty_array_returns_blank_canvas(self):
        z = np.array([], dtype=np.complex128)
        canvas = render_to_canvas(z, pix=50, margin_frac=0.0)
        assert canvas.shape == (50, 50)
        assert np.sum(canvas) == 0


class TestWarmupRasterKernels:
    """Tests for warmup_raster_kernels function."""

    def test_warmup_completes_without_error(self):
        # Should not raise any exception
        warmup_raster_kernels()
