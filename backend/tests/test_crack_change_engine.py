"""
Unit tests for CrackChangeEngine:
Mask warping, spatial sets, tolerance buffers, connected components, and metrics.
"""
import base64
import cv2
import numpy as np
import pytest

from app.services.crack_change_engine import CrackChangeEngine, crack_change_engine


class TestCrackChangeEngineBasics:
    """Basic unit operations, warping, and base64 round-trip."""

    def test_warp_before_mask_identity(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:50] = 255
        H = np.eye(3, dtype=np.float32)
        warped = crack_change_engine.warp_before_mask(mask, H, (100, 100))
        assert np.array_equal(mask, warped)

    def test_warp_before_mask_translation(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:30, 20:30] = 255
        # Shift x by +10, y by +15
        H = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 15.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        warped = crack_change_engine.warp_before_mask(mask, H, (100, 100))
        assert np.count_nonzero(warped > 0) == 100
        assert warped[35, 30] == 255
        assert warped[25, 25] == 0

    def test_warp_before_mask_none_returns_zeros(self):
        mask = np.full((100, 100), 255, dtype=np.uint8)
        warped = crack_change_engine.warp_before_mask(mask, None, (120, 80))
        assert warped.shape == (120, 80)
        assert np.count_nonzero(warped) == 0

    def test_base64_mask_roundtrip(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        b64 = crack_change_engine.encode_image_png(mask)
        decoded = crack_change_engine.decode_base64_mask(b64)
        assert np.array_equal(mask, decoded)


class TestSpatialMaskComparison:
    """Deterministic spatial mask operations: persistent, new, no-longer-observed."""

    def test_both_masks_empty(self):
        shape = (100, 100)
        B = np.zeros(shape, dtype=np.uint8)
        A = np.zeros(shape, dtype=np.uint8)
        V = np.full(shape, 255, dtype=np.uint8)

        res = crack_change_engine.analyze_crack_change(B, A, V, tolerance_px=2)
        assert res["before_crack_pixels"] == 0
        assert res["after_crack_pixels"] == 0
        assert res["persistent_pixels"] == 0
        assert res["newly_observed_pixels"] == 0
        assert res["no_longer_observed_pixels"] == 0
        assert res["raw_change_pixels"] == 0
        assert res["before_crack_coverage_percentage"] == 0.0
        assert res["after_crack_coverage_percentage"] == 0.0
        assert res["coverage_change_percentage_points"] == 0.0
        assert res["potential_new_components"] == 0
        assert res["potential_expanded_components"] == 0

    def test_before_empty_after_nonempty(self):
        shape = (100, 100)
        B = np.zeros(shape, dtype=np.uint8)
        A = np.zeros(shape, dtype=np.uint8)
        A[30:50, 40:45] = 255  # 20 * 5 = 100 pixels
        V = np.full(shape, 255, dtype=np.uint8)

        res = crack_change_engine.analyze_crack_change(B, A, V, tolerance_px=2)
        assert res["before_crack_pixels"] == 0
        assert res["after_crack_pixels"] == 100
        assert res["persistent_pixels"] == 0
        assert res["newly_observed_pixels"] == 100
        assert res["no_longer_observed_pixels"] == 0
        assert res["raw_change_pixels"] == 100
        assert res["potential_new_components"] == 1
        assert res["potential_expanded_components"] == 0
        assert res["coverage_change_percentage_points"] == 1.0  # 100 / 10000 * 100

    def test_before_nonempty_after_empty(self):
        shape = (100, 100)
        B = np.zeros(shape, dtype=np.uint8)
        B[10:30, 10:15] = 255  # 100 px
        A = np.zeros(shape, dtype=np.uint8)
        V = np.full(shape, 255, dtype=np.uint8)

        res = crack_change_engine.analyze_crack_change(B, A, V, tolerance_px=2)
        assert res["before_crack_pixels"] == 100
        assert res["after_crack_pixels"] == 0
        assert res["persistent_pixels"] == 0
        assert res["newly_observed_pixels"] == 0
        assert res["no_longer_observed_pixels"] == 100
        assert res["raw_change_pixels"] == 100
        assert res["no_longer_observed_components"] == 1
        assert res["coverage_change_percentage_points"] == -1.0

    def test_identical_crack_masks(self):
        shape = (100, 100)
        B = np.zeros(shape, dtype=np.uint8)
        B[25:75, 48:52] = 255  # 50 * 4 = 200 px
        A = B.copy()
        V = np.full(shape, 255, dtype=np.uint8)

        res = crack_change_engine.analyze_crack_change(B, A, V, tolerance_px=2)
        assert res["before_crack_pixels"] == 200
        assert res["after_crack_pixels"] == 200
        assert res["persistent_pixels"] == 200
        assert res["newly_observed_pixels"] == 0
        assert res["no_longer_observed_pixels"] == 0
        assert res["raw_change_pixels"] == 0
        assert res["coverage_change_percentage_points"] == 0.0
        assert res["persistent_components"] == 1
        assert res["potential_new_components"] == 0

    def test_new_disconnected_component(self):
        shape = (100, 100)
        B = np.zeros(shape, dtype=np.uint8)
        B[10:30, 10:15] = 255  # Component 1 (100 px)

        A = B.copy()
        A[60:80, 70:75] = 255  # Component 2 (100 px)
        V = np.full(shape, 255, dtype=np.uint8)

        res = crack_change_engine.analyze_crack_change(B, A, V, tolerance_px=2)
        assert res["persistent_pixels"] == 100
        assert res["newly_observed_pixels"] == 100
        assert res["potential_new_components"] == 1
        assert res["persistent_components"] == 1

    def test_crack_expansion_of_existing_component(self):
        shape = (100, 100)
        B = np.zeros(shape, dtype=np.uint8)
        B[20:40, 50:53] = 255  # 20 * 3 = 60 px

        A = np.zeros(shape, dtype=np.uint8)
        A[20:65, 50:53] = 255  # 45 * 3 = 135 px (lengthened downwards by 25*3 = 75 px, >15% and >20px)
        V = np.full(shape, 255, dtype=np.uint8)

        res = crack_change_engine.analyze_crack_change(B, A, V, tolerance_px=2)
        assert res["persistent_pixels"] == 60
        assert res["newly_observed_pixels"] == 75
        assert res["potential_expanded_components"] == 1
        assert res["potential_new_components"] == 0

    def test_one_pixel_alignment_shift_tolerance(self):
        """A 1-pixel shift along the normal is absorbed by tolerance buffer tau=2."""
        shape = (100, 100)
        B = np.zeros(shape, dtype=np.uint8)
        B[20:60, 50:52] = 255  # vertical crack w=2

        A = np.zeros(shape, dtype=np.uint8)
        A[20:60, 51:53] = 255  # shifted 1 pixel to the right
        V = np.full(shape, 255, dtype=np.uint8)

        res_raw = crack_change_engine.analyze_crack_change(B, A, V, tolerance_px=0)
        # Without tolerance, 40 px overlap, 40 px new, 40 px no-longer-observed
        assert res_raw["newly_observed_pixels"] == 40
        assert res_raw["no_longer_observed_pixels"] == 40
        assert res_raw["tolerance_adjusted_new_pixels"] == 40

        res_tol = crack_change_engine.analyze_crack_change(B, A, V, tolerance_px=1)
        # With tolerance=1 px, the 1-px shift is absorbed
        assert res_tol["tolerance_adjusted_new_pixels"] == 0
        assert res_tol["tolerance_adjusted_no_longer_observed_pixels"] == 0

    def test_boundary_touching_cracks(self):
        shape = (100, 100)
        B = np.zeros(shape, dtype=np.uint8)
        B[0:50, 0:2] = 255  # Left edge touch
        A = np.zeros(shape, dtype=np.uint8)
        A[0:50, 0:2] = 255
        V = np.full(shape, 255, dtype=np.uint8)

        res = crack_change_engine.analyze_crack_change(B, A, V, tolerance_px=2)
        assert res["before_crack_pixels"] == 100
        assert res["persistent_pixels"] == 100

    def test_invalid_overlap_pixels_excluded(self):
        shape = (100, 100)
        B = np.zeros(shape, dtype=np.uint8)
        B[10:30, 10:20] = 255  # 200 px in valid area
        B[70:90, 70:80] = 255  # 200 px in INVALID area

        A = B.copy()
        V = np.zeros(shape, dtype=np.uint8)
        V[0:50, 0:50] = 255  # Top-left quadrant valid (2500 px)

        res = crack_change_engine.analyze_crack_change(B, A, V, tolerance_px=2)
        assert res["valid_overlap_pixels"] == 2500
        assert res["before_crack_pixels"] == 200  # Only the 200 px inside valid ROI
        assert res["after_crack_pixels"] == 200
        assert res["persistent_pixels"] == 200

    def test_non_square_dimensions_landscape(self):
        shape = (416, 700)
        B = np.zeros(shape, dtype=np.uint8)
        B[100:150, 200:250] = 255
        A = B.copy()
        V = np.full(shape, 255, dtype=np.uint8)

        res = crack_change_engine.analyze_crack_change(B, A, V, tolerance_px=2)
        assert res["valid_overlap_pixels"] == 416 * 700
        assert res["persistent_pixels"] == 2500

    def test_zero_division_guard(self):
        shape = (100, 100)
        B = np.zeros(shape, dtype=np.uint8)
        A = np.zeros(shape, dtype=np.uint8)
        V = np.zeros(shape, dtype=np.uint8)  # Empty valid overlap

        res = crack_change_engine.analyze_crack_change(B, A, V, tolerance_px=2)
        assert res["valid_overlap_pixels"] == 0
        assert res["valid_overlap_percentage"] == 0.0
        assert res["before_crack_coverage_percentage"] == 0.0
        assert res["after_crack_coverage_percentage"] == 0.0
        assert res["raw_change_percentage"] == 0.0


class TestHybridVisualizations:
    """Validate color composite generation, dims, legend, and channels."""

    def test_create_hybrid_overlay_colors_and_dims(self):
        shape = (200, 300, 3)
        after_bgr = np.full(shape, 150, dtype=np.uint8)
        valid_mask = np.full((200, 300), 255, dtype=np.uint8)

        p_mask = np.zeros((200, 300), dtype=np.uint8)
        p_mask[120:150, 50:80] = 255  # Persistent (outside top-left legend)

        n_mask = np.zeros((200, 300), dtype=np.uint8)
        n_mask[120:150, 120:150] = 255  # New

        l_mask = np.zeros((200, 300), dtype=np.uint8)
        l_mask[120:150, 200:230] = 255  # No longer observed

        overlay = crack_change_engine.create_hybrid_overlay(
            after_bgr=after_bgr,
            valid_mask=valid_mask,
            persistent_mask=p_mask,
            newly_observed_mask=n_mask,
            no_longer_observed_mask=l_mask,
        )

        assert overlay.shape == shape
        assert overlay.dtype == np.uint8

        # Persistent region has strong Yellow tint (high G and R, lower B)
        p_color = overlay[135, 65]
        assert p_color[1] > 180 and p_color[2] > 180

        # Newly observed region has strong Red tint (high R, lower B and G)
        n_color = overlay[135, 135]
        assert n_color[2] > 180 and n_color[0] < 120

        # No longer observed region has strong Blue tint (high B, lower R and G)
        l_color = overlay[135, 215]
        assert l_color[0] > 180 and l_color[2] < 120
