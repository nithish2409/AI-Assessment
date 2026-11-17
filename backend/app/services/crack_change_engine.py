"""
Crack Change Engine for Phase 4: Hybrid Damage Comparison.

Provides:
  - Coordinate warping of before crack masks into after-image coordinates.
  - Spatial mask comparison restricted to valid overlap regions.
  - Pixel-level classification: persistent, newly observed, no-longer-observed.
  - Spatial tolerance adjustment (absorbing small alignment shifts).
  - Connected component progression analysis (new, expanded, persistent, reduced).
  - High-visibility hybrid visualizations and overlays.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CrackChangeEngine:
    """Specialized engine for spatial crack mask comparison and progression analysis."""

    def __init__(self, default_tolerance_px: int = 2):
        self.default_tolerance_px = default_tolerance_px

    @staticmethod
    def encode_image_png(image: np.ndarray) -> str:
        """Encode numpy array (grayscale or BGR) to base64 PNG string."""
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Unable to encode output image to PNG.")
        return base64.b64encode(buffer).decode("ascii")

    @staticmethod
    def decode_base64_mask(mask_b64: str) -> np.ndarray:
        """Decode base64 PNG string into single-channel uint8 binary mask."""
        raw_bytes = base64.b64decode(mask_b64)
        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        mask = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("Failed to decode base64 mask asset.")
        # Ensure strict binary {0, 255}
        return np.where(mask > 127, 255, 0).astype(np.uint8)

    @staticmethod
    def warp_before_mask(
        before_mask: np.ndarray,
        H: Optional[np.ndarray],
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Warp before binary crack mask into after-image coordinate system using H.
        Strictly employs INTER_NEAREST to preserve binary semantics.
        """
        h_after, w_after = target_shape
        if H is None:
            return np.zeros((h_after, w_after), dtype=np.uint8)

        warped = cv2.warpPerspective(
            before_mask,
            H,
            (w_after, h_after),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return np.where(warped > 127, 255, 0).astype(np.uint8)

    def analyze_crack_change(
        self,
        aligned_before_mask: np.ndarray,
        after_mask: np.ndarray,
        valid_mask: np.ndarray,
        tolerance_px: int = 2,
        calibrated_noise_floor_pp: float = 0.89,
    ) -> Dict[str, Any]:
        """
        Execute spatial comparison between aligned before mask and after mask.
        All calculations are strictly restricted to the valid overlap region (valid_mask > 0).
        """
        h_after, w_after = after_mask.shape[:2]
        V_bool = (valid_mask > 0)
        B_bool = (aligned_before_mask > 0) & V_bool
        A_bool = (after_mask > 0) & V_bool

        valid_pixels = int(np.count_nonzero(V_bool))
        total_canvas_pixels = h_after * w_after
        valid_overlap_pct = (
            round((100.0 * valid_pixels) / total_canvas_pixels, 4)
            if total_canvas_pixels > 0 else 0.0
        )

        # 1. Pixel-level spatial classification
        persistent_bool = B_bool & A_bool
        newly_observed_bool = A_bool & (~B_bool)
        no_longer_observed_bool = B_bool & (~A_bool)
        raw_change_bool = (B_bool ^ A_bool)

        before_crack_pixels = int(np.count_nonzero(B_bool))
        after_crack_pixels = int(np.count_nonzero(A_bool))
        persistent_pixels = int(np.count_nonzero(persistent_bool))
        newly_observed_pixels = int(np.count_nonzero(newly_observed_bool))
        no_longer_observed_pixels = int(np.count_nonzero(no_longer_observed_bool))
        raw_change_pixels = int(np.count_nonzero(raw_change_bool))

        # Percentage calculations strictly over valid overlap denominator
        denom = max(1, valid_pixels)
        before_crack_coverage_pct = round((100.0 * before_crack_pixels) / denom, 4)
        after_crack_coverage_pct = round((100.0 * after_crack_pixels) / denom, 4)
        coverage_change_pp = round(after_crack_coverage_pct - before_crack_coverage_pct, 4)
        raw_change_pct = round((100.0 * raw_change_pixels) / denom, 4)

        # 2. Spatial tolerance adjustment
        tolerance_px = max(0, min(10, int(tolerance_px)))
        if tolerance_px > 0:
            ksize = 2 * tolerance_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            B_dilated = (cv2.dilate((B_bool).astype(np.uint8) * 255, kernel) > 0)
            A_dilated = (cv2.dilate((A_bool).astype(np.uint8) * 255, kernel) > 0)

            # Tolerance-adjusted: pixels in after that are not even close to any before crack
            new_adj_bool = A_bool & (~B_dilated)
            # Tolerance-adjusted: pixels in before that are not even close to any after crack
            no_longer_adj_bool = B_bool & (~A_dilated)
        else:
            B_dilated = B_bool
            A_dilated = A_bool
            new_adj_bool = newly_observed_bool
            no_longer_adj_bool = no_longer_observed_bool

        tol_adj_new_pixels = int(np.count_nonzero(new_adj_bool))
        tol_adj_no_longer_pixels = int(np.count_nonzero(no_longer_adj_bool))

        # 3. Connected Component Analysis for Crack Progression
        comp_metrics = self._classify_connected_components(
            B_bool=B_bool,
            A_bool=A_bool,
            B_dilated=B_dilated,
            A_dilated=A_dilated,
            tolerance_px=tolerance_px,
        )

        # 4. Calibrated Noise Floor & Progression Categorization
        abs_change_pp = abs(coverage_change_pp)
        is_above_noise_floor = bool(abs_change_pp > calibrated_noise_floor_pp)

        new_comps = comp_metrics["potential_new_components"]
        exp_comps = comp_metrics["potential_expanded_components"]

        if new_comps > 0 and tol_adj_new_pixels >= 20:
            progression_label = "Potential newly observed crack regions"
            uncertainty_level = "low" if is_above_noise_floor else "medium"
            noise_context = (
                f"Coverage change ({coverage_change_pp:+.2f} pp) exceeds empirical noise floor ({calibrated_noise_floor_pp:.2f} pp)."
                if is_above_noise_floor
                else f"Coverage change ({coverage_change_pp:+.2f} pp) remains within empirical noise floor ({calibrated_noise_floor_pp:.2f} pp)."
            )
            interpretation_summary = (
                f"Potential newly observed crack regions detected: {new_comps} new component(s) "
                f"with {tol_adj_new_pixels} tolerance-adjusted pixels. {noise_context}"
            )
        elif exp_comps > 0 and tol_adj_new_pixels >= 20:
            progression_label = "Potential expansion of detected crack regions"
            uncertainty_level = "low" if is_above_noise_floor else "medium"
            noise_context = (
                f"Coverage change ({coverage_change_pp:+.2f} pp) exceeds empirical noise floor ({calibrated_noise_floor_pp:.2f} pp)."
                if is_above_noise_floor
                else f"Coverage change ({coverage_change_pp:+.2f} pp) remains within empirical noise floor ({calibrated_noise_floor_pp:.2f} pp)."
            )
            interpretation_summary = (
                f"Potential expansion of detected crack regions: {exp_comps} component(s) "
                f"exhibiting >=15% area growth. {noise_context}"
            )
        elif tol_adj_new_pixels == 0 and tol_adj_no_longer_pixels == 0:
            progression_label = "No substantial predicted change detected"
            uncertainty_level = "low"
            interpretation_summary = (
                "No substantial predicted change detected. Masks align within spatial tolerance buffer."
            )
        elif not is_above_noise_floor and tol_adj_new_pixels < 20:
            progression_label = "No substantial predicted change detected"
            uncertainty_level = "low"
            interpretation_summary = (
                f"Apparent mask variation (|Delta| = {abs_change_pp:.2f} pp) falls within the calibrated empirical noise floor "
                f"({calibrated_noise_floor_pp:.2f} pp) and spatial tolerance buffer."
            )
        else:
            progression_label = "No substantial predicted change detected"
            uncertainty_level = "medium"
            interpretation_summary = (
                f"Mask differences observed ({tol_adj_new_pixels} new / {tol_adj_no_longer_pixels} no-longer-observed pixels), "
                f"but lack distinct new or expanded components. Likely reflects registration or segmentation boundary variance."
            )

        # 5. Generate individual binary uint8 masks {0, 255}
        persistent_mask = np.where(persistent_bool, 255, 0).astype(np.uint8)
        newly_observed_mask = np.where(newly_observed_bool, 255, 0).astype(np.uint8)
        no_longer_observed_mask = np.where(no_longer_observed_bool, 255, 0).astype(np.uint8)
        tolerance_adjusted_new_mask = np.where(new_adj_bool, 255, 0).astype(np.uint8)
        tolerance_adjusted_no_longer_mask = np.where(no_longer_adj_bool, 255, 0).astype(np.uint8)
        before_crack_mask = np.where(B_bool, 255, 0).astype(np.uint8)
        after_crack_mask = np.where(A_bool, 255, 0).astype(np.uint8)

        return {
            "valid_overlap_pixels": valid_pixels,
            "valid_overlap_percentage": valid_overlap_pct,
            "before_crack_pixels": before_crack_pixels,
            "after_crack_pixels": after_crack_pixels,
            "before_crack_coverage_percentage": before_crack_coverage_pct,
            "after_crack_coverage_percentage": after_crack_coverage_pct,
            "coverage_change_percentage_points": coverage_change_pp,
            "persistent_pixels": persistent_pixels,
            "newly_observed_pixels": newly_observed_pixels,
            "no_longer_observed_pixels": no_longer_observed_pixels,
            "raw_change_pixels": raw_change_pixels,
            "raw_change_percentage": raw_change_pct,
            "tolerance_px": tolerance_px,
            "tolerance_adjusted_new_pixels": tol_adj_new_pixels,
            "tolerance_adjusted_no_longer_observed_pixels": tol_adj_no_longer_pixels,
            "potential_new_components": comp_metrics["potential_new_components"],
            "potential_expanded_components": comp_metrics["potential_expanded_components"],
            "persistent_components": comp_metrics["persistent_components"],
            "no_longer_observed_components": comp_metrics["no_longer_observed_components"],
            "progression_label": progression_label,
            "calibrated_noise_floor_pp": calibrated_noise_floor_pp,
            "is_above_noise_floor": is_above_noise_floor,
            "uncertainty_level": uncertainty_level,
            "interpretation_summary": interpretation_summary,
            "masks": {
                "before_crack": before_crack_mask,
                "after_crack": after_crack_mask,
                "persistent": persistent_mask,
                "newly_observed": newly_observed_mask,
                "no_longer_observed": no_longer_observed_mask,
                "tolerance_adjusted_new": tolerance_adjusted_new_mask,
                "tolerance_adjusted_no_longer": tolerance_adjusted_no_longer_mask,
            },
        }

    def _classify_connected_components(
        self,
        B_bool: np.ndarray,
        A_bool: np.ndarray,
        B_dilated: np.ndarray,
        A_dilated: np.ndarray,
        tolerance_px: int,
    ) -> Dict[str, int]:
        """Classify connected crack instances into new, expanded, persistent, and reduced."""
        num_B, labels_B, stats_B, _ = cv2.connectedComponentsWithStats(
            B_bool.astype(np.uint8), connectivity=8
        )
        num_A, labels_A, stats_A, _ = cv2.connectedComponentsWithStats(
            A_bool.astype(np.uint8), connectivity=8
        )

        potential_new = 0
        potential_expanded = 0
        persistent_comps = 0

        # Examine after-image components
        for j in range(1, num_A):
            comp_mask = (labels_A == j)
            comp_area = stats_A[j, cv2.CC_STAT_AREA]

            # Overlap with before crack mask
            direct_overlap = np.count_nonzero(comp_mask & B_bool)
            proximate_overlap = (
                np.count_nonzero(comp_mask & B_dilated) if tolerance_px > 0 else direct_overlap
            )

            if proximate_overlap == 0:
                potential_new += 1
            else:
                # Find which before components touch or overlap this after component
                touching_B = labels_B[comp_mask & (labels_B > 0)]
                if len(touching_B) > 0:
                    matched_B_labels = np.unique(touching_B)
                    matched_B_area = sum(stats_B[lbl, cv2.CC_STAT_AREA] for lbl in matched_B_labels)
                else:
                    matched_B_area = 0

                growth = comp_area - matched_B_area
                # Condition for potential expansion: area grew by >= 15% and at least 20 pixels
                if matched_B_area > 0 and growth >= 20 and (growth / matched_B_area) >= 0.15:
                    potential_expanded += 1
                else:
                    persistent_comps += 1

        # Examine before-image components for disappearance/reduction
        no_longer_observed_comps = 0
        for i in range(1, num_B):
            comp_mask = (labels_B == i)
            proximate_overlap = (
                np.count_nonzero(comp_mask & A_dilated) if tolerance_px > 0 else np.count_nonzero(comp_mask & A_bool)
            )
            if proximate_overlap == 0:
                no_longer_observed_comps += 1

        return {
            "potential_new_components": potential_new,
            "potential_expanded_components": potential_expanded,
            "persistent_components": persistent_comps,
            "no_longer_observed_components": no_longer_observed_comps,
        }

    def create_hybrid_overlay(
        self,
        after_bgr: np.ndarray,
        valid_mask: np.ndarray,
        persistent_mask: np.ndarray,
        newly_observed_mask: np.ndarray,
        no_longer_observed_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Generate high-clarity color-coded composite overlay.
          - Yellow (0, 255, 255): Persistent crack regions
          - Red (0, 0, 255): Newly observed crack regions
          - Blue (255, 0, 0): No-longer-observed crack regions
          - Outside valid overlap: Dimmed background
          - Includes explanatory visual legend in top corner.
        """
        h, w = after_bgr.shape[:2]
        canvas = after_bgr.copy()

        # 1. Dim pixels outside valid overlap to highlight comparison ROI
        invalid_mask = (valid_mask == 0)
        canvas[invalid_mask] = (canvas[invalid_mask].astype(np.float32) * 0.40).astype(np.uint8)

        # 2. Build colored crack layer
        color_layer = canvas.copy()

        # Masks
        p_idx = (persistent_mask > 0)
        n_idx = (newly_observed_mask > 0)
        l_idx = (no_longer_observed_mask > 0)

        # Assign colors: Yellow, Red, Blue
        color_layer[p_idx] = [0, 255, 255]   # BGR: Yellow
        color_layer[n_idx] = [0, 0, 255]     # BGR: Red
        color_layer[l_idx] = [255, 0, 0]     # BGR: Blue

        # Alpha blend crack pixels (60% crack color, 40% background)
        any_crack = p_idx | n_idx | l_idx
        blended = cv2.addWeighted(canvas, 0.40, color_layer, 0.60, 0)
        canvas[any_crack] = blended[any_crack]

        # 3. Contour outlines for crisp border definition
        for mask, color in [
            (persistent_mask, (0, 255, 255)),
            (newly_observed_mask, (0, 0, 255)),
            (no_longer_observed_mask, (255, 0, 0)),
        ]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 1)

        # 4. Render visual legend in top-left corner
        legend_x, legend_y = 12, 12
        legend_w, legend_h = 240, 78

        # Semi-transparent dark backing card
        overlay_card = canvas.copy()
        cv2.rectangle(
            overlay_card,
            (legend_x, legend_y),
            (legend_x + legend_w, legend_y + legend_h),
            (20, 20, 20),
            cv2.FILLED,
        )
        canvas = cv2.addWeighted(canvas, 0.25, overlay_card, 0.75, 0)
        cv2.rectangle(
            canvas,
            (legend_x, legend_y),
            (legend_x + legend_w, legend_y + legend_h),
            (80, 80, 80),
            1,
        )

        # Legend items: color patch + text
        items = [
            ((0, 255, 255), "Persistent Crack Region"),
            ((0, 0, 255), "Newly Observed Crack Region"),
            ((255, 0, 0), "No-Longer-Observed Region"),
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness = 1

        for idx, (col, text) in enumerate(items):
            py = legend_y + 16 + idx * 20
            # Color badge
            cv2.rectangle(canvas, (legend_x + 8, py - 9), (legend_x + 22, py + 3), col, cv2.FILLED)
            cv2.rectangle(canvas, (legend_x + 8, py - 9), (legend_x + 22, py + 3), (255, 255, 255), 1)
            # Label
            cv2.putText(
                canvas,
                text,
                (legend_x + 28, py),
                font,
                font_scale,
                (230, 230, 230),
                thickness,
                cv2.LINE_AA,
            )

        return canvas


crack_change_engine = CrackChangeEngine()
