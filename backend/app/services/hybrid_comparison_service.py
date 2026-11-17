"""
Hybrid Building Damage Comparison Service.

Integrates:
  1. Geometric feature alignment (Before -> After reference frame via SIFT/AKAZE/ORB/Direct).
  2. Common valid overlap ROI computation.
  3. Structural Similarity Index (SSIM) visual change detection.
  4. YOLO26s-seg crack segmentation on both original photographs.
  5. Coordinate warping of before crack masks into reference frame.
  6. Spatial mask comparison (persistent, newly observed, no-longer-observed).
  7. Spatial tolerance adjustment and connected component progression analysis.
  8. Qualified reporting distinguishing general visual change from crack change.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from app.services.comparison_service import comparison_service
from app.services.image_alignment import ImageAlignmentService
from app.services.damage_client import damage_client
from app.services.crack_change_engine import crack_change_engine
from app.schemas.hybrid_comparison import (
    HybridAlignmentInfo,
    HybridVisualComparison,
    HybridCrackComparison,
    HybridModelMeta,
    HybridImagesPayload,
    HybridTimingMeta,
    HybridComparisonResponse,
)

logger = logging.getLogger(__name__)


class HybridComparisonService:
    """Orchestrator for hybrid before/after building damage comparison."""

    def __init__(self, min_overlap_ratio: float = 0.50):
        self.min_overlap_ratio = min_overlap_ratio
        self.alignment_service = ImageAlignmentService(
            min_overlap_ratio=self.min_overlap_ratio
        )
        self.crack_engine = crack_change_engine
        self.min_ssim_region_area = 100

    async def compare_hybrid(
        self,
        before_bytes: bytes,
        before_type: str,
        after_bytes: bytes,
        after_type: str,
        alignment_mode: str = "auto",
        confidence: float = 0.25,
        iou: float = 0.45,
        tolerance_px: int = 2,
    ) -> HybridComparisonResponse:
        """
        Execute end-to-end hybrid damage comparison pipeline.
        """
        t_total_0 = time.perf_counter()

        # 1. Image Decoding
        t_dec_0 = time.perf_counter()
        before_bgr = comparison_service.decode_image(before_bytes, before_type)
        after_bgr = comparison_service.decode_image(after_bytes, after_type)
        h_after, w_after = after_bgr.shape[:2]

        # 2. Geometric Alignment (Before -> After reference frame)
        t_align_0 = time.perf_counter()
        align_res = self.alignment_service.align_before_to_after(
            before_image=before_bgr,
            after_image=after_bgr,
            mode=alignment_mode,
        )
        t_align = (time.perf_counter() - t_align_0) * 1000.0

        if align_res["status"] == "failed":
            raise ValueError(
                "Image alignment failed. Use photos showing the same surface with "
                "sufficient visible shared details and similar viewpoints."
            )

        H = align_res.get("homography")
        aligned_before_bgr = align_res["aligned_image"]
        valid_mask = align_res["valid_mask"]

        # Validate overlap ratio
        overlap_ratio = align_res.get("overlap_ratio", 0.0)
        if overlap_ratio < self.min_overlap_ratio:
            raise ValueError(
                f"Valid image overlap is insufficient ({overlap_ratio * 100:.1f}%). "
                f"At least {self.min_overlap_ratio * 100:.0f}% common visible area is required."
            )

        # Classify alignment status
        align_status = align_res["status"]
        align_conf = align_res.get("confidence", 0.0)
        if align_status == "aligned" and align_conf < 0.40:
            align_status_label = "uncertain"
        elif align_status in ("aligned", "identical"):
            align_status_label = "accepted"
        elif align_status == "direct":
            align_status_label = "direct"
        else:
            align_status_label = align_status

        alignment_warnings = list(align_res.get("warnings", []))
        if align_status_label == "uncertain":
            alignment_warnings.append(
                "Geometric alignment confidence is low; spatial crack comparisons may reflect slight registration shifts."
            )

        # 3. SSIM Visual Comparison (strictly on valid overlap region)
        t_ssim_0 = time.perf_counter()
        ssim_res = self._compute_valid_ssim(
            reference_bgr=after_bgr,
            aligned_moving_bgr=aligned_before_bgr,
            valid_mask=valid_mask,
        )
        t_ssim = (time.perf_counter() - t_ssim_0) * 1000.0

        # 4. YOLO Segmentation on Original Photographs (Parallel microservice calls)
        t_yolo_0 = time.perf_counter()
        yolo_before_task = damage_client.segment_image(
            image_bytes=before_bytes,
            filename="before_image.jpg",
            content_type=before_type,
            confidence=confidence,
            iou=iou,
        )
        yolo_after_task = damage_client.segment_image(
            image_bytes=after_bytes,
            filename="after_image.jpg",
            content_type=after_type,
            confidence=confidence,
            iou=iou,
        )

        resp_before, resp_after = await asyncio.gather(
            yolo_before_task, yolo_after_task
        )
        t_yolo_total = (time.perf_counter() - t_yolo_0) * 1000.0
        # Individual timings from microservice response if available
        yolo_before_ms = resp_before.get("inference", {}).get("duration_ms", t_yolo_total / 2)
        yolo_after_ms = resp_after.get("inference", {}).get("duration_ms", t_yolo_total / 2)

        # Decode masks from base64
        M_B_orig = self.crack_engine.decode_base64_mask(resp_before["images"]["mask"])
        M_A = self.crack_engine.decode_base64_mask(resp_after["images"]["mask"])

        # 5. Warp Before Crack Mask into After Coordinate Frame
        t_crack_0 = time.perf_counter()
        aligned_before_mask = self.crack_engine.warp_before_mask(
            before_mask=M_B_orig,
            H=H,
            target_shape=(h_after, w_after),
        )

        # 6. Spatial Mask Comparison & Progression Analysis
        crack_analysis = self.crack_engine.analyze_crack_change(
            aligned_before_mask=aligned_before_mask,
            after_mask=M_A,
            valid_mask=valid_mask,
            tolerance_px=tolerance_px,
        )
        t_crack = (time.perf_counter() - t_crack_0) * 1000.0

        # 7. Visualization Generation & Base64 Encoding
        t_vis_0 = time.perf_counter()
        masks = crack_analysis["masks"]
        hybrid_overlay = self.crack_engine.create_hybrid_overlay(
            after_bgr=after_bgr,
            valid_mask=valid_mask,
            persistent_mask=masks["persistent"],
            newly_observed_mask=masks["newly_observed"],
            no_longer_observed_mask=masks["no_longer_observed"],
        )

        images_payload = HybridImagesPayload(
            aligned_before=self.crack_engine.encode_image_png(aligned_before_bgr),
            visual_difference=self.crack_engine.encode_image_png(ssim_res["difference_heatmap"]),
            before_crack_mask=self.crack_engine.encode_image_png(masks["before_crack"]),
            after_crack_mask=self.crack_engine.encode_image_png(masks["after_crack"]),
            newly_observed_mask=self.crack_engine.encode_image_png(masks["newly_observed"]),
            persistent_mask=self.crack_engine.encode_image_png(masks["persistent"]),
            no_longer_observed_mask=self.crack_engine.encode_image_png(masks["no_longer_observed"]),
            tolerance_adjusted_new_mask=self.crack_engine.encode_image_png(masks["tolerance_adjusted_new"]),
            hybrid_overlay=self.crack_engine.encode_image_png(hybrid_overlay),
        )
        t_vis = (time.perf_counter() - t_vis_0) * 1000.0

        t_total = (time.perf_counter() - t_total_0) * 1000.0

        # 8. Assemble Qualified Response
        progression_label = crack_analysis["progression_label"]
        uncertainty_level = crack_analysis["uncertainty_level"]
        interpretation_summary = crack_analysis["interpretation_summary"]

        if align_status_label == "uncertain":
            uncertainty_level = "high"
            progression_label = "Comparison inconclusive"
            interpretation_summary = (
                "Comparison inconclusive due to low geometric alignment confidence. "
                "Spatial crack differences may be artifacts of geometric misregistration."
            )

        overall_warnings: List[str] = [
            "Crack-region differences represent model predictions and do not establish physical structural deterioration or repair.",
            "Visual change (SSIM) encompasses lighting, surface texture, staining, and camera variations in addition to physical alterations.",
            f"Coverage changes under {crack_analysis['calibrated_noise_floor_pp']:.2f} percentage points fall within the empirical noise floor of camera and lighting variation.",
        ]
        overall_warnings.extend(alignment_warnings)

        return HybridComparisonResponse(
            status="completed",
            image_width=w_after,
            image_height=h_after,
            alignment=HybridAlignmentInfo(
                status=align_status_label,
                method=align_res.get("method", "NONE"),
                method_used=align_res.get("method_used", align_res.get("method", "NONE")),
                detector=align_res.get("detector", align_res.get("method", "NONE")),
                geometric_status=align_res.get(
                    "geometric_status",
                    "VERIFIED_FEATURE_REGISTRATION"
                    if align_status_label == "accepted"
                    else "ASSUMED_UNVERIFIED"
                    if align_status_label == "direct"
                    else "IDENTICAL_INPUTS"
                    if align_status_label == "identical"
                    else "UNCERTAIN",
                ),
                confidence=float(align_res["confidence"]) if align_res.get("confidence") is not None else None,
                inlier_count=int(align_res["inliers"]) if align_res.get("inliers") is not None else None,
                inlier_ratio=float(align_res["inlier_ratio"]) if align_res.get("inlier_ratio") is not None else None,
                reprojection_error=float(align_res["reprojection_error"]) if align_res.get("reprojection_error") is not None else None,
                valid_overlap_pixels=crack_analysis["valid_overlap_pixels"],
                valid_overlap_percentage=crack_analysis["valid_overlap_percentage"],
                warnings=alignment_warnings,
            ),
            visual_comparison=HybridVisualComparison(
                ssim_score=ssim_res["mean_ssim"],
                visual_changed_pixels=ssim_res["changed_pixels"],
                visual_change_percentage=ssim_res["changed_percentage"],
                changed_regions_count=ssim_res["changed_regions_count"],
            ),
            crack_comparison=HybridCrackComparison(
                before_crack_pixels=crack_analysis["before_crack_pixels"],
                after_crack_pixels=crack_analysis["after_crack_pixels"],
                before_crack_coverage_percentage=crack_analysis["before_crack_coverage_percentage"],
                after_crack_coverage_percentage=crack_analysis["after_crack_coverage_percentage"],
                coverage_change_percentage_points=crack_analysis["coverage_change_percentage_points"],
                persistent_pixels=crack_analysis["persistent_pixels"],
                newly_observed_pixels=crack_analysis["newly_observed_pixels"],
                no_longer_observed_pixels=crack_analysis["no_longer_observed_pixels"],
                raw_change_pixels=crack_analysis["raw_change_pixels"],
                raw_change_percentage=crack_analysis["raw_change_percentage"],
                tolerance_px=crack_analysis["tolerance_px"],
                tolerance_adjusted_new_pixels=crack_analysis["tolerance_adjusted_new_pixels"],
                tolerance_adjusted_no_longer_observed_pixels=crack_analysis["tolerance_adjusted_no_longer_observed_pixels"],
                potential_new_components=crack_analysis["potential_new_components"],
                potential_expanded_components=crack_analysis["potential_expanded_components"],
                persistent_components=crack_analysis["persistent_components"],
                no_longer_observed_components=crack_analysis["no_longer_observed_components"],
                progression_label=progression_label,
                calibrated_noise_floor_pp=crack_analysis["calibrated_noise_floor_pp"],
                is_above_noise_floor=crack_analysis["is_above_noise_floor"],
                uncertainty_level=uncertainty_level,
                interpretation_summary=interpretation_summary,
            ),
            model=HybridModelMeta(
                name="YOLO26s-seg",
                class_names=["crack"],
                confidence_threshold=confidence,
                iou_threshold=iou,
            ),
            images=images_payload,
            timing=HybridTimingMeta(
                total_duration_ms=round(t_total, 2),
                alignment_ms=round(t_align, 2),
                ssim_ms=round(t_ssim, 2),
                yolo_before_ms=round(yolo_before_ms, 2),
                yolo_after_ms=round(yolo_after_ms, 2),
                crack_comparison_ms=round(t_crack, 2),
                visualization_ms=round(t_vis, 2),
            ),
            warnings=overall_warnings,
        )

    def _compute_valid_ssim(
        self,
        reference_bgr: np.ndarray,
        aligned_moving_bgr: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute SSIM strictly restricted to the valid overlap mask."""
        gray_ref = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        gray_mov = cv2.cvtColor(aligned_moving_bgr, cv2.COLOR_BGR2GRAY)

        # Gaussian smoothing to reduce sensor noise
        gray_ref = cv2.GaussianBlur(gray_ref, (5, 5), 0)
        gray_mov = cv2.GaussianBlur(gray_mov, (5, 5), 0)

        score, similarity_map = structural_similarity(
            gray_mov,
            gray_ref,
            full=True,
            data_range=255,
        )

        difference = ((1.0 - similarity_map) * 255).clip(0, 255).astype(np.uint8)

        # Safe mask: erode valid overlap to avoid boundary window artifacts
        safe_mask = cv2.erode(valid_mask, np.ones((9, 9), np.uint8), iterations=1)
        difference[safe_mask == 0] = 0

        # Otsu thresholding for visual change mask
        if np.count_nonzero(difference) > 0:
            _, binary_diff = cv2.threshold(
                difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            binary_diff = np.zeros_like(difference)

        binary_diff[safe_mask == 0] = 0

        # Clean noise via morphological open/close
        kernel = np.ones((3, 3), np.uint8)
        binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, kernel)
        binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Filter small regions
        contours, _ = cv2.findContours(binary_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(binary_diff)
        valid_regions_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_ssim_region_area:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                valid_regions_count += 1

        valid_pixels = int(np.count_nonzero(safe_mask))
        changed_pixels = int(np.count_nonzero(filtered_mask))
        changed_pct = round((100.0 * changed_pixels) / max(1, valid_pixels), 2)

        valid_similarity = similarity_map[safe_mask > 0]
        mean_ssim = round(float(valid_similarity.mean()), 4) if valid_similarity.size else 0.0

        # Create color heatmap for visual difference
        diff_heatmap = cv2.applyColorMap(difference, cv2.COLORMAP_JET)
        diff_heatmap[safe_mask == 0] = 0

        return {
            "mean_ssim": mean_ssim,
            "changed_pixels": changed_pixels,
            "changed_percentage": changed_pct,
            "changed_regions_count": valid_regions_count,
            "difference_heatmap": diff_heatmap,
            "binary_diff_mask": filtered_mask,
        }


hybrid_comparison_service = HybridComparisonService()
