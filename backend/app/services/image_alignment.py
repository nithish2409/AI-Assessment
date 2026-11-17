"""
Dedicated image alignment service.

Supports SIFT, AKAZE, and ORB feature-based alignment plus an explicit
direct-comparison mode.  The ``auto`` strategy tries each detector in
quality order (SIFT → AKAZE → ORB) and selects the geometrically best
result, or returns a structured failure — it **never** silently falls
back to a direct pixel comparison.

Confidence score
~~~~~~~~~~~~~~~~
A bounded heuristic in [0, 1] that combines four geometric quality
signals with manually tuned weights:

    confidence = clamp(
        0.30 × (inlier_ratio)
      + 0.25 × clamp(inliers / 100, 0, 1)
      + 0.25 × (1 − clamp(reproj_error / 5, 0, 1))
      + 0.20 × overlap_ratio
    , 0, 1)

* *inlier_ratio*:  fraction of good matches confirmed by RANSAC.
  Higher ⇒ more consistent geometry.
* *inlier_count*:  absolute number of inliers, saturated at 100.
  More inliers ⇒ more evidence.
* *reprojection_error*:  mean pixel distance of inliers after
  transformation.  Lower is better; capped at 5 px.
* *overlap_ratio*:  fraction of the reference frame covered by valid
  warped pixels.  More overlap ⇒ more comparable area.

The score is **not** a calibrated probability.  It is a coarse
indicator of geometric alignment quality.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime capability probes — AKAZE was removed in some OpenCV 5 builds.
# ---------------------------------------------------------------------------
_HAS_AKAZE: bool = hasattr(cv2, "AKAZE_create")


class ImageAlignmentService:
    """Multi-method image alignment with geometric validation."""

    # ---- configurable thresholds (class-level defaults) -------------------
    MIN_GOOD_MATCHES: int = 10
    MIN_INLIERS: int = 8
    MIN_INLIER_RATIO: float = 0.45
    MIN_OVERLAP_RATIO: float = 0.65
    RANSAC_REPROJ_THRESHOLD: float = 3.0
    LOWE_RATIO: float = 0.75
    MAX_REPROJ_ERROR: float = 5.0
    MAX_SCALE_CHANGE: float = 4.0     # no side may grow/shrink >4×
    MAX_SKEW: float = 0.8             # tangent of max acceptable shear

    VALID_MODES = {"auto", "sift", "akaze", "orb", "direct"}

    def __init__(self, **overrides: Any) -> None:
        for key, value in overrides.items():
            attr = key.upper()
            if hasattr(self, attr):
                setattr(self, attr, value)

    # ====================================================================
    # Public entry points
    # ====================================================================
    def align_before_to_after(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
        mode: str = "auto",
    ) -> dict:
        """Align *before_image* into *after_image* reference coordinate space (Phase 4).

        Maps Before -> After. Returns estimated homography H, aligned_image,
        valid_mask, inliers, reprojection_error, and quality metrics.
        """
        mode = mode.lower().strip()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown alignment mode '{mode}'. "
                f"Valid modes: {sorted(self.VALID_MODES)}"
            )

        h_after, w_after = after_image.shape[:2]
        h_before, w_before = before_image.shape[:2]

        # Fast path: byte-identical images.
        if np.array_equal(before_image, after_image):
            res = self._identical_result(before_image, h_after, w_after)
            res["homography"] = np.eye(3, dtype=np.float32)
            return res

        if mode == "direct":
            dim_norm = (h_before, w_before) != (h_after, w_after)
            if not dim_norm:
                H = np.eye(3, dtype=np.float32)
                aligned = before_image.copy()
            else:
                sx = float(w_after) / float(w_before)
                sy = float(h_after) / float(h_before)
                H = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
                aligned = cv2.resize(before_image, (w_after, h_after), interpolation=cv2.INTER_AREA)

            warnings = [
                "Direct comparison mode: geometric alignment was not "
                "verified. Results assume that corresponding pixels "
                "represent the same scene locations. Resizing or "
                "normalization does not independently verify alignment."
            ]
            if dim_norm:
                warnings.append(
                    f"Source images had different dimensions ({w_before}x{h_before} vs {w_after}x{h_after}). "
                    "Before image was resized to after coordinate frame."
                )
                aspect_before = w_before / h_before
                aspect_after = w_after / h_after
                if abs(aspect_before - aspect_after) > 0.02:
                    warnings.append(
                        f"Source aspect ratios differ ({aspect_before:.2f} vs {aspect_after:.2f}). "
                        "Resizing introduces non-uniform scale distortion."
                    )

            return {
                "status": "direct",
                "method": "DIRECT",
                "method_used": "DIRECT",
                "detector": "DIRECT",
                "geometric_status": "ASSUMED_UNVERIFIED",
                "confidence": None,
                "alignment_quality": None,
                "matches": None,
                "inliers": None,
                "inlier_ratio": None,
                "overlap_ratio": 1.0,
                "reprojection_error": None,
                "homography": H,
                "aligned_image": aligned,
                "valid_mask": np.full((h_after, w_after), 255, dtype=np.uint8),
                "dimension_normalized": dim_norm,
                "warnings": warnings,
            }

        # ----- feature-based alignment ----------------------------------
        gray_before = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)

        if mode == "auto":
            candidates: list[dict] = []
            attempted: list[str] = []
            reasons: list[str] = []
            for method in self._AUTO_ORDER:
                res = self._try_method_b2a(
                    method, before_image, after_image,
                    gray_before, gray_after,
                )
                attempted.append(method.upper())
                if res is not None:
                    candidates.append(res)
                else:
                    reasons.append(f"{method.upper()} did not produce a reliable alignment.")

            if not candidates:
                fail_res = self._failure_result(attempted=attempted, reasons=reasons)
                fail_res["homography"] = None
                return fail_res

            best = max(candidates, key=lambda r: r["confidence"])
            return best

        # Single-method request.
        result = self._try_method_b2a(
            mode, before_image, after_image,
            gray_before, gray_after,
        )
        if result is not None:
            return result

        fail_res = self._failure_result(
            attempted=[mode.upper()],
            reasons=[f"{mode.upper()} did not produce a reliable alignment."],
        )
        fail_res["homography"] = None
        return fail_res

    def _try_method_b2a(
        self,
        method: str,
        before_bgr: np.ndarray,
        after_bgr: np.ndarray,
        gray_before: np.ndarray,
        gray_after: np.ndarray,
    ) -> dict | None:
        """Feature detection and homography estimation mapping before -> after."""
        if method == "sift":
            return self._align_b2a_with_detector(
                "SIFT", cv2.SIFT_create(), cv2.NORM_L2,
                before_bgr, after_bgr, gray_before, gray_after,
            )
        if method == "akaze":
            if not _HAS_AKAZE:
                return None
            return self._align_b2a_with_detector(
                "AKAZE", cv2.AKAZE_create(), cv2.NORM_HAMMING,
                before_bgr, after_bgr, gray_before, gray_after,
            )
        if method == "orb":
            return self._align_b2a_with_detector(
                "ORB", cv2.ORB_create(nfeatures=3000), cv2.NORM_HAMMING,
                before_bgr, after_bgr, gray_before, gray_after,
            )
        return None

    def _align_b2a_with_detector(
        self,
        name: str,
        detector: cv2.Feature2D,
        norm_type: int,
        before_bgr: np.ndarray,
        after_bgr: np.ndarray,
        gray_before: np.ndarray,
        gray_after: np.ndarray,
    ) -> dict | None:
        h_after, w_after = after_bgr.shape[:2]
        h_before, w_before = before_bgr.shape[:2]

        kp_before, desc_before = detector.detectAndCompute(gray_before, None)
        kp_after, desc_after = detector.detectAndCompute(gray_after, None)

        if desc_before is None or desc_after is None:
            return None
        if len(desc_before) < 2 or len(desc_after) < 2:
            return None

        matcher = cv2.BFMatcher(norm_type)
        raw_matches = matcher.knnMatch(desc_before, desc_after, k=2)

        good_matches = [
            m for pair in raw_matches
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < self.LOWE_RATIO * n.distance
        ]

        if len(good_matches) < self.MIN_GOOD_MATCHES:
            return None

        # Source = before, Destination = after
        src_pts = np.float32([kp_before[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_after[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        matrix, inlier_mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            self.RANSAC_REPROJ_THRESHOLD,
        )

        if matrix is None or inlier_mask is None:
            return None

        inlier_mask_flat = inlier_mask.ravel().astype(bool)
        inliers = int(inlier_mask_flat.sum())
        inlier_ratio = inliers / len(good_matches)

        if inliers < self.MIN_INLIERS or inlier_ratio < self.MIN_INLIER_RATIO:
            return None

        validation = self._validate_homography(matrix, w_before, h_before)
        if not validation["valid"]:
            return None

        reproj_error = self._reprojection_error(
            src_pts[inlier_mask_flat],
            dst_pts[inlier_mask_flat],
            matrix,
        )
        if reproj_error > self.MAX_REPROJ_ERROR:
            return None

        aligned = cv2.warpPerspective(before_bgr, matrix, (w_after, h_after))

        ones_mask = np.full((h_before, w_before), 255, dtype=np.uint8)
        valid_mask = cv2.warpPerspective(
            ones_mask, matrix, (w_after, h_after),
            flags=cv2.INTER_NEAREST,
        )

        overlap_ratio = float(np.count_nonzero(valid_mask)) / (w_after * h_after)
        if overlap_ratio < self.MIN_OVERLAP_RATIO:
            return None

        confidence = self._confidence(
            inlier_ratio, inliers, reproj_error, overlap_ratio,
        )

        warnings: list[str] = []
        if confidence < 0.4:
            warnings.append(
                "Alignment confidence is low; comparison results may be unreliable."
            )

        return {
            "status": "aligned",
            "method": name,
            "method_used": name,
            "detector": name,
            "geometric_status": "VERIFIED_FEATURE_REGISTRATION",
            "confidence": round(confidence, 4),
            "alignment_quality": round(confidence, 4),
            "matches": len(good_matches),
            "inliers": inliers,
            "inlier_ratio": round(inlier_ratio, 4),
            "overlap_ratio": round(overlap_ratio, 4),
            "reprojection_error": round(reproj_error, 4),
            "homography": matrix,
            "aligned_image": aligned,
            "valid_mask": valid_mask,
            "warnings": warnings,
        }

    def align(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
        mode: str = "auto",
    ) -> dict:
        """Align *after_image* into *before_image* coordinate space.

        Parameters
        ----------
        before_image, after_image:
            BGR images (already at processing resolution).
        mode:
            ``"auto"`` | ``"sift"`` | ``"akaze"`` | ``"orb"`` |
            ``"direct"``.

        Returns
        -------
        dict  — always contains at least ``status``, ``method``,
        ``confidence``, ``warnings``.
        """
        mode = mode.lower().strip()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown alignment mode '{mode}'. "
                f"Valid modes: {sorted(self.VALID_MODES)}"
            )

        h, w = before_image.shape[:2]

        # Fast path: byte-identical images.
        if np.array_equal(before_image, after_image):
            return self._identical_result(before_image, h, w)

        if mode == "direct":
            return self._direct_result(before_image, after_image)

        # ----- feature-based alignment ----------------------------------
        gray_before = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)

        if mode == "auto":
            return self._auto_align(
                before_image, after_image,
                gray_before, gray_after,
            )

        # Single-method request.
        result = self._try_method(
            mode, before_image, after_image,
            gray_before, gray_after,
        )
        if result is not None:
            return result

        return self._failure_result(
            attempted=[mode.upper()],
            reasons=[f"{mode.upper()} did not produce a reliable alignment."],
        )

    # ====================================================================
    # Auto strategy
    # ====================================================================
    _AUTO_ORDER = ["sift", "akaze", "orb"]

    def _auto_align(
        self,
        before_bgr: np.ndarray,
        after_bgr: np.ndarray,
        gray_before: np.ndarray,
        gray_after: np.ndarray,
    ) -> dict:
        """Try each method; keep the geometrically best result."""
        candidates: list[dict] = []
        attempted: list[str] = []
        reasons: list[str] = []

        for method in self._AUTO_ORDER:
            result = self._try_method(
                method, before_bgr, after_bgr,
                gray_before, gray_after,
            )
            if result is not None:
                candidates.append(result)
                attempted.append(method.upper())
            else:
                attempted.append(method.upper())
                reasons.append(
                    f"{method.upper()} did not produce a reliable alignment."
                )

        if not candidates:
            return self._failure_result(attempted=attempted, reasons=reasons)

        # Select the best candidate by confidence.
        best = max(candidates, key=lambda r: r["confidence"])
        return best

    # ====================================================================
    # Individual method dispatcher
    # ====================================================================
    def _try_method(
        self,
        method: str,
        before_bgr: np.ndarray,
        after_bgr: np.ndarray,
        gray_before: np.ndarray,
        gray_after: np.ndarray,
    ) -> dict | None:
        """Return an alignment result dict or ``None`` on failure."""
        if method == "sift":
            return self._align_with_detector(
                "SIFT", cv2.SIFT_create(),
                cv2.NORM_L2,
                before_bgr, after_bgr,
                gray_before, gray_after,
            )
        if method == "akaze":
            if not _HAS_AKAZE:
                logger.info(
                    "AKAZE requested but not available in this OpenCV build."
                )
                return None
            return self._align_with_detector(
                "AKAZE", cv2.AKAZE_create(),
                cv2.NORM_HAMMING,
                before_bgr, after_bgr,
                gray_before, gray_after,
            )
        if method == "orb":
            return self._align_with_detector(
                "ORB", cv2.ORB_create(nfeatures=3000),
                cv2.NORM_HAMMING,
                before_bgr, after_bgr,
                gray_before, gray_after,
            )
        return None

    # ====================================================================
    # Core alignment pipeline (shared by all feature detectors)
    # ====================================================================
    def _align_with_detector(
        self,
        name: str,
        detector: cv2.Feature2D,
        norm_type: int,
        before_bgr: np.ndarray,
        after_bgr: np.ndarray,
        gray_before: np.ndarray,
        gray_after: np.ndarray,
    ) -> dict | None:
        """Run the full detect → match → RANSAC → warp → validate pipeline."""
        h, w = before_bgr.shape[:2]

        # 1. Detect keypoints & descriptors.
        kp_before, desc_before = detector.detectAndCompute(gray_before, None)
        kp_after, desc_after = detector.detectAndCompute(gray_after, None)

        if desc_before is None or desc_after is None:
            return None
        if len(desc_before) < 2 or len(desc_after) < 2:
            return None

        # 2. Match with Lowe's ratio test.
        matcher = cv2.BFMatcher(norm_type)
        raw_matches = matcher.knnMatch(desc_after, desc_before, k=2)

        good_matches = [
            m
            for pair in raw_matches
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < self.LOWE_RATIO * n.distance
        ]

        if len(good_matches) < self.MIN_GOOD_MATCHES:
            return None

        # 3. RANSAC homography.
        src_pts = np.float32(
            [kp_after[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        dst_pts = np.float32(
            [kp_before[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        matrix, inlier_mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            self.RANSAC_REPROJ_THRESHOLD,
        )

        if matrix is None or inlier_mask is None:
            return None

        inlier_mask_flat = inlier_mask.ravel().astype(bool)
        inliers = int(inlier_mask_flat.sum())
        inlier_ratio = inliers / len(good_matches)

        if inliers < self.MIN_INLIERS:
            return None
        if inlier_ratio < self.MIN_INLIER_RATIO:
            return None

        # 4. Validate the homography geometrically.
        validation = self._validate_homography(matrix, w, h)
        if not validation["valid"]:
            return None

        # 5. Compute reprojection error for inliers.
        reproj_error = self._reprojection_error(
            src_pts[inlier_mask_flat],
            dst_pts[inlier_mask_flat],
            matrix,
        )
        if reproj_error > self.MAX_REPROJ_ERROR:
            return None

        # 6. Warp the after image.
        aligned = cv2.warpPerspective(after_bgr, matrix, (w, h))

        # 7. Valid-pixel mask (non-black warp border).
        ones_mask = np.full(
            after_bgr.shape[:2], 255, dtype=np.uint8
        )
        valid_mask = cv2.warpPerspective(
            ones_mask, matrix, (w, h),
            flags=cv2.INTER_NEAREST,
        )

        overlap_ratio = float(np.count_nonzero(valid_mask)) / (w * h)
        if overlap_ratio < self.MIN_OVERLAP_RATIO:
            return None

        # 8. Confidence score.
        confidence = self._confidence(
            inlier_ratio, inliers, reproj_error, overlap_ratio,
        )

        warnings: list[str] = []
        if confidence < 0.4:
            warnings.append(
                "Alignment confidence is low; comparison results "
                "may be unreliable."
            )

        return {
            "status": "aligned",
            "method": name,
            "method_used": name,
            "detector": name,
            "geometric_status": "VERIFIED_FEATURE_REGISTRATION",
            "confidence": round(confidence, 4),
            "alignment_quality": round(confidence, 4),
            "matches": len(good_matches),
            "inliers": inliers,
            "inlier_ratio": round(inlier_ratio, 4),
            "overlap_ratio": round(overlap_ratio, 4),
            "reprojection_error": round(reproj_error, 4),
            "aligned_image": aligned,
            "valid_mask": valid_mask,
            "warnings": warnings,
        }

    # ====================================================================
    # Homography geometric validation
    # ====================================================================
    def _validate_homography(
        self, H: np.ndarray, w: int, h: int,
    ) -> dict:
        """Check that *H* is non-degenerate and geometrically plausible."""
        # Finite check.
        if not np.isfinite(H).all():
            return {"valid": False, "reason": "Non-finite values in H."}

        # Determinant — must not be near zero (degenerate) or negative
        # (reflection / flip).
        det = np.linalg.det(H)
        if abs(det) < 1e-6:
            return {"valid": False, "reason": "Degenerate homography (det≈0)."}

        # Transform corners of the source image.
        corners = np.float32([
            [0, 0], [w, 0], [w, h], [0, h],
        ]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

        # Side lengths — check extreme scaling.
        for i in range(4):
            p1 = transformed[i]
            p2 = transformed[(i + 1) % 4]
            side_len = np.linalg.norm(p2 - p1)
            original_sides = [w, h, w, h]
            ratio = side_len / max(original_sides[i], 1)
            if ratio > self.MAX_SCALE_CHANGE or ratio < 1 / self.MAX_SCALE_CHANGE:
                return {
                    "valid": False,
                    "reason": f"Extreme scaling on side {i} (ratio={ratio:.2f}).",
                }

        # Convexity check — the transformed quad must be convex and
        # not self-intersecting.
        if not self._is_convex(transformed):
            return {"valid": False, "reason": "Transformed quad is not convex."}

        # Perspective distortion — check that the h[2,0] and h[2,1]
        # perspective components don't cause unreasonable shear.
        if abs(H[2, 0]) * w > self.MAX_SKEW or abs(H[2, 1]) * h > self.MAX_SKEW:
            return {
                "valid": False,
                "reason": "Unreasonable perspective distortion.",
            }

        return {"valid": True, "reason": "OK"}

    @staticmethod
    def _is_convex(quad: np.ndarray) -> bool:
        """Return True if the 4-point polygon is convex (consistent winding)."""
        n = len(quad)
        sign = None
        for i in range(n):
            v1 = quad[(i + 1) % n] - quad[i]
            v2 = quad[(i + 2) % n] - quad[(i + 1) % n]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if sign is None:
                sign = cross > 0
            elif (cross > 0) != sign:
                return False
        return True

    # ====================================================================
    # Reprojection error
    # ====================================================================
    @staticmethod
    def _reprojection_error(
        src: np.ndarray, dst: np.ndarray, H: np.ndarray,
    ) -> float:
        """Mean L2 reprojection error over inlier correspondences."""
        if len(src) == 0:
            return float("inf")
        src_2d = src.reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(src_2d, H).reshape(-1, 2)
        dst_2d = dst.reshape(-1, 2)
        errors = np.linalg.norm(projected - dst_2d, axis=1)
        return float(errors.mean())

    # ====================================================================
    # Confidence score
    # ====================================================================
    @staticmethod
    def _confidence(
        inlier_ratio: float,
        inlier_count: int,
        reproj_error: float,
        overlap_ratio: float,
    ) -> float:
        """Bounded heuristic confidence in [0, 1].

        Weights (manually tuned):
            0.30 × inlier_ratio
            0.25 × min(inlier_count / 100, 1)
            0.25 × (1 − min(reproj_error / 5, 1))
            0.20 × overlap_ratio
        """
        score = (
            0.30 * inlier_ratio
            + 0.25 * min(inlier_count / 100.0, 1.0)
            + 0.25 * (1.0 - min(reproj_error / 5.0, 1.0))
            + 0.20 * overlap_ratio
        )
        return max(0.0, min(1.0, score))

    # ====================================================================
    # Identical / direct / failure result constructors
    # ====================================================================
    @staticmethod
    def _identical_result(image: np.ndarray, h: int, w: int) -> dict:
        return {
            "status": "identical",
            "method": "NONE",
            "method_used": "NONE",
            "detector": "NONE",
            "geometric_status": "IDENTICAL_INPUTS",
            "confidence": 1.0,
            "alignment_quality": 1.0,
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "overlap_ratio": 1.0,
            "reprojection_error": 0.0,
            "aligned_image": image.copy(),
            "valid_mask": np.ones((h, w), dtype=np.uint8) * 255,
            "warnings": [],
        }

    @staticmethod
    def _direct_result(
        before: np.ndarray, after: np.ndarray,
    ) -> dict:
        h, w = before.shape[:2]
        ah, aw = after.shape[:2]
        dimension_normalized = (ah, aw) != (h, w)

        if dimension_normalized:
            after = cv2.resize(after, (w, h), interpolation=cv2.INTER_AREA)

        warnings = [
            "Direct comparison mode: geometric alignment was not "
            "verified. Results assume that corresponding pixels "
            "represent the same scene locations. Resizing or "
            "normalization does not independently verify alignment."
        ]
        if dimension_normalized:
            warnings.append(
                f"Source images had different dimensions ({aw}x{ah} vs {w}x{h}). "
                "After image was resized to reference coordinate frame."
            )
            aspect_before = w / h
            aspect_after = aw / ah
            if abs(aspect_before - aspect_after) > 0.02:
                warnings.append(
                    f"Source aspect ratios differ ({aspect_before:.2f} vs {aspect_after:.2f}). "
                    "Resizing introduces non-uniform scale distortion."
                )

        return {
            "status": "direct",
            "method": "DIRECT",
            "method_used": "DIRECT",
            "detector": "DIRECT",
            "geometric_status": "ASSUMED_UNVERIFIED",
            "confidence": None,
            "alignment_quality": None,
            "matches": None,
            "inliers": None,
            "inlier_ratio": None,
            "overlap_ratio": 1.0,
            "reprojection_error": None,
            "aligned_image": after,
            "valid_mask": np.ones((h, w), dtype=np.uint8) * 255,
            "dimension_normalized": dimension_normalized,
            "warnings": warnings,
        }

    @staticmethod
    def _failure_result(
        attempted: list[str] | None = None,
        reasons: list[str] | None = None,
    ) -> dict:
        return {
            "status": "failed",
            "method": "NONE",
            "method_used": "NONE",
            "detector": "NONE",
            "geometric_status": "FAILED",
            "confidence": 0.0,
            "alignment_quality": 0.0,
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "overlap_ratio": 0.0,
            "reprojection_error": 0.0,
            "aligned_image": None,
            "valid_mask": None,
            "warnings": [],
            "attempted_methods": attempted or [],
            "failure_reasons": reasons or [],
        }
