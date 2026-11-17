
import base64

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from app.services.image_alignment import ImageAlignmentService


class ImageComparisonEngine:
    def __init__(self):
        self.max_dimension = 1200
        self.min_region_area = 100
        self.min_overlap_ratio = 0.65
        self.alignment_service = ImageAlignmentService(
            min_overlap_ratio=self.min_overlap_ratio,
        )

    def resize_for_processing(self, image, target_size=None):
        height, width = image.shape[:2]

        if target_size is not None:
            return cv2.resize(
                image, target_size,
                interpolation=cv2.INTER_AREA
            )

        scale = min(
            1.0,
            self.max_dimension / max(width, height)
        )

        if scale == 1.0:
            return image.copy()

        return cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA
        )

    def align_images(self, before, after, mode="auto"):
        orig_h_before, orig_w_before = before.shape[:2]
        orig_h_after, orig_w_after = after.shape[:2]
        dim_normalized = (orig_h_before, orig_w_before) != (orig_h_after, orig_w_after)

        aspect_before = orig_w_before / orig_h_before
        aspect_after = orig_w_after / orig_h_after
        aspect_diff = abs(aspect_before - aspect_after)

        # Align both images at a common processing resolution.
        reference = self.resize_for_processing(before)
        h, w = reference.shape[:2]

        moving = self.resize_for_processing(
            after, target_size=(w, h)
        )

        result = self.alignment_service.align(
            reference, moving, mode=mode
        )

        # Telemetry & coordinate policy tracking
        result["dimension_normalized"] = dim_normalized
        result["dimension_details"] = {
            "before_source": {"width": orig_w_before, "height": orig_h_before, "aspect_ratio": round(aspect_before, 3)},
            "after_source": {"width": orig_w_after, "height": orig_h_after, "aspect_ratio": round(aspect_after, 3)},
            "processing_dimensions": {"width": w, "height": h},
            "aspect_ratio_match": aspect_diff <= 0.02,
        }

        # If Direct mode, ensure dimension and aspect ratio warnings are explicit
        if mode == "direct":
            if dim_normalized:
                dim_warn = (
                    f"Source images have different dimensions ({orig_w_before}x{orig_h_before} vs {orig_w_after}x{orig_h_after}). "
                    "The after image was resized to reference coordinate frame. Geometric alignment was not verified."
                )
                if dim_warn not in result["warnings"]:
                    result["warnings"].append(dim_warn)
                if aspect_diff > 0.02:
                    aspect_warn = (
                        f"Source aspect ratios differ ({aspect_before:.2f} vs {aspect_after:.2f}). "
                        "Resizing introduces non-uniform scale distortion; structural differences may be magnified."
                    )
                    if aspect_warn not in result["warnings"]:
                        result["warnings"].append(aspect_warn)

        status = result["status"]

        if status == "failed":
            # Return the same 4-tuple the rest of the engine expects,
            # but with a full-white mask and the failure metadata.
            return (
                reference,
                moving,
                np.full((h, w), 255, dtype=np.uint8),
                result,
            )

        aligned_image = result["aligned_image"]
        valid_mask = result["valid_mask"]

        return (
            reference,
            aligned_image,
            valid_mask,
            result,
        )

    @staticmethod
    def encode_image(image):
        success, buffer = cv2.imencode(".png", image)

        if not success:
            raise ValueError("Unable to encode output image.")

        return base64.b64encode(buffer).decode("ascii")

    def compare(self, before, after, mode="auto"):
        (
            reference,
            aligned,
            valid_mask,
            alignment
        ) = self.align_images(before, after, mode=mode)

        # Do not calculate a trustworthy change score if
        # geometrical alignment was not established.
        if alignment["status"] == "failed":
            raise ValueError(
                "Image alignment failed. Use photos with "
                "similar viewpoints and visible common details."
            )

        gray_before = cv2.cvtColor(
            reference, cv2.COLOR_BGR2GRAY
        )
        gray_after = cv2.cvtColor(
            aligned, cv2.COLOR_BGR2GRAY
        )

        # Slight smoothing reduces sensor noise.
        gray_before = cv2.GaussianBlur(
            gray_before, (5, 5), 0
        )
        gray_after = cv2.GaussianBlur(
            gray_after, (5, 5), 0
        )

        score, similarity_map = structural_similarity(
            gray_before,
            gray_after,
            full=True,
            data_range=255
        )

        # Convert similarity into difference intensity.
        difference = (
            (1.0 - similarity_map) * 255
        ).clip(0, 255).astype(np.uint8)

        # Ignore pixels outside the valid aligned area.
        # Erode to avoid SSIM window effects at warp borders.
        safe_mask = cv2.erode(
            valid_mask,
            np.ones((9, 9), np.uint8),
            iterations=1
        )

        difference[safe_mask == 0] = 0

        # Adaptive binary threshold from the difference image.
        _, binary_mask = cv2.threshold(
            difference,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        binary_mask[safe_mask == 0] = 0

        # Remove isolated noise and join nearby changed pixels.
        kernel = np.ones((3, 3), np.uint8)

        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        # Detect connected changed regions.
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        highlighted = aligned.copy()
        regions = []
        filtered_mask = np.zeros_like(binary_mask)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_region_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            cv2.drawContours(
                filtered_mask,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED
            )

            cv2.rectangle(
                highlighted,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            regions.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area_pixels": round(float(area), 2)
            })

        regions.sort(
            key=lambda item: item["area_pixels"],
            reverse=True
        )


        valid_pixels = np.count_nonzero(safe_mask)
        changed_pixels = np.count_nonzero(filtered_mask)

        changed_percentage = (
            100 * changed_pixels / valid_pixels
            if valid_pixels else 0.0
        )

        # SSIM's global score includes warp boundaries.
        # Use the valid-area similarity map for reporting.
        valid_similarity = similarity_map[safe_mask > 0]

        mean_ssim = (
            float(valid_similarity.mean())
            if valid_similarity.size else 0.0
        )

        # Build a JSON-safe alignment sub-dict (strip numpy arrays).
        alignment_info = {
            k: v for k, v in alignment.items()
            if k not in ("aligned_image", "valid_mask", "homography")
        }

        # For Direct mode, ensure feature-based metrics are explicitly None (not applicable)
        if mode == "direct":
            alignment_info["geometric_status"] = "ASSUMED_UNVERIFIED"
            alignment_info["method_used"] = "DIRECT"
            alignment_info["detector"] = "DIRECT"
            alignment_info["confidence"] = None
            alignment_info["alignment_quality"] = None
            alignment_info["inliers"] = None
            alignment_info["matches"] = None
            alignment_info["inlier_ratio"] = None
            alignment_info["reprojection_error"] = None

        return {
            "status": "completed",
            "alignment": alignment_info,
            "ssim_score": round(mean_ssim, 4),
            "changed_percentage": round(
                float(changed_percentage), 2
            ),
            "changed_regions_count": len(regions),
            "regions": regions,
            "image_width": reference.shape[1],
            "image_height": reference.shape[0],
            "images": {
                "before": self.encode_image(reference),
                "aligned_after": self.encode_image(aligned),
                "difference": self.encode_image(difference),
                "mask": self.encode_image(filtered_mask),
                "highlighted": self.encode_image(highlighted)
            }
        }


image_comparison_engine = ImageComparisonEngine()
