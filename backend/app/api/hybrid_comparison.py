"""
FastAPI Router for Phase 4: Hybrid Building Damage Comparison.
Endpoint:
  POST /api/comparison/hybrid
"""
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app.services.comparison_service import MAX_FILE_SIZE, ALLOWED_TYPES, MAX_PIXELS
from app.services.image_alignment import ImageAlignmentService
from app.services.damage_client import (
    DamageServiceError,
    DamageServiceUnavailableError,
    DamageServiceTimeoutError,
)
from app.services.hybrid_comparison_service import hybrid_comparison_service
from app.schemas.hybrid_comparison import HybridComparisonResponse

logger = logging.getLogger("api.hybrid_comparison")
router = APIRouter()

VALID_ALIGNMENT_MODES = ImageAlignmentService.VALID_MODES


async def read_and_validate_file(file: UploadFile, label: str) -> bytes:
    """Validate content-type and size of an uploaded image."""
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "INVALID_IMAGE_TYPE",
                "message": f"Unsupported format for {label} image: '{file.content_type}'. Only JPEG, PNG and WebP are accepted.",
            },
        )

    data = await file.read(MAX_FILE_SIZE + 1)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "EMPTY_IMAGE",
                "message": f"{label.capitalize()} image upload is empty.",
            },
        )

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "IMAGE_TOO_LARGE",
                "message": f"{label.capitalize()} image exceeds the 10 MB file-size limit.",
            },
        )

    return data


@router.post(
    "/hybrid",
    response_model=HybridComparisonResponse,
    tags=["Hybrid Damage Comparison"],
    summary="Compare before and after building photographs with alignment, SSIM, and YOLO crack segmentation",
)
async def compare_hybrid_images(
    before: UploadFile = File(..., description="Before building photograph (JPEG, PNG, WebP)"),
    after: UploadFile = File(..., description="After building photograph (JPEG, PNG, WebP)"),
    alignment_mode: str = Form("auto", description="Alignment detector: auto, sift, akaze, orb, direct"),
    confidence: float = Form(0.25, description="YOLO detection confidence threshold in (0, 1]"),
    iou: float = Form(0.45, description="YOLO NMS IoU threshold in (0, 1]"),
    tolerance_px: int = Form(2, description="Spatial alignment tolerance in pixels [0, 10]"),
):
    """
    Executes comprehensive hybrid damage assessment:
      1. Geometrically aligns before image to after image reference frame.
      2. Computes common valid overlap ROI.
      3. Calculates SSIM visual difference over valid pixels.
      4. Segments cracks in both original photographs via YOLO26s-seg.
      5. Warps before crack mask into reference frame using homography H.
      6. Spatially compares crack masks (persistent, newly observed, no-longer-observed).
      7. Employs spatial tolerance buffering to absorb minor registration shifts.
      8. Analyzes connected components to detect potential crack expansion vs new cracks.
      9. Distinguishes general visual change from crack-specific change.
    """
    # 1. Parameter Validation
    mode = alignment_mode.lower().strip()
    if mode not in VALID_ALIGNMENT_MODES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "INVALID_ALIGNMENT_MODE",
                "message": f"Unknown alignment mode '{alignment_mode}'. Accepted: {sorted(VALID_ALIGNMENT_MODES)}.",
            },
        )

    if not (0.0 < confidence <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "INVALID_CONFIDENCE",
                "message": f"Confidence threshold must be in (0, 1], got {confidence}.",
            },
        )

    if not (0.0 < iou <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "INVALID_IOU",
                "message": f"IoU threshold must be in (0, 1], got {iou}.",
            },
        )

    if not (0 <= tolerance_px <= 10):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "INVALID_TOLERANCE",
                "message": f"Spatial tolerance must be between 0 and 10 pixels, got {tolerance_px}.",
            },
        )

    # 2. Upload file validation
    try:
        before_bytes = await read_and_validate_file(before, "before")
        after_bytes = await read_and_validate_file(after, "after")
    finally:
        await before.close()
        await after.close()

    # 3. Execution via Hybrid Service
    try:
        result = await hybrid_comparison_service.compare_hybrid(
            before_bytes=before_bytes,
            before_type=before.content_type or "image/jpeg",
            after_bytes=after_bytes,
            after_type=after.content_type or "image/jpeg",
            alignment_mode=mode,
            confidence=confidence,
            iou=iou,
            tolerance_px=tolerance_px,
        )
        return result

    except DamageServiceUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": exc.code,
                "message": exc.message,
                **exc.details,
            },
        )

    except DamageServiceTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error": exc.code,
                "message": exc.message,
                **exc.details,
            },
        )

    except DamageServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "error": exc.code,
                "message": exc.message,
                **exc.details,
            },
        )

    except ValueError as ve:
        err_msg = str(ve)
        if "alignment failed" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "ALIGNMENT_FAILED",
                    "message": err_msg,
                    "attempted_methods": (
                        ["SIFT", "AKAZE", "ORB"] if mode == "auto" else [mode.upper()]
                    ),
                    "failure_reasons": [
                        "Insufficient matching feature points between before and after photographs."
                    ],
                    "guidance": (
                        "Ensure both photographs show the same building surface from a similar "
                        "viewpoint with visible shared texture. If images are already geometrically aligned, "
                        "you may specify alignment_mode='direct'."
                    ),
                },
            )
        if "overlap is insufficient" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "INSUFFICIENT_OVERLAP",
                    "message": err_msg,
                    "guidance": (
                        "The common visible area between the two photographs is too small to yield "
                        "a reliable comparison. Re-capture with greater field-of-view overlap."
                    ),
                },
            )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "COMPARISON_VALIDATION_ERROR",
                "message": err_msg,
            },
        )

    except Exception as exc:
        logger.exception("Unexpected error in hybrid damage comparison")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred while performing hybrid damage comparison.",
            },
        )
