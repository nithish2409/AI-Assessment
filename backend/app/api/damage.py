"""
FastAPI Router for Building Damage Detection & Crack Segmentation.
Endpoints:
  POST /api/damage/segment
  GET  /api/damage/health
"""
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
import cv2
import numpy as np

from app.schemas.damage import DamageSegmentResponse, DamageServiceStatus
from app.services.comparison_service import MAX_FILE_SIZE, MAX_PIXELS, ALLOWED_TYPES
from app.services.damage_client import (
    damage_client,
    DamageServiceError,
    DamageServiceUnavailableError,
    DamageServiceTimeoutError,
)

logger = logging.getLogger("api.damage")
router = APIRouter()


async def read_and_validate_upload(file: UploadFile) -> bytes:
    """Validate upload headers, size limit, and content structure."""
    if not file.content_type or file.content_type.lower().split(";")[0].strip() not in ALLOWED_TYPES:
        raise ValueError("Only JPEG, PNG and WebP images are supported.")

    data = await file.read(MAX_FILE_SIZE + 1)
    if not data:
        raise ValueError("Uploaded image is empty.")

    if len(data) > MAX_FILE_SIZE:
        raise ValueError("Maximum file size is 10 MB.")

    # Preliminary dimension verification to prevent decompression bombs
    raw = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid or corrupted image.")

    h, w = image.shape[:2]
    if h * w > MAX_PIXELS:
        raise ValueError("Image exceeds the 20-megapixel limit.")

    return data


@router.get(
    "/health",
    response_model=DamageServiceStatus,
    tags=["Damage Detection"],
    summary="Check status of the damage segmentation microservice",
)
async def get_damage_service_status():
    """Verify connectivity to the internal YOLO inference microservice."""
    try:
        readiness = await damage_client.check_readiness()
        return DamageServiceStatus(
            status="healthy",
            service_url=damage_client.base_url,
            is_available=True,
            details=readiness,
        )
    except DamageServiceUnavailableError as exc:
        return DamageServiceStatus(
            status="unavailable",
            service_url=damage_client.base_url,
            is_available=False,
            details={"error": exc.code, "message": exc.message, **exc.details},
        )
    except Exception as exc:
        return DamageServiceStatus(
            status="degraded",
            service_url=damage_client.base_url,
            is_available=False,
            details={"error": "HEALTH_CHECK_ERROR", "message": str(exc)},
        )


@router.post(
    "/segment",
    response_model=DamageSegmentResponse,
    tags=["Damage Detection"],
    summary="Segment concrete cracks in an uploaded building photograph",
)
async def segment_damage(
    image: UploadFile = File(..., description="Building photograph to analyze (JPEG, PNG, WebP)"),
    confidence: float = Form(0.25, description="Detection confidence threshold in (0, 1]"),
    iou: float = Form(0.45, description="NMS IoU threshold in (0, 1]"),
):
    """
    Accepts an uploaded image, executes YOLO26s-seg crack segmentation on the
    dedicated inference microservice, and returns binary masks, overlays, and
    crack coverage percentage.
    """
    # 1. Parameter range validation
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

    # 2. Image upload validation
    try:
        image_bytes = await read_and_validate_upload(image)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "INVALID_IMAGE_UPLOAD",
                "message": str(ve),
            },
        )
    finally:
        await image.close()

    # 3. Delegate to internal YOLO inference microservice
    try:
        response_data = await damage_client.segment_image(
            image_bytes=image_bytes,
            filename=image.filename or "upload.jpg",
            content_type=image.content_type or "image/jpeg",
            confidence=confidence,
            iou=iou,
        )
        return response_data

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
    except Exception as exc:
        logger.exception("Unexpected error in damage segmentation route")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred while processing the damage segmentation request.",
            },
        )
