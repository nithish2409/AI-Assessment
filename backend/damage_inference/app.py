"""
FastAPI application for the isolated YOLO Damage Inference Microservice.
Runs under backend/venv_yolo on http://127.0.0.1:8001.
"""
from contextlib import asynccontextmanager
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from damage_inference.config import HOST, PORT, DEFAULT_CONF, DEFAULT_IOU, MAX_UPLOAD_SIZE
from damage_inference.schemas import (
    HealthResponse,
    ReadinessResponse,
    DamageInferenceResponse,
    MicroserviceInspectionResponse,
)
from damage_inference.service import inference_service
from damage_inference.inspection_service import inspection_service

logger = logging.getLogger("damage_inference.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load YOLO model checkpoint and warmup GPU
    logger.info("Damage Inference Microservice starting up...")
    success = inference_service.initialize()
    if not success:
        logger.error("Model initialization failed on startup. Service will report not ready.")

    # Initialize multi-model inspection pipeline
    insp_success = inspection_service.initialize()
    if not insp_success:
        logger.warning("InspectionService initialization encountered warnings; some models may lazy-load.")

    yield
    # Shutdown
    logger.info("Damage Inference Microservice shutting down...")


app = FastAPI(
    title="YOLO Damage Inference Microservice",
    description="Dedicated microservice for YOLO26s-seg concrete crack segmentation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """Liveness probe: verifies microservice process is running."""
    return HealthResponse()


@app.get("/ready", response_model=ReadinessResponse, tags=["Diagnostics"])
async def readiness_check():
    """Readiness probe: verifies YOLO checkpoint is loaded and ready for inference."""
    readiness = inference_service.get_readiness()
    if not readiness.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=readiness.model_dump(),
        )
    return readiness


@app.post(
    "/infer",
    response_model=DamageInferenceResponse,
    tags=["Inference"],
    summary="Segment concrete cracks in an uploaded image",
)
async def infer_crack_segmentation(
    image: UploadFile = File(..., description="Uploaded photograph (JPEG, PNG, WebP)"),
    confidence: float = Form(
        DEFAULT_CONF, description="Detection confidence threshold in (0, 1]"
    ),
    iou: float = Form(
        DEFAULT_IOU, description="NMS IoU threshold in (0, 1]"
    ),
):
    """
    Accept an uploaded image, run YOLO26s-seg crack segmentation, and return
    original, binary mask, visual overlay, and detection statistics.
    """
    # 1. Parameter bounds validation
    if not (0.0 < confidence <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Confidence threshold must be in (0, 1], got {confidence}.",
        )
    if not (0.0 < iou <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"IoU threshold must be in (0, 1], got {iou}.",
        )

    # 2. Read limited file content to prevent memory exhaustion
    try:
        content = await image.read(MAX_UPLOAD_SIZE + 1)
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Maximum file size is 10 MB.",
            )
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Uploaded image is empty.",
            )

        # 3. Delegate to segmentation service
        return await inference_service.segment_image(
            image_data=content,
            content_type=image.content_type,
            confidence=confidence,
            iou=iou,
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except RuntimeError as re:
        err_msg = str(re)
        if "not ready" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=err_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=err_msg,
        )
    finally:
        await image.close()


@app.get("/inspect/status", tags=["Diagnostics"])
async def inspection_service_status():
    """Return status of multi-model inspection pipelines."""
    return {
        "status": "ready" if inspection_service.is_initialized else "initializing",
        "device": inspection_service.device,
        "cuda_available": torch.cuda.is_available() if "torch" in globals() else True,
        "models": {
            "crack_segmentation": inspection_service.crack_model is not None,
            "industrial_defect": inspection_service.industrial_model is not None,
            "yolo_world": inspection_service.world_model is not None,
            "codebrim_classifier": inspection_service.classifier_service is not None,
            "easyocr": inspection_service.ocr_reader is not None,
        }
    }


@app.post(
    "/inspect/building",
    response_model=MicroserviceInspectionResponse,
    tags=["Inspection"],
    summary="Execute building inspection: ResNet18 classifier + YOLO26s crack segmentor + YOLO-World components",
)
async def inspect_building(
    image: UploadFile = File(..., description="Uploaded building photograph"),
    confidence: float = Form(DEFAULT_CONF, description="Detection confidence threshold"),
    iou: float = Form(DEFAULT_IOU, description="NMS IoU threshold"),
):
    if not (0.0 < confidence <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Confidence threshold must be in (0, 1], got {confidence}.",
        )
    if not (0.0 < iou <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"IoU threshold must be in (0, 1], got {iou}.",
        )

    content = await image.read(MAX_UPLOAD_SIZE + 1)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE // (1024*1024)} MB.",
        )

    try:
        response = await inspection_service.run_building_inspection(
            image_bytes=content,
            confidence=confidence,
            iou=iou,
        )
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.exception("Building inspection failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    finally:
        await image.close()


@app.post(
    "/inspect/industrial",
    response_model=MicroserviceInspectionResponse,
    tags=["Inspection"],
    summary="Execute industrial inspection: YOLO26s defects + YOLO-World components + EasyOCR text",
)
async def inspect_industrial(
    image: UploadFile = File(..., description="Uploaded industrial equipment photograph"),
    confidence: float = Form(DEFAULT_CONF, description="Detection confidence threshold"),
    iou: float = Form(DEFAULT_IOU, description="NMS IoU threshold"),
):
    if not (0.0 < confidence <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Confidence threshold must be in (0, 1], got {confidence}.",
        )
    if not (0.0 < iou <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"IoU threshold must be in (0, 1], got {iou}.",
        )

    content = await image.read(MAX_UPLOAD_SIZE + 1)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE // (1024*1024)} MB.",
        )

    try:
        response = await inspection_service.run_industrial_inspection(
            image_bytes=content,
            confidence=confidence,
            iou=iou,
        )
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.exception("Industrial inspection failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    finally:
        await image.close()


def main():
    uvicorn.run(
        "damage_inference.app:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
