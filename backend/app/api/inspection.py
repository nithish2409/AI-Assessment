"""
FastAPI Router for Unified AI Inspection & Visual Severity Assessment.
Endpoints:
  POST /api/inspection/analyze     - Multi-model inspection with visual severity screening
  POST /api/inspection/report/pdf  - Generate and download formal PDF inspection report
  GET  /api/inspection/health      - Check microservice model readiness
"""
import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Literal
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Response
from fastapi.responses import Response

from app.schemas.inspection import (
    InspectionAnalysisResponse,
    ComponentFinding,
    DefectFinding,
    OCRFinding,
    VisualSeveritySummary,
    InspectionImages,
)
from app.services.comparison_service import MAX_FILE_SIZE, MAX_PIXELS, ALLOWED_TYPES
from app.services.inspection_client import inspection_client
from app.services.visual_severity_engine import VisualSeverityEngine, STANDARD_DISCLAIMER
from app.services.inspection_report_generator import generate_pdf_report
from app.services.damage_client import DamageServiceError, DamageServiceUnavailableError, DamageServiceTimeoutError

logger = logging.getLogger("api.inspection")
router = APIRouter()


async def read_and_validate_inspection_upload(file: UploadFile) -> bytes:
    """Validate uploaded inspection image size, MIME type, and dimensions."""
    if not file.content_type or file.content_type.lower().split(";")[0].strip() not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported image format. Only JPEG, PNG, and WebP are accepted.",
        )

    data = await file.read(MAX_FILE_SIZE + 1)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded image file is empty.",
        )
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded image exceeds maximum allowed size of {MAX_FILE_SIZE // (1024*1024)} MB.",
        )
    return data


@router.get("/health", tags=["Inspection"], summary="Check health of AI inspection microservice and models")
async def get_inspection_health():
    """Verify operational readiness of vision models in GPU microservice."""
    try:
        health_info = await inspection_client.check_health()
        return {"status": "healthy", "service": "AI Inspection Gateway", "microservice": health_info}
    except DamageServiceUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.details)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/analyze",
    response_model=InspectionAnalysisResponse,
    tags=["Inspection"],
    summary="Execute multi-model building or industrial inspection with visual severity assessment",
)
async def analyze_inspection(
    image: UploadFile = File(..., description="Inspection photograph (JPEG, PNG, WebP)"),
    domain: Literal["building", "industrial"] = Form("building", description="Inspection domain"),
    inspection_id: Optional[str] = Form(None, description="Optional custom inspection identifier"),
    confidence: float = Form(0.25, description="Detection confidence threshold in (0, 1]"),
    iou: float = Form(0.45, description="NMS IoU threshold in (0, 1]"),
):
    """
    Accepts an uploaded image, executes multi-model inspection on the dedicated GPU microservice,
    evaluates deterministic visual severity, and returns structured findings.
    """
    t_start = time.perf_counter()

    # 1. Parameter validations
    if domain not in ("building", "industrial"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid domain '{domain}'. Must be 'building' or 'industrial'.",
        )
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

    # 2. File size and header validation
    image_bytes = await read_and_validate_inspection_upload(image)
    active_insp_id = inspection_id or f"INSP-{domain.upper()[:4]}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
    iso_timestamp = datetime.now(timezone.utc).isoformat()

    try:
        # 3. Call dedicated GPU microservice
        if domain == "building":
            raw_res = await inspection_client.inspect_building(
                image_bytes=image_bytes,
                confidence=confidence,
                iou=iou,
            )
        else:
            raw_res = await inspection_client.inspect_industrial(
                image_bytes=image_bytes,
                confidence=confidence,
                iou=iou,
            )

        # 4. Map components
        components: List[ComponentFinding] = []
        for c in raw_res.get("components", []):
            components.append(ComponentFinding(
                name=c["name"],
                confidence=c["confidence"],
                bbox=c["bbox"]
            ))

        # 5. Map defect findings and assign IDs
        defects: List[DefectFinding] = []
        for idx, d in enumerate(raw_res.get("defects", []), start=1):
            defects.append(DefectFinding(
                defect_id=f"DEF-{idx:03d}",
                category=d["category"],
                source_model=d["source_model"],
                finding_type=d["finding_type"],
                confidence=d["confidence"],
                bbox=d.get("bbox"),
                area_pixels=d.get("area_pixels"),
                polygon=d.get("polygon"),
                associated_component=d.get("associated_component", "unknown"),
            ))

        # 6. Map OCR results
        ocr_results: List[OCRFinding] = []
        for o in raw_res.get("ocr_items", []):
            ocr_results.append(OCRFinding(
                text=o["text"],
                confidence=o["confidence"],
                bbox=o["bbox"]
            ))

        # 7. Evaluate Visual Severity via Deterministic Engine
        crack_cov = float(raw_res.get("crack_coverage_percentage", 0.0) or 0.0)
        component_names = [c.name for c in components]

        if domain == "building":
            clf = raw_res.get("classification") or {}
            clf_classes = clf.get("classes", [])
            clf_probs = clf.get("probabilities", {})
            crack_count = sum(1 for d in defects if d.category.lower() == "crack")

            severity_eval = VisualSeverityEngine.evaluate_building(
                classified_categories=clf_classes,
                classified_probabilities=clf_probs,
                crack_coverage_pct=crack_cov,
                crack_count=crack_count,
                associated_components=component_names,
            )
        else:
            ind_defect_dicts = [{"category": d.category, "confidence": d.confidence} for d in defects]
            severity_eval = VisualSeverityEngine.evaluate_industrial(
                detected_defects=ind_defect_dicts,
                relative_defect_area_pct=crack_cov,
                associated_components=component_names,
                has_ocr_text=len(ocr_results) > 0,
            )

        visual_severity_summary = VisualSeveritySummary(
            visual_severity=severity_eval.visual_severity,
            inspection_priority=severity_eval.inspection_priority,
            method=severity_eval.method,
            evidence=severity_eval.evidence,
            rationale=severity_eval.rationale,
            limitations=severity_eval.limitations,
            disclaimer=severity_eval.disclaimer,
        )

        total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return InspectionAnalysisResponse(
            inspection_id=active_insp_id,
            timestamp=iso_timestamp,
            domain=domain,
            status="completed",
            image_width=raw_res["image_width"],
            image_height=raw_res["image_height"],
            defects_count=len(defects),
            defects=defects,
            components=components,
            ocr_results=ocr_results,
            classification_summary=raw_res.get("classification"),
            crack_coverage_percentage=crack_cov,
            visual_severity=visual_severity_summary,
            images=InspectionImages(
                original=raw_res["images"]["original"],
                mask=raw_res["images"]["mask"],
                overlay=raw_res["images"]["overlay"],
            ),
            processing_time_ms=total_time_ms,
            warnings=raw_res.get("warnings", []),
            disclaimer=STANDARD_DISCLAIMER,
        )

    except DamageServiceUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.details)
    except DamageServiceTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=exc.details)
    except DamageServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": exc.code, "message": exc.message})
    except Exception as exc:
        logger.exception("Inspection analysis encountered internal error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    finally:
        await image.close()


@router.post(
    "/report/pdf",
    tags=["Inspection"],
    summary="Generate and download downloadable ReportLab PDF inspection report",
)
async def download_pdf_report(payload: InspectionAnalysisResponse):
    """Compile structured inspection findings into an official PDF document."""
    try:
        pdf_bytes = generate_pdf_report(payload.model_dump())
        filename = f"{payload.inspection_id}_report.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as exc:
        logger.exception("PDF generation failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"PDF generation error: {str(exc)}")
