"""
Pydantic schemas for YOLO Damage Inference Microservice requests and responses.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "YOLO Crack Inference Microservice"


class ReadinessResponse(BaseModel):
    ready: bool
    model: str
    checkpoint: str
    task: str
    classes: Dict[str, str]
    device: str
    cuda_available: bool
    message: str


class DetectionItem(BaseModel):
    class_id: int = Field(..., description="Numeric class identifier")
    class_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., description="Detection confidence score in (0, 1]")
    bbox: List[int] = Field(..., description="Bounding box in original image coords: [x1, y1, x2, y2]")
    area_pixels: int = Field(..., description="Mask area for this detection in pixels")
    polygon: Optional[List[List[float]]] = Field(
        default=None, description="Optional polygon coordinate list [[x, y], ...]"
    )


class ImagesPayload(BaseModel):
    original: str = Field(..., description="Base64-encoded original image (PNG)")
    mask: str = Field(..., description="Base64-encoded binary mask (PNG, crack=255, bg=0)")
    overlay: str = Field(..., description="Base64-encoded visual overlay image (PNG)")


class InferenceMeta(BaseModel):
    device: str = Field(..., description="Inference device used, e.g. 'cuda:0' or 'cpu'")
    duration_ms: float = Field(..., description="Total processing duration in milliseconds")
    preprocess_ms: Optional[float] = Field(None, description="Image preprocessing and letterboxing duration")
    inference_ms: Optional[float] = Field(None, description="PyTorch model forward pass duration")
    postprocess_ms: Optional[float] = Field(None, description="Mask extraction, union, and overlay duration")


class DamageInferenceResponse(BaseModel):
    status: str = Field(default="completed", description="Inference outcome status")
    model: str = Field(default="YOLO26s-seg", description="Model architecture identifier")
    class_names: List[str] = Field(default=["crack"], description="Class names supported by model")
    image_width: int = Field(..., description="Original image width in pixels")
    image_height: int = Field(..., description="Original image height in pixels")
    detections_count: int = Field(..., description="Number of crack instances detected")
    crack_coverage_percentage: float = Field(
        ..., description="Percentage of total image pixels occupied by crack mask union"
    )
    detections: List[DetectionItem] = Field(default=[], description="List of detected crack instances")
    images: ImagesPayload = Field(..., description="Base64-encoded visual assets")
    inference: InferenceMeta = Field(..., description="Device and timing telemetry")
    warnings: List[str] = Field(default=[], description="Non-fatal warnings or operational notices")


# -------------------------------------------------------------
# Unified Inspection Schemas (Phase 3 Multi-Model Integration)
# -------------------------------------------------------------
class ComponentItem(BaseModel):
    name: str = Field(..., description="Detected component category (e.g. wall, column, pipe)")
    confidence: float = Field(..., description="Confidence score in (0, 1]")
    bbox: List[int] = Field(..., description="Bounding box in pixel coordinates: [x1, y1, x2, y2]")


class DefectItem(BaseModel):
    category: str = Field(..., description="Defect category (e.g. crack, spallation, corrosion)")
    source_model: str = Field(..., description="Source model that identified the defect")
    finding_type: str = Field(..., description="Identification method: 'segmentation', 'bounding_box', or 'classification'")
    confidence: float = Field(..., description="Model detection confidence or probability in (0, 1]")
    bbox: Optional[List[int]] = Field(default=None, description="Bounding box [x1, y1, x2, y2] if localized")
    area_pixels: Optional[int] = Field(default=None, description="Mask or bounding-box area in pixels if localized")
    polygon: Optional[List[List[float]]] = Field(default=None, description="Segmentation vertices if segmented")
    associated_component: Optional[str] = Field(default="unknown", description="Associated structural/equipment component")


class OCRItem(BaseModel):
    text: str = Field(..., description="Extracted text snippet")
    confidence: float = Field(..., description="OCR detection confidence score in (0, 1]")
    bbox: List[int] = Field(..., description="Bounding box in pixel coordinates: [x1, y1, x2, y2]")


class ClassificationSummary(BaseModel):
    classes: List[str] = Field(default=[], description="List of predicted defect classes above threshold")
    probabilities: Dict[str, float] = Field(default={}, description="Raw sigmoid probabilities per class")
    thresholds: Dict[str, float] = Field(default={}, description="Calibrated decision thresholds applied")


class MicroserviceInspectionResponse(BaseModel):
    status: str = Field(default="completed", description="Inspection execution status")
    domain: str = Field(..., description="Inspection domain: 'building' or 'industrial'")
    image_width: int = Field(..., description="Image width in pixels")
    image_height: int = Field(..., description="Image height in pixels")
    classification: Optional[ClassificationSummary] = Field(default=None, description="Multi-label image classification results")
    defects: List[DefectItem] = Field(default=[], description="List of all detected and categorized defects")
    components: List[ComponentItem] = Field(default=[], description="Detected structural or equipment components")
    ocr_items: Optional[List[OCRItem]] = Field(default=[], description="OCR text snippets detected in industrial inspection")
    crack_coverage_percentage: Optional[float] = Field(default=0.0, description="Relative crack area percentage if segmented")
    images: ImagesPayload = Field(..., description="Base64 encoded original, mask, and overlay images")
    inference: InferenceMeta = Field(..., description="Hardware and timing telemetry")
    warnings: List[str] = Field(default=[], description="Limitations, operational notices, and disclaimers")

