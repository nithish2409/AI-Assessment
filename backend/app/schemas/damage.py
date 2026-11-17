"""
Pydantic schemas for the main FastAPI Building Damage API endpoints.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DamageDetection(BaseModel):
    class_id: int = Field(..., description="Numeric class identifier (0 for crack)")
    class_name: str = Field(..., description="Class name, e.g. 'crack'")
    confidence: float = Field(..., description="Confidence score in (0, 1]")
    bbox: List[int] = Field(..., description="Bounding box in pixel coordinates: [x1, y1, x2, y2]")
    area_pixels: int = Field(..., description="Mask area in pixels")
    polygon: Optional[List[List[float]]] = Field(
        default=None, description="Optional polygon coordinate vertices [[x, y], ...]"
    )


class DamageImages(BaseModel):
    original: str = Field(..., description="Base64-encoded original PNG image")
    mask: str = Field(..., description="Base64-encoded binary mask PNG (background=0, crack=255)")
    overlay: str = Field(..., description="Base64-encoded visual overlay PNG with highlighted cracks")


class DamageInferenceTiming(BaseModel):
    device: str = Field(..., description="Inference compute device, e.g. 'cuda:0' or 'cpu'")
    duration_ms: float = Field(..., description="Internal inference pipeline latency in milliseconds")
    preprocess_ms: Optional[float] = Field(None, description="Image decode and preprocess duration")
    inference_ms: Optional[float] = Field(None, description="PyTorch forward pass duration")
    postprocess_ms: Optional[float] = Field(None, description="Mask extraction and overlay duration")


class DamageSegmentResponse(BaseModel):
    status: str = Field(default="completed", description="Inference completion status")
    model: str = Field(default="YOLO26s-seg", description="Model architecture name")
    class_names: List[str] = Field(default=["crack"], description="Active defect classes")
    image_width: int = Field(..., description="Image width in pixels")
    image_height: int = Field(..., description="Image height in pixels")
    detections_count: int = Field(..., description="Number of detected crack instances")
    crack_coverage_percentage: float = Field(
        ..., description="Percentage of total image area occupied by crack mask union"
    )
    detections: List[DamageDetection] = Field(default=[], description="List of crack detections")
    images: DamageImages = Field(..., description="Base64-encoded image assets")
    inference: DamageInferenceTiming = Field(..., description="Hardware and latency telemetry")
    warnings: List[str] = Field(default=[], description="Operational warnings or notices")


class DamageServiceStatus(BaseModel):
    status: str
    service_url: str
    is_available: bool
    details: Optional[Dict] = None
