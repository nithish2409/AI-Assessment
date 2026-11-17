"""
Pydantic Schemas for Unified Building and Industrial Inspection API.
"""
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field


class ComponentFinding(BaseModel):
    name: str = Field(..., description="Detected structural or equipment component name (e.g. column, wall, pipe)")
    confidence: float = Field(..., description="Component detection confidence in (0, 1]")
    bbox: List[int] = Field(..., description="Bounding box in pixel coordinates [x1, y1, x2, y2]")


class DefectFinding(BaseModel):
    defect_id: str = Field(..., description="Unique defect finding identifier, e.g. DEF-001")
    category: str = Field(..., description="Defect category (crack, spallation, corrosion, etc.)")
    source_model: str = Field(..., description="AI model that identified this defect")
    finding_type: Literal["segmentation", "bounding_box", "classification"] = Field(
        ..., description="Granularity of detection: pixel segmentation, bounding box, or image classification"
    )
    confidence: float = Field(..., description="Model confidence or sigmoid probability in (0, 1]")
    bbox: Optional[List[int]] = Field(default=None, description="Bounding box [x1, y1, x2, y2] if localized")
    area_pixels: Optional[int] = Field(default=None, description="Pixel area if localized")
    polygon: Optional[List[List[float]]] = Field(default=None, description="Segmentation vertices if available")
    associated_component: str = Field(default="unknown", description="Associated structural or equipment component")


class OCRFinding(BaseModel):
    text: str = Field(..., description="Extracted text string from nameplate or tag")
    confidence: float = Field(..., description="OCR recognition confidence in (0, 1]")
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")


class VisualSeveritySummary(BaseModel):
    visual_severity: Literal["low", "moderate", "high", "undetermined"] = Field(
        ..., description="Visual screening severity level"
    )
    inspection_priority: Literal["monitor", "review_recommended", "immediate_action", "further_investigation_required"] = Field(
        ..., description="Recommended inspection prioritization status"
    )
    method: str = Field(default="rule_based_visual_screening", description="Assessment method")
    evidence: Dict[str, Any] = Field(default={}, description="Input features and measurements evaluated")
    rationale: List[str] = Field(default=[], description="Human-readable decision rationale bullets")
    limitations: List[str] = Field(default=[], description="Explicit engineering limitations and caveats")
    disclaimer: str = Field(..., description="Mandatory safety and legal screening disclaimer")


class InspectionImages(BaseModel):
    original: str = Field(..., description="Base64-encoded original image PNG")
    mask: str = Field(..., description="Base64-encoded binary damage mask PNG")
    overlay: str = Field(..., description="Base64-encoded visual overlay image PNG with highlights")


class InspectionAnalysisResponse(BaseModel):
    inspection_id: str = Field(..., description="Unique inspection transaction identifier")
    timestamp: str = Field(..., description="ISO 8601 inspection timestamp")
    domain: Literal["building", "industrial"] = Field(..., description="Active inspection domain")
    status: str = Field(default="completed", description="Analysis execution status")
    image_width: int = Field(..., description="Original image width in pixels")
    image_height: int = Field(..., description="Original image height in pixels")
    defects_count: int = Field(..., description="Total number of discrete defects identified")
    defects: List[DefectFinding] = Field(default=[], description="Detailed defect findings")
    components: List[ComponentFinding] = Field(default=[], description="Localized components")
    ocr_results: List[OCRFinding] = Field(default=[], description="OCR text detections (industrial domain)")
    classification_summary: Optional[Dict[str, Any]] = Field(default=None, description="Multi-label classification results")
    crack_coverage_percentage: float = Field(default=0.0, description="Relative surface damage area percentage")
    visual_severity: VisualSeveritySummary = Field(..., description="Deterministic visual severity assessment")
    images: InspectionImages = Field(..., description="Base64-encoded visual inspection artifacts")
    processing_time_ms: float = Field(..., description="Total end-to-end processing latency in milliseconds")
    warnings: List[str] = Field(default=[], description="Operational warnings or notices")
    disclaimer: str = Field(..., description="Inspection screening disclaimer")
