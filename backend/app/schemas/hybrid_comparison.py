"""
Pydantic schemas for Phase 4: Hybrid Damage Comparison API.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HybridAlignmentInfo(BaseModel):
    status: str = Field(..., description="Alignment status: accepted, uncertain, identical, direct, rejected")
    method: str = Field(..., description="Feature alignment method: SIFT, AKAZE, ORB, DIRECT, NONE")
    confidence: Optional[float] = Field(None, description="Heuristic alignment quality score in [0, 1]")
    inlier_count: Optional[int] = Field(None, description="Number of RANSAC inlier feature correspondences")
    inlier_ratio: Optional[float] = Field(None, description="Ratio of inliers to good feature matches")
    reprojection_error: Optional[float] = Field(None, description="Mean reprojection error in pixels")
    valid_overlap_pixels: int = Field(..., description="Number of valid corresponding pixels in reference frame")
    valid_overlap_percentage: float = Field(..., description="Valid overlap percentage of after image canvas")
    warnings: List[str] = Field(default=[], description="Alignment-related caveats or warnings")
    method_used: Optional[str] = Field(None, description="Feature alignment method/detector used")
    detector: Optional[str] = Field(None, description="Detector used")
    geometric_status: Optional[str] = Field(None, description="Geometric verification status")


class HybridVisualComparison(BaseModel):
    ssim_score: float = Field(..., description="Mean structural similarity index computed strictly over valid overlap")
    visual_changed_pixels: int = Field(..., description="Count of general visual change pixels from SSIM thresholding")
    visual_change_percentage: float = Field(..., description="Percentage of valid overlap exhibiting visual difference")
    changed_regions_count: int = Field(..., description="Number of connected visual change contours")


class HybridCrackComparison(BaseModel):
    before_crack_pixels: int = Field(..., description="Crack pixels in aligned before mask within valid overlap")
    after_crack_pixels: int = Field(..., description="Crack pixels in after mask within valid overlap")
    before_crack_coverage_percentage: float = Field(
        ..., description="Crack coverage percentage in before image relative to valid overlap"
    )
    after_crack_coverage_percentage: float = Field(
        ..., description="Crack coverage percentage in after image relative to valid overlap"
    )
    coverage_change_percentage_points: float = Field(
        ..., description="Change in crack coverage in percentage points (after - before)"
    )
    persistent_pixels: int = Field(..., description="Crack pixels present in both before and after masks")
    newly_observed_pixels: int = Field(..., description="Crack pixels present in after mask but not before")
    no_longer_observed_pixels: int = Field(..., description="Crack pixels present in before mask but not after")
    raw_change_pixels: int = Field(..., description="Total crack pixel differences (symmetric difference)")
    raw_change_percentage: float = Field(..., description="Raw crack change percentage relative to valid overlap")
    tolerance_px: int = Field(..., description="Spatial tolerance radius applied in pixels")
    tolerance_adjusted_new_pixels: int = Field(
        ..., description="Newly observed crack pixels outside the spatial tolerance buffer"
    )
    tolerance_adjusted_no_longer_observed_pixels: int = Field(
        ..., description="No-longer-observed crack pixels outside the spatial tolerance buffer"
    )
    potential_new_components: int = Field(
        ..., description="Count of after crack components with no spatial proximity to before cracks"
    )
    potential_expanded_components: int = Field(
        ..., description="Count of crack components that overlap before cracks with >= 15% area growth"
    )
    persistent_components: int = Field(
        ..., description="Count of crack components persisting with stable area"
    )
    no_longer_observed_components: int = Field(
        ..., description="Count of before crack components not detected in after image"
    )
    progression_label: str = Field(
        default="No substantial predicted change detected",
        description="Categorical progression classification: 'No substantial predicted change detected', 'Potential newly observed crack regions', 'Potential expansion of detected crack regions', 'Comparison inconclusive'",
    )
    calibrated_noise_floor_pp: float = Field(
        default=0.89,
        description="Calibrated empirical coverage-change noise floor in percentage points (90th percentile)",
    )
    is_above_noise_floor: bool = Field(
        default=False,
        description="Whether absolute coverage change exceeds the calibrated noise floor",
    )
    uncertainty_level: str = Field(
        default="low",
        description="Confidence/uncertainty level: 'low', 'medium', 'high'",
    )
    interpretation_summary: str = Field(
        default="",
        description="Qualified interpretation explaining whether observed differences reflect plausible crack progression or fall within empirical alignment/segmentation noise",
    )


class HybridModelMeta(BaseModel):
    name: str = Field(default="YOLO26s-seg", description="Model architecture")
    class_names: List[str] = Field(default=["crack"], description="Detected defect classes")
    confidence_threshold: float = Field(default=0.25, description="Confidence threshold applied")
    iou_threshold: float = Field(default=0.45, description="NMS IoU threshold applied")


class HybridImagesPayload(BaseModel):
    aligned_before: str = Field(..., description="Base64-encoded aligned before image PNG")
    visual_difference: str = Field(..., description="Base64-encoded SSIM visual difference heatmap PNG")
    before_crack_mask: str = Field(..., description="Base64-encoded aligned before crack mask PNG")
    after_crack_mask: str = Field(..., description="Base64-encoded after crack mask PNG")
    newly_observed_mask: str = Field(..., description="Base64-encoded newly observed crack mask PNG")
    persistent_mask: str = Field(..., description="Base64-encoded persistent crack mask PNG")
    no_longer_observed_mask: str = Field(..., description="Base64-encoded no-longer-observed crack mask PNG")
    tolerance_adjusted_new_mask: Optional[str] = Field(
        default=None,
        description="Base64-encoded tolerance-adjusted newly observed crack mask PNG",
    )
    hybrid_overlay: str = Field(..., description="Base64-encoded color-coded composite overlay PNG")


class HybridTimingMeta(BaseModel):
    total_duration_ms: float = Field(..., description="End-to-end processing latency in milliseconds")
    alignment_ms: float = Field(..., description="Feature alignment and homography computation latency")
    ssim_ms: float = Field(..., description="SSIM visual comparison latency")
    yolo_before_ms: float = Field(..., description="YOLO inference latency on before photograph")
    yolo_after_ms: float = Field(..., description="YOLO inference latency on after photograph")
    crack_comparison_ms: float = Field(..., description="Spatial mask comparison and component analysis latency")
    visualization_ms: float = Field(..., description="Overlay rendering and base64 PNG encoding latency")


class HybridComparisonResponse(BaseModel):
    status: str = Field(default="completed", description="Overall execution status")
    image_width: int = Field(..., description="Reference after image width in pixels")
    image_height: int = Field(..., description="Reference after image height in pixels")
    alignment: HybridAlignmentInfo = Field(..., description="Geometric alignment and overlap metrics")
    visual_comparison: HybridVisualComparison = Field(..., description="SSIM visual difference metrics")
    crack_comparison: HybridCrackComparison = Field(..., description="Crack-specific progression metrics")
    model: HybridModelMeta = Field(..., description="Inference model metadata")
    images: HybridImagesPayload = Field(..., description="Base64-encoded visual output assets")
    timing: Optional[HybridTimingMeta] = Field(None, description="Detailed stage-by-stage latency telemetry")
    warnings: List[str] = Field(default=[], description="Operational notices and methodological qualifications")
