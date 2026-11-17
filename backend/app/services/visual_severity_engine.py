"""
Deterministic Rule-Based Visual Severity & Prioritization Engine.
Provides explainable, conservative visual screening assessments for building and industrial inspections.

IMPORTANT DISCLAIMER:
Automated visual inspection results are heuristic screening estimates intended solely for
triage and inspection prioritization. They do NOT constitute certified structural, mechanical,
or safety engineering evaluations. Qualified human inspection is mandatory for regulatory decisions.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

# Configurable heuristic screening thresholds (documented as unvalidated heuristic guidelines)
HEURISTIC_THRESHOLDS = {
    "building": {
        "crack_coverage_high_pct": 1.50,     # >1.50% total image area crack coverage
        "crack_coverage_moderate_pct": 0.20, # 0.20% - 1.50% crack coverage
        "multi_defect_high_count": 3,        # 3 or more concurrent damage classes
        "min_detection_confidence": 0.25,    # Minimum confidence to consider a finding valid
    },
    "industrial": {
        "corrosion_area_high_pct": 8.0,      # >8.0% bounding box area
        "corrosion_area_moderate_pct": 1.0,  # 1.0% - 8.0% bounding box area
        "min_detection_confidence": 0.25,    # Minimum confidence to consider a finding valid
    }
}

STANDARD_DISCLAIMER = (
    "Automated visual inspection results are preliminary heuristic estimates intended solely for "
    "screening and triage prioritization. They do not constitute a certified structural, mechanical, "
    "or safety assessment. Imminent failure or structural load-bearing capacity cannot be determined "
    "from 2D image appearance alone. Qualified professional human inspection is required."
)


@dataclass
class VisualSeverityResult:
    visual_severity: str  # "low" | "moderate" | "high" | "undetermined"
    inspection_priority: str  # "monitor" | "review_recommended" | "immediate_action" | "further_investigation_required"
    method: str
    evidence: Dict[str, Any]
    rationale: List[str]
    limitations: List[str]
    disclaimer: str = STANDARD_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VisualSeverityEngine:
    """Pure deterministic, explainable visual severity assessment engine."""

    @classmethod
    def evaluate_building(
        cls,
        classified_categories: List[str],
        classified_probabilities: Dict[str, float],
        crack_coverage_pct: float,
        crack_count: int,
        associated_components: List[str],
        has_segmentation_masks: bool = True,
    ) -> VisualSeverityResult:
        """
        Evaluate visual severity for civil/concrete structural inspection.
        Requires corroborated multi-model evidence before assigning high visual severity.
        """
        rationale: List[str] = []
        limitations: List[str] = [
            "2D photograph lacks physical crack-width calibration reference (e.g. gauge or coin).",
            "Internal concrete reinforcement condition cannot be assessed without destructive testing or NDT.",
            "Load-bearing status and reserve structural capacity cannot be inferred from surface visual appearance.",
        ]

        cfg = HEURISTIC_THRESHOLDS["building"]
        norm_classes = [c.lower() for c in classified_categories]

        evidence = {
            "classified_categories": classified_categories,
            "crack_coverage_percentage": round(crack_coverage_pct, 4),
            "crack_instances_count": crack_count,
            "associated_components": sorted(list(set(associated_components))) if associated_components else ["unknown"],
            "has_segmentation_masks": has_segmentation_masks,
            "heuristic_thresholds_applied": cfg,
        }

        # Case 0: No defects detected across classification and segmentation
        if len(classified_categories) == 0 and crack_count == 0 and crack_coverage_pct <= 0.001:
            return VisualSeverityResult(
                visual_severity="low",
                inspection_priority="monitor",
                method="rule_based_visual_screening",
                evidence=evidence,
                rationale=[
                    "No concrete damage categories exceeded calibrated classification thresholds.",
                    "No concrete cracks detected by segmentation model.",
                    "Surface visual appearance is consistent with unblemished or non-damaged concrete.",
                ],
                limitations=limitations,
            )

        # High Severity Rule 1: Exposed Reinforcing Bars (Severe structural spall)
        has_exposed_bars = "exposedbars" in norm_classes
        if has_exposed_bars:
            prob = classified_probabilities.get("ExposedBars", classified_probabilities.get("exposedbars", 1.0))
            if prob >= 0.40:
                rationale.append(f"Visual classification indicates exposed reinforcing bars (probability: {prob:.2f}).")
                rationale.append("Loss of concrete cover exposes internal rebar to accelerated environmental degradation.")
                return VisualSeverityResult(
                    visual_severity="high",
                    inspection_priority="immediate_action",
                    method="rule_based_visual_screening",
                    evidence=evidence,
                    rationale=rationale,
                    limitations=limitations,
                )

        # High Severity Rule 2: Concurrent Spalling AND Corrosion Staining
        has_spalling = "spallation" in norm_classes
        has_corrosion_stain = "corrosionstain" in norm_classes
        if has_spalling and has_corrosion_stain:
            rationale.append("Concurrent concrete surface spallation and rust leachate staining observed.")
            rationale.append("Co-occurrence indicates active volumetric expansion from subsurface rebar oxidation.")
            return VisualSeverityResult(
                visual_severity="high",
                inspection_priority="immediate_action",
                method="rule_based_visual_screening",
                evidence=evidence,
                rationale=rationale,
                limitations=limitations,
            )

        # High Severity Rule 3: Heavy Relative Crack Coverage (> 1.5% total surface area)
        if crack_coverage_pct >= cfg["crack_coverage_high_pct"]:
            rationale.append(
                f"Relative crack surface coverage ({crack_coverage_pct:.2f}%) exceeds high heuristic threshold ({cfg['crack_coverage_high_pct']}%)."
            )
            rationale.append("Widespread surface fissure network observed across photographic field of view.")
            return VisualSeverityResult(
                visual_severity="high",
                inspection_priority="immediate_action",
                method="rule_based_visual_screening",
                evidence=evidence,
                rationale=rationale,
                limitations=limitations,
            )

        # High Severity Rule 4: Multi-Defect Concurrence (>= 3 distinct damage categories)
        if len(classified_categories) >= cfg["multi_defect_high_count"]:
            rationale.append(
                f"Multiple concurrent defect categories ({len(classified_categories)}) identified on same concrete element: {', '.join(classified_categories)}."
            )
            rationale.append("Compound degradation mechanisms present elevated maintenance risk.")
            return VisualSeverityResult(
                visual_severity="high",
                inspection_priority="immediate_action",
                method="rule_based_visual_screening",
                evidence=evidence,
                rationale=rationale,
                limitations=limitations,
            )

        # Moderate Severity Rule 1: Moderate Crack Coverage (0.2% - 1.5%)
        if crack_coverage_pct >= cfg["crack_coverage_moderate_pct"]:
            rationale.append(
                f"Relative crack surface coverage ({crack_coverage_pct:.2f}%) is within moderate screening range [{cfg['crack_coverage_moderate_pct']}%, {cfg['crack_coverage_high_pct']}%]."
            )
            rationale.append("Measurable structural crack network present; periodic width monitoring recommended.")
            return VisualSeverityResult(
                visual_severity="moderate",
                inspection_priority="review_recommended",
                method="rule_based_visual_screening",
                evidence=evidence,
                rationale=rationale,
                limitations=limitations,
            )

        # Moderate Severity Rule 2: Localized Spalling or Corrosion Stain Alone
        if has_spalling or has_corrosion_stain:
            if has_spalling:
                rationale.append("Localized surface spallation identified without visible reinforcing bar exposure.")
            if has_corrosion_stain:
                rationale.append("Surface corrosion stain staining identified; suggests subsurface moisture contact.")
            return VisualSeverityResult(
                visual_severity="moderate",
                inspection_priority="review_recommended",
                method="rule_based_visual_screening",
                evidence=evidence,
                rationale=rationale,
                limitations=limitations,
            )

        # Moderate Severity Rule 3: Crack located on Critical Structural Element (column/beam/slab)
        critical_comps = {"column", "beam", "slab"}
        found_critical = [c for c in evidence["associated_components"] if c.lower() in critical_comps]
        if crack_count > 0 and found_critical:
            rationale.append(f"Crack detected on primary structural component ({', '.join(found_critical)}).")
            rationale.append("Structural component context warrants human engineering review.")
            return VisualSeverityResult(
                visual_severity="moderate",
                inspection_priority="review_recommended",
                method="rule_based_visual_screening",
                evidence=evidence,
                rationale=rationale,
                limitations=limitations,
            )

        # Low Severity Rule 1: Hairline Cracking Only (< 0.20% coverage)
        if crack_count > 0:
            rationale.append(f"Superficial or hairline crack detected with low relative surface area ({crack_coverage_pct:.2f}%).")
            rationale.append("Damage appears localized and superficial at current photographic resolution.")
            return VisualSeverityResult(
                visual_severity="low",
                inspection_priority="monitor",
                method="rule_based_visual_screening",
                evidence=evidence,
                rationale=rationale,
                limitations=limitations,
            )

        # Low Severity Rule 2: Efflorescence Only (Mineral leaching)
        if "efflorescence" in norm_classes:
            rationale.append("Isolated calcite/mineral efflorescence staining detected without cracking or spallation.")
            rationale.append("Indicates water transport or past moisture exposure without direct mechanical section loss.")
            return VisualSeverityResult(
                visual_severity="low",
                inspection_priority="monitor",
                method="rule_based_visual_screening",
                evidence=evidence,
                rationale=rationale,
                limitations=limitations,
            )

        # Fallback / Ambiguous: Undetermined
        return VisualSeverityResult(
            visual_severity="undetermined",
            inspection_priority="further_investigation_required",
            method="rule_based_visual_screening",
            evidence=evidence,
            rationale=["Visual findings are borderline or insufficient to assign a deterministic screening category."],
            limitations=limitations,
        )

    @classmethod
    def evaluate_industrial(
        cls,
        detected_defects: List[Dict[str, Any]],
        relative_defect_area_pct: float,
        associated_components: List[str],
        has_ocr_text: bool = False,
    ) -> VisualSeverityResult:
        """
        Evaluate visual severity for industrial mechanical & equipment inspection.
        Requires confirmed defect evidence before assigning high visual severity.
        """
        rationale: List[str] = []
        limitations: List[str] = [
            "Internal wall thinning or remaining pipe schedule cannot be measured visually without ultrasonic thickness gauges.",
            "Operating pressure, internal temperature, fluid toxicity, and flow conditions are unknown from image alone.",
            "Mechanical vibration, dynamic balance, and electrical continuity cannot be inferred from a single photograph.",
        ]

        cfg = HEURISTIC_THRESHOLDS["industrial"]
        defect_categories = [d.get("category", "").lower() for d in detected_defects]

        evidence = {
            "detected_defects_count": len(detected_defects),
            "defect_categories": list(set(defect_categories)),
            "relative_defect_area_percentage": round(relative_defect_area_pct, 4),
            "associated_components": sorted(list(set(associated_components))) if associated_components else ["unknown"],
            "has_ocr_text": has_ocr_text,
            "heuristic_thresholds_applied": cfg,
        }

        # Case 0: No defects detected
        if len(detected_defects) == 0:
            return VisualSeverityResult(
                visual_severity="low",
                inspection_priority="monitor",
                method="rule_based_visual_screening",
                evidence=evidence,
                rationale=[
                    "No industrial defect bounding boxes detected across corrosion, metal crack, or slippage detectors.",
                    "Equipment surface appears intact within current image resolution.",
                ],
                limitations=limitations,
            )

        # High Severity Rule 1: Mechanical Slippage Detected (Bolt looseness / joint misalignment)
        if "mechanical_slippage" in defect_categories:
            rationale.append("Mechanical slippage indicator displacement or bolt back-off pattern detected.")
            rationale.append("Coupling/flange displacement represents potential loss of mechanical joint clamping force.")
            return VisualSeverityResult(
                visual_severity="high",
                inspection_priority="immediate_action",
                method="rule_based_visual_screening",
                evidence=evidence,
                rationale=rationale,
                limitations=limitations,
            )

        # High Severity Rule 2: Metal Crack Detected (Fatigue fissure / weld crack)
        if "metal_crack" in defect_categories:
            rationale.append("Structural metal crack or weld seam fissure visually detected.")
            rationale.append("Linear crack in metallic asset represents potential fracture propagation risk.")
            return VisualSeverityResult(
                visual_severity="high",
                inspection_priority="immediate_action",
                method="rule_based_visual_screening",
                evidence=evidence,
                rationale=rationale,
                limitations=limitations,
            )

        # High Severity Rule 3: Heavy Surface Corrosion Coverage (> 8.0%)
        if relative_defect_area_pct >= cfg["corrosion_area_high_pct"]:
            rationale.append(
                f"Surface corrosion extent ({relative_defect_area_pct:.2f}%) exceeds heavy screening threshold ({cfg['corrosion_area_high_pct']}%)."
            )
            rationale.append("Widespread surface oxidation or flaking rust scale observed across equipment asset.")
            return VisualSeverityResult(
                visual_severity="high",
                inspection_priority="immediate_action",
                method="rule_based_visual_screening",
                evidence=evidence,
                rationale=rationale,
                limitations=limitations,
            )

        # Moderate Severity Rule 1: Moderate Corrosion (1.0% - 8.0%)
        if relative_defect_area_pct >= cfg["corrosion_area_moderate_pct"]:
            rationale.append(
                f"Moderate surface corrosion ({relative_defect_area_pct:.2f}%) detected within screening range [{cfg['corrosion_area_moderate_pct']}%, {cfg['corrosion_area_high_pct']}%]."
            )
            rationale.append("Localized iron oxide rust observed; surface prep and protective recoating recommended.")
            return VisualSeverityResult(
                visual_severity="moderate",
                inspection_priority="review_recommended",
                method="rule_based_visual_screening",
                evidence=evidence,
                rationale=rationale,
                limitations=limitations,
            )

        # Low Severity Rule: Superficial Localized Corrosion (< 1.0%)
        if "corrosion" in defect_categories:
            rationale.append(f"Minor superficial corrosion spotting ({relative_defect_area_pct:.2f}%) detected.")
            rationale.append("Isolated oxidation with limited surface footprint; schedule for routine monitoring.")
            return VisualSeverityResult(
                visual_severity="low",
                inspection_priority="monitor",
                method="rule_based_visual_screening",
                evidence=evidence,
                rationale=rationale,
                limitations=limitations,
            )

        # Fallback / Undetermined
        return VisualSeverityResult(
            visual_severity="undetermined",
            inspection_priority="further_investigation_required",
            method="rule_based_visual_screening",
            evidence=evidence,
            rationale=["Defect findings do not clearly align with documented screening categories."],
            limitations=limitations,
        )
