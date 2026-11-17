"""
Unit tests for the Deterministic Visual Severity Assessment Engine.
Verifies all rule triggers, priorities, rationales, limitations, and edge cases.
"""
import pytest
from app.services.visual_severity_engine import (
    VisualSeverityEngine,
    VisualSeverityResult,
    HEURISTIC_THRESHOLDS,
)


class TestBuildingSeverityAssessment:
    """Test suite for Building/Civil Structural Visual Severity assessment."""

    def test_clean_concrete_low_severity(self):
        """Clean surface with no defects should evaluate to low visual severity and monitor priority."""
        res = VisualSeverityEngine.evaluate_building(
            classified_categories=[],
            classified_probabilities={},
            crack_coverage_pct=0.0,
            crack_count=0,
            associated_components=["wall"],
        )
        assert res.visual_severity == "low"
        assert res.inspection_priority == "monitor"
        assert len(res.rationale) > 0
        assert "clean" in res.rationale[-1].lower() or "unblemished" in res.rationale[-1].lower()

    def test_exposed_rebar_high_severity(self):
        """Visible exposed reinforcement must trigger high visual severity and immediate action."""
        res = VisualSeverityEngine.evaluate_building(
            classified_categories=["ExposedBars", "Spallation"],
            classified_probabilities={"ExposedBars": 0.85, "Spallation": 0.90},
            crack_coverage_pct=0.10,
            crack_count=1,
            associated_components=["beam"],
        )
        assert res.visual_severity == "high"
        assert res.inspection_priority == "immediate_action"
        assert any("exposed reinforcing bars" in r.lower() for r in res.rationale)

    def test_concurrent_spalling_and_rust_high_severity(self):
        """Co-occurrence of spalling and rust staining must trigger high visual severity."""
        res = VisualSeverityEngine.evaluate_building(
            classified_categories=["Spallation", "CorrosionStain"],
            classified_probabilities={"Spallation": 0.75, "CorrosionStain": 0.80},
            crack_coverage_pct=0.05,
            crack_count=0,
            associated_components=["column"],
        )
        assert res.visual_severity == "high"
        assert res.inspection_priority == "immediate_action"
        assert any("concurrent" in r.lower() for r in res.rationale)

    def test_extensive_crack_coverage_high_severity(self):
        """Crack coverage >= 1.5% must trigger high visual severity regardless of other categories."""
        res = VisualSeverityEngine.evaluate_building(
            classified_categories=["Crack"],
            classified_probabilities={"Crack": 0.92},
            crack_coverage_pct=2.15,
            crack_count=5,
            associated_components=["slab"],
        )
        assert res.visual_severity == "high"
        assert res.inspection_priority == "immediate_action"
        assert any("exceeds high heuristic threshold" in r for r in res.rationale)

    def test_moderate_crack_coverage_moderate_severity(self):
        """Crack coverage between 0.20% and 1.50% must trigger moderate visual severity."""
        res = VisualSeverityEngine.evaluate_building(
            classified_categories=["Crack"],
            classified_probabilities={"Crack": 0.78},
            crack_coverage_pct=0.65,
            crack_count=2,
            associated_components=["wall"],
        )
        assert res.visual_severity == "moderate"
        assert res.inspection_priority == "review_recommended"
        assert any("moderate" in r.lower() for r in res.rationale)

    def test_isolated_spalling_alone_moderate_severity(self):
        """Isolated spalling without exposed bars triggers moderate severity."""
        res = VisualSeverityEngine.evaluate_building(
            classified_categories=["Spallation"],
            classified_probabilities={"Spallation": 0.72},
            crack_coverage_pct=0.0,
            crack_count=0,
            associated_components=["wall"],
        )
        assert res.visual_severity == "moderate"
        assert res.inspection_priority == "review_recommended"

    def test_hairline_crack_low_severity(self):
        """Hairline crack with < 0.20% coverage triggers low visual severity."""
        res = VisualSeverityEngine.evaluate_building(
            classified_categories=["Crack"],
            classified_probabilities={"Crack": 0.65},
            crack_coverage_pct=0.08,
            crack_count=1,
            associated_components=["wall"],
        )
        assert res.visual_severity == "low"
        assert res.inspection_priority == "monitor"

    def test_isolated_efflorescence_low_severity(self):
        """Isolated efflorescence leaching triggers low visual severity."""
        res = VisualSeverityEngine.evaluate_building(
            classified_categories=["Efflorescence"],
            classified_probabilities={"Efflorescence": 0.81},
            crack_coverage_pct=0.0,
            crack_count=0,
            associated_components=["wall"],
        )
        assert res.visual_severity == "low"
        assert res.inspection_priority == "monitor"
        assert any("efflorescence" in r.lower() for r in res.rationale)

    def test_engineering_limitations_and_disclaimer_present(self):
        """All evaluation results must contain explicit engineering limitations and disclaimer."""
        res = VisualSeverityEngine.evaluate_building(
            classified_categories=["Crack"],
            classified_probabilities={"Crack": 0.75},
            crack_coverage_pct=0.5,
            crack_count=1,
            associated_components=["beam"],
        )
        assert len(res.limitations) >= 3
        assert "calibration reference" in res.limitations[0]
        assert "preliminary heuristic" in res.disclaimer.lower()


class TestIndustrialSeverityAssessment:
    """Test suite for Industrial Mechanical Visual Severity assessment."""

    def test_clean_industrial_low_severity(self):
        """No detected defects triggers low visual severity."""
        res = VisualSeverityEngine.evaluate_industrial(
            detected_defects=[],
            relative_defect_area_pct=0.0,
            associated_components=["pipe"],
        )
        assert res.visual_severity == "low"
        assert res.inspection_priority == "monitor"

    def test_mechanical_slippage_high_severity(self):
        """Mechanical flange slippage triggers high visual severity and immediate action."""
        res = VisualSeverityEngine.evaluate_industrial(
            detected_defects=[{"category": "mechanical_slippage", "confidence": 0.95}],
            relative_defect_area_pct=0.45,
            associated_components=["flange"],
        )
        assert res.visual_severity == "high"
        assert res.inspection_priority == "immediate_action"
        assert any("slippage" in r.lower() for r in res.rationale)

    def test_metal_crack_high_severity(self):
        """Metal weld or fatigue crack triggers high visual severity and immediate action."""
        res = VisualSeverityEngine.evaluate_industrial(
            detected_defects=[{"category": "metal_crack", "confidence": 0.82}],
            relative_defect_area_pct=0.15,
            associated_components=["pipe"],
        )
        assert res.visual_severity == "high"
        assert res.inspection_priority == "immediate_action"
        assert any("metal crack" in r.lower() for r in res.rationale)

    def test_heavy_corrosion_high_severity(self):
        """Extensive surface corrosion (> 8.0%) triggers high visual severity."""
        res = VisualSeverityEngine.evaluate_industrial(
            detected_defects=[{"category": "corrosion", "confidence": 0.90}],
            relative_defect_area_pct=9.5,
            associated_components=["tank"],
        )
        assert res.visual_severity == "high"
        assert res.inspection_priority == "immediate_action"

    def test_moderate_corrosion_moderate_severity(self):
        """Corrosion between 1.0% and 8.0% triggers moderate visual severity."""
        res = VisualSeverityEngine.evaluate_industrial(
            detected_defects=[{"category": "corrosion", "confidence": 0.85}],
            relative_defect_area_pct=3.2,
            associated_components=["pipe"],
        )
        assert res.visual_severity == "moderate"
        assert res.inspection_priority == "review_recommended"

    def test_minor_corrosion_low_severity(self):
        """Localized corrosion < 1.0% triggers low visual severity."""
        res = VisualSeverityEngine.evaluate_industrial(
            detected_defects=[{"category": "corrosion", "confidence": 0.70}],
            relative_defect_area_pct=0.4,
            associated_components=["panel"],
        )
        assert res.visual_severity == "low"
        assert res.inspection_priority == "monitor"
