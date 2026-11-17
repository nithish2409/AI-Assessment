"""
Unit tests for the FastAPI Inspection API endpoints (/api/inspection/analyze, /report/pdf, /health).
"""
import io
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def create_test_image_bytes(format="JPEG", size=(100, 100), color=(128, 128, 128)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color)
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


class TestInspectionInputValidation:
    """Test suite for request parameter and upload validation."""

    def test_invalid_domain_rejected(self):
        """Domain other than 'building' or 'industrial' must be rejected with 422."""
        img_bytes = create_test_image_bytes()
        res = client.post(
            "/api/inspection/analyze",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"domain": "marine_subsea"},
        )
        assert res.status_code == 422
        assert "domain" in res.text.lower()

    def test_unsupported_mime_type_rejected(self):
        """Unsupported MIME types like text/plain or image/gif must be rejected."""
        res = client.post(
            "/api/inspection/analyze",
            files={"image": ("test.txt", b"not an image", "text/plain")},
            data={"domain": "building"},
        )
        assert res.status_code == 422

    def test_empty_image_rejected(self):
        """Zero-byte image upload must be rejected."""
        res = client.post(
            "/api/inspection/analyze",
            files={"image": ("empty.jpg", b"", "image/jpeg")},
            data={"domain": "building"},
        )
        assert res.status_code == 422

    def test_invalid_confidence_bounds(self):
        """Confidence <= 0 or > 1 must be rejected with 422."""
        img_bytes = create_test_image_bytes()
        res = client.post(
            "/api/inspection/analyze",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"domain": "building", "confidence": "0.0"},
        )
        assert res.status_code == 422


class TestInspectionPipelineResponses:
    """Test response structure and severity synthesis with mocked microservice."""

    @patch("app.services.inspection_client.inspection_client.inspect_building", new_callable=AsyncMock)
    def test_building_inspection_success(self, mock_inspect):
        """Successful building inspection returns structured findings and visual severity."""
        mock_inspect.return_value = {
            "status": "completed",
            "domain": "building",
            "image_width": 640,
            "image_height": 480,
            "classification": {
                "classes": ["Crack", "Spallation"],
                "probabilities": {"Crack": 0.85, "Spallation": 0.72},
                "thresholds": {"Crack": 0.475, "Spallation": 0.375}
            },
            "defects": [
                {
                    "category": "crack",
                    "source_model": "YOLO26s-seg",
                    "finding_type": "segmentation",
                    "confidence": 0.88,
                    "bbox": [50, 50, 200, 200],
                    "area_pixels": 1500,
                    "polygon": [[50.0, 50.0], [200.0, 200.0]],
                    "associated_component": "wall"
                },
                {
                    "category": "Spallation",
                    "source_model": "ResNet18-CODEBRIM",
                    "finding_type": "classification",
                    "confidence": 0.72,
                    "bbox": None,
                    "area_pixels": None,
                    "polygon": None,
                    "associated_component": "unknown"
                }
            ],
            "components": [
                {"name": "wall", "confidence": 0.91, "bbox": [0, 0, 640, 480]}
            ],
            "ocr_items": [],
            "crack_coverage_percentage": 0.488,
            "images": {
                "original": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "mask": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "overlay": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            },
            "warnings": []
        }

        img_bytes = create_test_image_bytes()
        res = client.post(
            "/api/inspection/analyze",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"domain": "building"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["domain"] == "building"
        assert data["defects_count"] == 2
        assert data["visual_severity"]["visual_severity"] in ("moderate", "high")
        assert "immediate_action" in data["visual_severity"]["inspection_priority"] or "review_recommended" in data["visual_severity"]["inspection_priority"]
        assert "disclaimer" in data

    @patch("app.services.inspection_client.inspection_client.inspect_industrial", new_callable=AsyncMock)
    def test_industrial_inspection_success(self, mock_inspect):
        """Successful industrial inspection returns defects, components, OCR, and severity."""
        mock_inspect.return_value = {
            "status": "completed",
            "domain": "industrial",
            "image_width": 640,
            "image_height": 480,
            "classification": None,
            "defects": [
                {
                    "category": "corrosion",
                    "source_model": "YOLO26s-Industrial",
                    "finding_type": "bounding_box",
                    "confidence": 0.94,
                    "bbox": [100, 100, 300, 300],
                    "area_pixels": 40000,
                    "polygon": None,
                    "associated_component": "pipe"
                }
            ],
            "components": [
                {"name": "pipe", "confidence": 0.88, "bbox": [50, 50, 600, 450]}
            ],
            "ocr_items": [
                {"text": "LINE-4B 150PSI", "confidence": 0.91, "bbox": [120, 80, 250, 110]}
            ],
            "crack_coverage_percentage": 13.02,
            "images": {
                "original": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "mask": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "overlay": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            },
            "warnings": []
        }

        img_bytes = create_test_image_bytes()
        res = client.post(
            "/api/inspection/analyze",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"domain": "industrial"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["domain"] == "industrial"
        assert len(data["ocr_results"]) == 1
        assert data["ocr_results"][0]["text"] == "LINE-4B 150PSI"
        assert data["visual_severity"]["visual_severity"] == "high"


class TestInspectionReportPdfEndpoint:
    """Test suite for PDF report compilation and streaming download."""

    def test_pdf_report_generation(self):
        """Valid inspection response payload generates downloadable PDF."""
        payload = {
            "inspection_id": "INSP-TEST-001",
            "timestamp": "2026-11-17T21:00:00Z",
            "domain": "building",
            "status": "completed",
            "image_width": 640,
            "image_height": 480,
            "defects_count": 1,
            "defects": [
                {
                    "defect_id": "DEF-001",
                    "category": "crack",
                    "source_model": "YOLO26s-seg",
                    "finding_type": "segmentation",
                    "confidence": 0.89,
                    "bbox": [50, 50, 150, 150],
                    "area_pixels": 800,
                    "polygon": None,
                    "associated_component": "beam"
                }
            ],
            "components": [
                {"name": "beam", "confidence": 0.92, "bbox": [0, 0, 640, 200]}
            ],
            "ocr_results": [],
            "classification_summary": {"classes": ["Crack"]},
            "crack_coverage_percentage": 0.26,
            "visual_severity": {
                "visual_severity": "moderate",
                "inspection_priority": "review_recommended",
                "method": "rule_based_visual_screening",
                "evidence": {"crack_coverage_percentage": 0.26},
                "rationale": ["Moderate crack coverage observed."],
                "limitations": ["No physical calibration reference."],
                "disclaimer": "Automated preliminary screening."
            },
            "images": {
                "original": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "mask": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "overlay": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            },
            "processing_time_ms": 125.4,
            "warnings": [],
            "disclaimer": "Automated preliminary screening."
        }

        res = client.post("/api/inspection/report/pdf", json=payload)
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/pdf"
        assert res.content.startswith(b"%PDF-")
        assert len(res.content) > 1000
