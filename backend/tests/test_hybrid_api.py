"""
Integration tests for the Hybrid Comparison FastAPI router:
  POST /api/comparison/hybrid
"""
import base64
import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from app.main import app
from app.services.damage_client import (
    DamageServiceUnavailableError,
    DamageServiceTimeoutError,
)
from app.services.crack_change_engine import crack_change_engine

client = TestClient(app)


def make_test_image(width: int = 350, height: int = 350, seed: int = 42) -> bytes:
    """Generate textured image bytes rich in keypoints."""
    rng = np.random.default_rng(seed)
    base = np.full((height, width, 3), 150, dtype=np.uint8)
    noise = rng.integers(-40, 40, (height, width, 3), dtype=np.int16)
    textured = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    for i in range(15, width - 20, 40):
        for j in range(15, height - 20, 40):
            cv2.circle(textured, (i, j), 5, (20, 20, 20), -1)
            cv2.rectangle(textured, (i + 10, j + 10), (i + 25, j + 25), (220, 220, 220), 2)

    _, buf = cv2.imencode(".jpg", textured)
    return buf.tobytes()


def make_mock_yolo_resp(h: int = 350, w: int = 350):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[100:150, 100:104] = 255
    b64_mask = crack_change_engine.encode_image_png(mask)
    return {
        "status": "completed",
        "model": "YOLO26s-seg",
        "class_names": ["crack"],
        "image_width": w,
        "image_height": h,
        "detections_count": 1,
        "crack_coverage_percentage": round(100.0 * 200 / (h * w), 4),
        "detections": [],
        "images": {
            "original": b64_mask,
            "mask": b64_mask,
            "overlay": b64_mask,
        },
        "inference": {
            "device": "mock_cuda:0",
            "duration_ms": 12.5,
        },
        "warnings": [],
    }


class TestHybridAPIValidation:
    """Validate parameter ranges and payload constraints."""

    def test_missing_files_returns_422(self):
        resp = client.post("/api/comparison/hybrid", data={"alignment_mode": "auto"})
        assert resp.status_code == 422

    def test_invalid_image_type_rejected(self):
        img_bytes = make_test_image()
        files = {
            "before": ("before.txt", b"plain text is not an image", "text/plain"),
            "after": ("after.jpg", img_bytes, "image/jpeg"),
        }
        resp = client.post("/api/comparison/hybrid", files=files)
        assert resp.status_code == 422
        assert "INVALID_IMAGE_TYPE" in resp.text

    def test_empty_image_rejected(self):
        img_bytes = make_test_image()
        files = {
            "before": ("before.jpg", b"", "image/jpeg"),
            "after": ("after.jpg", img_bytes, "image/jpeg"),
        }
        resp = client.post("/api/comparison/hybrid", files=files)
        assert resp.status_code == 422
        assert "EMPTY_IMAGE" in resp.text

    def test_invalid_confidence_threshold_rejected(self):
        img_bytes = make_test_image()
        files = {
            "before": ("before.jpg", img_bytes, "image/jpeg"),
            "after": ("after.jpg", img_bytes, "image/jpeg"),
        }
        resp = client.post("/api/comparison/hybrid", files=files, data={"confidence": "1.5"})
        assert resp.status_code == 422
        assert "INVALID_CONFIDENCE" in resp.text

    def test_invalid_iou_threshold_rejected(self):
        img_bytes = make_test_image()
        files = {
            "before": ("before.jpg", img_bytes, "image/jpeg"),
            "after": ("after.jpg", img_bytes, "image/jpeg"),
        }
        resp = client.post("/api/comparison/hybrid", files=files, data={"iou": "-0.2"})
        assert resp.status_code == 422
        assert "INVALID_IOU" in resp.text

    def test_invalid_tolerance_px_rejected(self):
        img_bytes = make_test_image()
        files = {
            "before": ("before.jpg", img_bytes, "image/jpeg"),
            "after": ("after.jpg", img_bytes, "image/jpeg"),
        }
        resp = client.post("/api/comparison/hybrid", files=files, data={"tolerance_px": "15"})
        assert resp.status_code == 422
        assert "INVALID_TOLERANCE" in resp.text

    def test_invalid_alignment_mode_rejected(self):
        img_bytes = make_test_image()
        files = {
            "before": ("before.jpg", img_bytes, "image/jpeg"),
            "after": ("after.jpg", img_bytes, "image/jpeg"),
        }
        resp = client.post("/api/comparison/hybrid", files=files, data={"alignment_mode": "magic"})
        assert resp.status_code == 422
        assert "INVALID_ALIGNMENT_MODE" in resp.text


class TestHybridAPIExecution:
    """Validate successful execution, error mapping, and response schema."""

    @patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock)
    def test_successful_hybrid_comparison(self, mock_seg):
        mock_seg.return_value = make_mock_yolo_resp(350, 350)
        img_bytes = make_test_image(350, 350)

        files = {
            "before": ("before.jpg", img_bytes, "image/jpeg"),
            "after": ("after.jpg", img_bytes, "image/jpeg"),
        }
        data = {
            "alignment_mode": "auto",
            "confidence": "0.25",
            "iou": "0.45",
            "tolerance_px": "2",
        }

        resp = client.post("/api/comparison/hybrid", files=files, data=data)
        assert resp.status_code == 200
        payload = resp.json()

        assert payload["status"] == "completed"
        assert payload["image_width"] == 350
        assert payload["image_height"] == 350

        # Check alignment sub-dict
        align = payload["alignment"]
        assert align["status"] in ("accepted", "identical")
        assert align["valid_overlap_percentage"] > 95.0

        # Check visual comparison sub-dict
        vis = payload["visual_comparison"]
        assert "ssim_score" in vis
        assert "visual_change_percentage" in vis

        # Check crack comparison sub-dict
        crack = payload["crack_comparison"]
        assert "before_crack_pixels" in crack
        assert "after_crack_pixels" in crack
        assert "persistent_pixels" in crack
        assert "newly_observed_pixels" in crack
        assert "tolerance_adjusted_new_pixels" in crack
        assert "potential_new_components" in crack
        assert "potential_expanded_components" in crack

        # Check images payload
        images = payload["images"]
        assert "aligned_before" in images
        assert "visual_difference" in images
        assert "before_crack_mask" in images
        assert "after_crack_mask" in images
        assert "hybrid_overlay" in images
        assert len(images["hybrid_overlay"]) > 100

        # Check warnings
        assert len(payload["warnings"]) >= 1

    @patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock)
    def test_microservice_unavailable_maps_to_503(self, mock_seg):
        mock_seg.side_effect = DamageServiceUnavailableError("YOLO inference service down.")
        img_bytes = make_test_image(350, 350)

        files = {
            "before": ("before.jpg", img_bytes, "image/jpeg"),
            "after": ("after.jpg", img_bytes, "image/jpeg"),
        }
        resp = client.post("/api/comparison/hybrid", files=files)
        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert detail["error"] == "DAMAGE_SERVICE_UNAVAILABLE"

    @patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock)
    def test_microservice_timeout_maps_to_504(self, mock_seg):
        mock_seg.side_effect = DamageServiceTimeoutError("Inference timed out.")
        img_bytes = make_test_image(350, 350)

        files = {
            "before": ("before.jpg", img_bytes, "image/jpeg"),
            "after": ("after.jpg", img_bytes, "image/jpeg"),
        }
        resp = client.post("/api/comparison/hybrid", files=files)
        assert resp.status_code == 504
        detail = resp.json()["detail"]
        assert detail["error"] == "INFERENCE_TIMEOUT"

    def test_unrelated_images_structured_alignment_failure(self):
        """Unrelated images must return HTTP 422 with ALIGNMENT_FAILED and failure reasons."""
        img1 = make_test_image(300, 300, seed=1)
        # Random non-matching noise image
        img2 = np.zeros((300, 300, 3), dtype=np.uint8)
        for i in range(0, 300, 15):
            cv2.line(img2, (0, i), (300, 300 - i), (255, 255, 255), 2)
        _, buf2 = cv2.imencode(".jpg", img2)
        img2_bytes = buf2.tobytes()

        files = {
            "before": ("before.jpg", img1, "image/jpeg"),
            "after": ("after.jpg", img2_bytes, "image/jpeg"),
        }
        resp = client.post("/api/comparison/hybrid", files=files, data={"alignment_mode": "auto"})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["error"] == "ALIGNMENT_FAILED"
        assert "guidance" in detail
        assert "attempted_methods" in detail

    def test_existing_endpoints_remain_functional(self):
        """Ensure /api/comparison/validate and /api/health remain completely intact."""
        h_resp = client.get("/api/health")
        assert h_resp.status_code == 200

        img_bytes = make_test_image(200, 200)
        v_resp = client.post(
            "/api/comparison/validate",
            files={
                "before": ("before.jpg", img_bytes, "image/jpeg"),
                "after": ("after.jpg", img_bytes, "image/jpeg"),
            },
        )
        assert v_resp.status_code == 200
        assert v_resp.json()["status"] == "validated"
