"""
Integration tests for the main FastAPI Damage API router (POST /api/damage/segment).
"""
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

client = TestClient(app)


def create_test_image_bytes(width=100, height=100, fmt="JPEG", channels=3):
    img = np.zeros((height, width, channels), dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 2), min(width, height) // 4, (255, 255, 255), -1)
    ext = ".jpg" if fmt == "JPEG" else (f".{fmt.lower()}")
    success, encoded = cv2.imencode(ext, img)
    assert success
    return encoded.tobytes()


@pytest.fixture
def mock_segment_response():
    return {
        "status": "completed",
        "model": "YOLO26s-seg",
        "class_names": ["crack"],
        "image_width": 640,
        "image_height": 640,
        "detections_count": 1,
        "crack_coverage_percentage": 1.42,
        "detections": [
            {
                "class_id": 0,
                "class_name": "crack",
                "confidence": 0.87,
                "bbox": [100, 150, 420, 310],
                "area_pixels": 5800,
                "polygon": [
                    [100.0, 150.0],
                    [150.0, 160.0],
                    [210.0, 185.0],
                    [320.0, 240.0],
                    [420.0, 310.0],
                    [410.0, 315.0],
                    [305.0, 245.0],
                    [145.0, 165.0],
                ],
            }
        ],
        "images": {
            "original": "BASE64_ORIGINAL",
            "mask": "BASE64_MASK",
            "overlay": "BASE64_OVERLAY",
        },
        "inference": {
            "device": "cuda:0",
            "duration_ms": 25.5,
            "preprocess_ms": 5.0,
            "inference_ms": 13.5,
            "postprocess_ms": 7.0,
        },
        "warnings": [],
    }


def test_segment_valid_jpeg(mock_segment_response):
    img_bytes = create_test_image_bytes(fmt="JPEG")
    with patch("app.api.damage.damage_client.segment_image", new_callable=AsyncMock, return_value=mock_segment_response):
        response = client.post(
            "/api/damage/segment",
            files={"image": ("crack.jpg", img_bytes, "image/jpeg")},
            data={"confidence": "0.25", "iou": "0.45"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["model"] == "YOLO26s-seg"
        assert data["detections_count"] == 1
        assert data["crack_coverage_percentage"] == 1.42
        assert "original" in data["images"]
        assert "mask" in data["images"]
        assert "overlay" in data["images"]


def test_segment_valid_png(mock_segment_response):
    img_bytes = create_test_image_bytes(fmt="PNG")
    with patch("app.api.damage.damage_client.segment_image", new_callable=AsyncMock, return_value=mock_segment_response):
        response = client.post(
            "/api/damage/segment",
            files={"image": ("crack.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"


def test_segment_valid_webp(mock_segment_response):
    img_bytes = create_test_image_bytes(fmt="WEBP")
    with patch("app.api.damage.damage_client.segment_image", new_callable=AsyncMock, return_value=mock_segment_response):
        response = client.post(
            "/api/damage/segment",
            files={"image": ("crack.webp", img_bytes, "image/webp")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"


def test_segment_missing_upload():
    response = client.post("/api/damage/segment", data={"confidence": "0.25"})
    assert response.status_code == 422


def test_segment_empty_upload():
    response = client.post(
        "/api/damage/segment",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 422
    assert "empty" in response.json()["detail"]["message"].lower()


def test_segment_corrupted_image():
    response = client.post(
        "/api/damage/segment",
        files={"image": ("corrupted.jpg", b"NOT_A_VALID_JPEG_HEADER_XYZ", "image/jpeg")},
    )
    assert response.status_code == 422
    assert "corrupted" in response.json()["detail"]["message"].lower()


def test_segment_unsupported_format():
    response = client.post(
        "/api/damage/segment",
        files={"image": ("document.txt", b"Hello world", "text/plain")},
    )
    assert response.status_code == 422
    assert "supported" in response.json()["detail"]["message"].lower()


def test_segment_exceeds_max_file_size():
    # 10 MB + 100 bytes
    large_payload = b"0" * (10 * 1024 * 1024 + 100)
    response = client.post(
        "/api/damage/segment",
        files={"image": ("large.jpg", large_payload, "image/jpeg")},
    )
    assert response.status_code == 422
    assert "10 mb" in response.json()["detail"]["message"].lower()


def test_segment_invalid_confidence():
    img_bytes = create_test_image_bytes()
    # Confidence <= 0
    res1 = client.post(
        "/api/damage/segment",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        data={"confidence": "0.0"},
    )
    assert res1.status_code == 422
    assert "confidence" in str(res1.json()).lower()

    # Confidence > 1
    res2 = client.post(
        "/api/damage/segment",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        data={"confidence": "1.5"},
    )
    assert res2.status_code == 422


def test_segment_invalid_iou():
    img_bytes = create_test_image_bytes()
    # IoU <= 0
    res1 = client.post(
        "/api/damage/segment",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        data={"iou": "-0.1"},
    )
    assert res1.status_code == 422
    assert "iou" in str(res1.json()).lower()

    # IoU > 1
    res2 = client.post(
        "/api/damage/segment",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        data={"iou": "2.0"},
    )
    assert res2.status_code == 422


def test_segment_service_unavailable():
    img_bytes = create_test_image_bytes()
    with patch(
        "app.api.damage.damage_client.segment_image",
        new_callable=AsyncMock,
        side_effect=DamageServiceUnavailableError(),
    ):
        response = client.post(
            "/api/damage/segment",
            files={"image": ("crack.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["error"] == "DAMAGE_SERVICE_UNAVAILABLE"
        assert "unreachable" in data["detail"]["message"].lower()
        assert "guidance" in data["detail"]


def test_segment_service_timeout():
    img_bytes = create_test_image_bytes()
    with patch(
        "app.api.damage.damage_client.segment_image",
        new_callable=AsyncMock,
        side_effect=DamageServiceTimeoutError(),
    ):
        response = client.post(
            "/api/damage/segment",
            files={"image": ("crack.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 504
        data = response.json()
        assert data["detail"]["error"] == "INFERENCE_TIMEOUT"


def test_damage_health_endpoint():
    # Microservice healthy
    with patch("app.api.damage.damage_client.check_readiness", new_callable=AsyncMock, return_value={"ready": True}):
        res = client.get("/api/damage/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"
        assert res.json()["is_available"] is True

    # Microservice unavailable
    with patch(
        "app.api.damage.damage_client.check_readiness",
        new_callable=AsyncMock,
        side_effect=DamageServiceUnavailableError(),
    ):
        res2 = client.get("/api/damage/health")
        assert res2.status_code == 200
        assert res2.json()["status"] == "unavailable"
        assert res2.json()["is_available"] is False
