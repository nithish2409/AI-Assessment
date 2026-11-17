
import cv2
import numpy as np
import pytest

from fastapi.testclient import TestClient
from app.main import app

from app.services.image_comparison_engine import (
    image_comparison_engine
)

client = TestClient(app)


def make_test_pair():
    rng = np.random.default_rng(42)

    # Textured reference image with stable features.
    before = rng.integers(
        100, 220,
        size=(400, 600, 3),
        dtype=np.uint8
    )

    # Smooth the texture to resemble a surface.
    before = cv2.GaussianBlur(
        before, (3, 3), 0
    )

    after = before.copy()

    # Simulate a newly appearing dark crack.
    cv2.line(
        after,
        (150, 120),
        (430, 290),
        (10, 10, 10),
        thickness=8
    )

    return before, after


def encode(image):
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


def test_identical_images():
    before, _ = make_test_pair()

    result = image_comparison_engine.compare(
        before, before.copy()
    )

    assert result["status"] == "completed"
    assert result["alignment"]["status"] == "identical"
    assert result["ssim_score"] == pytest.approx(
        1.0, abs=0.0001
    )
    assert result["changed_percentage"] == 0


def test_new_visual_change():
    before, after = make_test_pair()

    result = image_comparison_engine.compare(
        before, after
    )

    assert result["status"] == "completed"
    assert result["changed_percentage"] > 0
    assert result["changed_regions_count"] >= 1


def test_comparison_api():
    before, after = make_test_pair()

    response = client.post(
        "/api/comparison/compare",
        files={
            "before": (
                "before.png",
                encode(before),
                "image/png"
            ),
            "after": (
                "after.png",
                encode(after),
                "image/png"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "ssim_score" in data
    assert "changed_percentage" in data
    assert "regions" in data
    assert "images" in data

    assert len(data["images"]["highlighted"]) > 0


def test_invalid_upload():
    before, _ = make_test_pair()

    response = client.post(
        "/api/comparison/compare",
        files={
            "before": (
                "before.png",
                encode(before),
                "image/png"
            ),
            "after": (
                "after.png",
                b"invalid data",
                "image/png"
            )
        }
    )

    assert response.status_code == 422
