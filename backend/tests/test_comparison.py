
import cv2
import numpy as np

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def make_image(width=200, height=150):
    image = np.full(
        (height, width, 3),
        200,
        dtype=np.uint8
    )

    success, encoded = cv2.imencode(".png", image)
    assert success

    return encoded.tobytes()


def test_valid_images():
    image = make_image()

    response = client.post(
        "/api/comparison/validate",
        files={
            "before": ("before.png", image, "image/png"),
            "after": ("after.png", image, "image/png"),
        }
    )

    assert response.status_code == 200
    assert response.json()["status"] == "validated"
    assert response.json()["same_dimensions"] is True


def test_different_dimensions():
    response = client.post(
        "/api/comparison/validate",
        files={
            "before": (
                "before.png", make_image(200, 150), "image/png"
            ),
            "after": (
                "after.png", make_image(300, 200), "image/png"
            ),
        }
    )

    assert response.status_code == 200
    assert response.json()["same_dimensions"] is False


def test_invalid_image():
    response = client.post(
        "/api/comparison/validate",
        files={
            "before": (
                "before.png", b"not an image", "image/png"
            ),
            "after": (
                "after.png", make_image(), "image/png"
            ),
        }
    )

    assert response.status_code == 422


def test_unsupported_format():
    response = client.post(
        "/api/comparison/validate",
        files={
            "before": (
                "before.txt", b"hello", "text/plain"
            ),
            "after": (
                "after.png", make_image(), "image/png"
            ),
        }
    )

    assert response.status_code == 422
