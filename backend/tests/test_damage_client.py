"""
Unit tests for the asynchronous DamageClient.
"""
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.damage_client import (
    DamageClient,
    DamageServiceUnavailableError,
    DamageServiceTimeoutError,
    DamageServiceError,
)


@pytest.mark.anyio
async def test_client_health_success():
    client = DamageClient(base_url="http://127.0.0.1:8001")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "healthy", "service": "YOLO Crack Inference Microservice"}

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        res = await client.check_health()
        assert res["status"] == "healthy"


@pytest.mark.anyio
async def test_client_connect_error_raises_unavailable():
    client = DamageClient(base_url="http://127.0.0.1:8001")
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(DamageServiceUnavailableError) as exc_info:
            await client.check_health()
        assert exc_info.value.code == "DAMAGE_SERVICE_UNAVAILABLE"
        assert exc_info.value.status_code == 503


@pytest.mark.anyio
async def test_client_timeout_raises_timeout_error():
    client = DamageClient(base_url="http://127.0.0.1:8001")
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.ReadTimeout("Timeout")):
        with pytest.raises(DamageServiceTimeoutError) as exc_info:
            await client.check_health()
        assert exc_info.value.code == "INFERENCE_TIMEOUT"
        assert exc_info.value.status_code == 504


@pytest.mark.anyio
async def test_client_segment_image_success():
    client = DamageClient(base_url="http://127.0.0.1:8001")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "completed",
        "model": "YOLO26s-seg",
        "class_names": ["crack"],
        "image_width": 640,
        "image_height": 640,
        "detections_count": 1,
        "crack_coverage_percentage": 2.5,
        "detections": [],
        "images": {"original": "abc", "mask": "def", "overlay": "ghi"},
        "inference": {"device": "cuda:0", "duration_ms": 15.0},
        "warnings": [],
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        res = await client.segment_image(b"fake_bytes", "test.jpg", "image/jpeg", 0.25, 0.45)
        assert res["status"] == "completed"
        assert res["detections_count"] == 1
        assert res["crack_coverage_percentage"] == 2.5


@pytest.mark.anyio
async def test_client_segment_image_validation_error():
    client = DamageClient(base_url="http://127.0.0.1:8001")
    mock_resp = MagicMock()
    mock_resp.status_code = 422
    mock_resp.json.return_value = {"detail": "Invalid image dimensions"}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(DamageServiceError) as exc_info:
            await client.segment_image(b"fake_bytes", "test.jpg", "image/jpeg")
        assert exc_info.value.code == "INVALID_INFERENCE_REQUEST"
        assert exc_info.value.status_code == 422


@pytest.mark.anyio
async def test_client_segment_image_service_unavailable():
    client = DamageClient(base_url="http://127.0.0.1:8001")
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.ConnectError("Refused")):
        with pytest.raises(DamageServiceUnavailableError) as exc_info:
            await client.segment_image(b"fake_bytes")
        assert exc_info.value.status_code == 503
