"""
Asynchronous HTTP client for the Multi-Model GPU Inspection Microservice.
Connects FastAPI Gateway (port 8000) to the YOLO microservice (port 8001).
"""
import os
import logging
from typing import Dict, Any, Optional
import httpx

from app.services.damage_client import (
    DamageServiceError,
    DamageServiceUnavailableError,
    DamageServiceTimeoutError,
)

logger = logging.getLogger("inspection_client")

DEFAULT_MICROSERVICE_URL = os.getenv("DAMAGE_INFERENCE_URL", "http://127.0.0.1:8001")
CONNECT_TIMEOUT = float(os.getenv("DAMAGE_CONNECT_TIMEOUT", "4.0"))
READ_TIMEOUT = float(os.getenv("DAMAGE_READ_TIMEOUT", "45.0"))


class InspectionClient:
    """Asynchronous client for multi-model building and industrial inspection."""

    def __init__(self, base_url: str = DEFAULT_MICROSERVICE_URL):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout=READ_TIMEOUT, connect=CONNECT_TIMEOUT)

    def _get_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def check_health(self) -> Dict[str, Any]:
        """Check microservice liveness and model status."""
        try:
            async with self._get_client() as client:
                res = await client.get("/inspect/status")
                res.raise_for_status()
                return res.json()
        except httpx.ConnectError:
            raise DamageServiceUnavailableError("YOLO GPU inspection microservice is not reachable.")
        except httpx.TimeoutException:
            raise DamageServiceTimeoutError("Inspection microservice health check timed out.")
        except Exception as exc:
            return {"status": "degraded", "error": str(exc)}

    async def inspect_building(
        self,
        image_bytes: bytes,
        confidence: float = 0.25,
        iou: float = 0.45,
    ) -> Dict[str, Any]:
        """Send image to microservice for complete building concrete inspection."""
        files = {"image": ("inspection_building.jpg", image_bytes, "image/jpeg")}
        data = {"confidence": str(confidence), "iou": str(iou)}

        try:
            async with self._get_client() as client:
                response = await client.post("/inspect/building", files=files, data=data)
                if response.status_code == 503:
                    raise DamageServiceUnavailableError("GPU models are still loading or unavailable.")
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError:
            raise DamageServiceUnavailableError("YOLO GPU inspection microservice is currently unreachable on port 8001.")
        except httpx.TimeoutException:
            raise DamageServiceTimeoutError("Building inspection request timed out on GPU microservice.")
        except httpx.HTTPStatusError as hse:
            raise DamageServiceError(
                code="INSPECTION_FAILED",
                message=f"Microservice inspection error: {hse.response.text}",
                status_code=hse.response.status_code,
            )

    async def inspect_industrial(
        self,
        image_bytes: bytes,
        confidence: float = 0.25,
        iou: float = 0.45,
    ) -> Dict[str, Any]:
        """Send image to microservice for complete industrial equipment inspection."""
        files = {"image": ("inspection_industrial.jpg", image_bytes, "image/jpeg")}
        data = {"confidence": str(confidence), "iou": str(iou)}

        try:
            async with self._get_client() as client:
                response = await client.post("/inspect/industrial", files=files, data=data)
                if response.status_code == 503:
                    raise DamageServiceUnavailableError("GPU models are still loading or unavailable.")
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError:
            raise DamageServiceUnavailableError("YOLO GPU inspection microservice is currently unreachable on port 8001.")
        except httpx.TimeoutException:
            raise DamageServiceTimeoutError("Industrial inspection request timed out on GPU microservice.")
        except httpx.HTTPStatusError as hse:
            raise DamageServiceError(
                code="INSPECTION_FAILED",
                message=f"Microservice industrial inspection error: {hse.response.text}",
                status_code=hse.response.status_code,
            )


inspection_client = InspectionClient()
