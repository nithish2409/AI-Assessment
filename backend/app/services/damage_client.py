"""
Asynchronous HTTP client for communicating with the isolated YOLO Damage Inference Microservice.
"""
import os
import logging
from typing import Dict, Any, Optional
import httpx

logger = logging.getLogger("damage_client")

DEFAULT_MICROSERVICE_URL = os.getenv("DAMAGE_INFERENCE_URL", "http://127.0.0.1:8001")
CONNECT_TIMEOUT = float(os.getenv("DAMAGE_CONNECT_TIMEOUT", "3.0"))
READ_TIMEOUT = float(os.getenv("DAMAGE_READ_TIMEOUT", "30.0"))


class DamageServiceError(Exception):
    """Base exception for damage client interactions."""
    def __init__(self, code: str, message: str, status_code: int = 500, details: Optional[Dict] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class DamageServiceUnavailableError(DamageServiceError):
    """Raised when the microservice is unreachable or not ready."""
    def __init__(self, message: str = "The damage segmentation inference microservice is currently unreachable."):
        super().__init__(
            code="DAMAGE_SERVICE_UNAVAILABLE",
            message=message,
            status_code=503,
            details={
                "guidance": (
                    "Ensure the YOLO inference microservice is running under "
                    "backend/venv_yolo on http://127.0.0.1:8001."
                )
            },
        )


class DamageServiceTimeoutError(DamageServiceError):
    """Raised when the microservice request times out."""
    def __init__(self, message: str = "The damage segmentation inference request timed out."):
        super().__init__(
            code="INFERENCE_TIMEOUT",
            message=message,
            status_code=504,
            details={
                "guidance": (
                    "The GPU or inference process took longer than expected to process the image. "
                    "Verify GPU memory and system load."
                )
            },
        )


class DamageClient:
    """Asynchronous HTTP client for the YOLO26s-seg microservice."""

    def __init__(self, base_url: str = DEFAULT_MICROSERVICE_URL):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(
            timeout=READ_TIMEOUT,
            connect=CONNECT_TIMEOUT,
        )

    def _get_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def check_health(self) -> Dict[str, Any]:
        """Check whether the microservice process is alive."""
        try:
            async with self._get_client() as client:
                response = await client.get("/health")
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError:
            raise DamageServiceUnavailableError()
        except httpx.TimeoutException:
            raise DamageServiceTimeoutError()
        except Exception as exc:
            raise DamageServiceError(
                code="HEALTH_CHECK_FAILED",
                message=f"Microservice health check failed: {str(exc)}",
                status_code=502,
            )

    async def check_readiness(self) -> Dict[str, Any]:
        """Check whether the YOLO model is loaded and ready."""
        try:
            async with self._get_client() as client:
                response = await client.get("/ready")
                if response.status_code == 503:
                    raise DamageServiceUnavailableError("Model is not ready for inference.")
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError:
            raise DamageServiceUnavailableError()
        except httpx.TimeoutException:
            raise DamageServiceTimeoutError()

    async def segment_image(
        self,
        image_bytes: bytes,
        filename: str = "upload.jpg",
        content_type: str = "image/jpeg",
        confidence: float = 0.25,
        iou: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Forward image bytes and parameters to microservice /infer endpoint.
        """
        files = {
            "image": (filename, image_bytes, content_type)
        }
        data = {
            "confidence": str(confidence),
            "iou": str(iou),
        }

        try:
            async with self._get_client() as client:
                response = await client.post("/infer", files=files, data=data)

                if response.status_code == 422:
                    error_detail = response.json().get("detail", "Validation error in inference service.")
                    raise DamageServiceError(
                        code="INVALID_INFERENCE_REQUEST",
                        message=str(error_detail),
                        status_code=422,
                    )

                if response.status_code == 503:
                    raise DamageServiceUnavailableError(
                        "Inference service reported that the model is unavailable."
                    )

                response.raise_for_status()
                return response.json()

        except httpx.ConnectError as ce:
            logger.warning(f"Connection to damage microservice failed: {ce}")
            raise DamageServiceUnavailableError()
        except httpx.TimeoutException as te:
            logger.warning(f"Damage microservice request timed out: {te}")
            raise DamageServiceTimeoutError()
        except DamageServiceError:
            raise
        except Exception as exc:
            logger.error(f"Unexpected error communicating with damage microservice: {exc}")
            raise DamageServiceError(
                code="INFERENCE_FAILED",
                message="An error occurred while processing the damage segmentation request.",
                status_code=500,
            )


# Global client instance
damage_client = DamageClient()
