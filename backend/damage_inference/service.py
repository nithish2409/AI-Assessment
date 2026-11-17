"""
Core YOLO Crack Segmentation Service.
Loads model checkpoint, manages GPU resources, executes inference, and processes masks.
"""
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from damage_inference.config import (
    CHECKPOINT_PATH,
    METADATA_PATH,
    DEFAULT_IMGSZ,
    DEFAULT_CONF,
    DEFAULT_IOU,
    MAX_CONCURRENT_INFERENCES,
)
from damage_inference.image_utils import (
    validate_and_decode_image,
    encode_image_to_base64_png,
    create_overlay,
)
from damage_inference.schemas import (
    DamageInferenceResponse,
    DetectionItem,
    ImagesPayload,
    InferenceMeta,
    ReadinessResponse,
)

logger = logging.getLogger("damage_inference.service")
logging.basicConfig(level=logging.INFO)


class DamageInferenceService:
    """Thread-safe and concurrency-controlled YOLO Crack Segmentation inference service."""

    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH):
        self.checkpoint_path = checkpoint_path
        self.model: Optional[YOLO] = None
        self.device: str = "cpu"
        self.is_loaded: bool = False
        self.init_error: Optional[str] = None
        self.task: str = "segment"
        self.class_names: Dict[int, str] = {0: "crack"}
        self._concurrency_semaphore: Optional[asyncio.Semaphore] = None

    def initialize(self) -> bool:
        """Load checkpoint once per process and prime GPU if available."""
        logger.info(f"Initializing YOLO segmentation model from: {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            self.init_error = f"Checkpoint not found at: {self.checkpoint_path}"
            logger.error(self.init_error)
            self.is_loaded = False
            return False

        try:
            # Determine execution device
            if torch.cuda.is_available():
                self.device = "cuda:0"
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA is available. Selected device: {self.device} ({gpu_name})")
            else:
                self.device = "cpu"
                logger.warning("CUDA is not available. Falling back to CPU execution.")

            # Load YOLO model
            self.model = YOLO(str(self.checkpoint_path))

            # Verify segmentation task
            if hasattr(self.model, "task") and self.model.task != "segment":
                logger.warning(f"Model task reported as '{self.model.task}', expected 'segment'.")

            # Verify class names
            if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                self.class_names = self.model.names
            logger.info(f"Model classes: {self.class_names}")

            # Warmup inference to initialize CUDA runtime and avoid latency spike
            logger.info("Executing warmup forward pass...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(
                dummy_img,
                device=self.device,
                imgsz=DEFAULT_IMGSZ,
                conf=DEFAULT_CONF,
                verbose=False,
            )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

            self.is_loaded = True
            self.init_error = None
            logger.info("YOLO Crack Segmentation service successfully initialized.")
            return True

        except Exception as exc:
            self.init_error = f"Failed to initialize model: {str(exc)}"
            logger.exception(self.init_error)
            self.is_loaded = False
            self.model = None
            return False

    def get_semaphore(self) -> asyncio.Semaphore:
        """Lazy initialization of asyncio concurrency semaphore in the running event loop."""
        if self._concurrency_semaphore is None:
            self._concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
        return self._concurrency_semaphore

    def get_readiness(self) -> ReadinessResponse:
        """Return operational readiness and device status."""
        return ReadinessResponse(
            ready=self.is_loaded,
            model="YOLO26s-seg",
            checkpoint=self.checkpoint_path.name,
            task=self.task,
            classes={str(k): v for k, v in self.class_names.items()},
            device=self.device,
            cuda_available=torch.cuda.is_available(),
            message=(
                "Service ready for crack segmentation"
                if self.is_loaded
                else f"Model not loaded: {self.init_error}"
            ),
        )

    async def segment_image(
        self,
        image_data: bytes,
        content_type: Optional[str] = None,
        confidence: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
    ) -> DamageInferenceResponse:
        """
        Execute full segmentation pipeline on image bytes under concurrency protection.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(
                f"Inference service is not ready. {self.init_error or 'Model not loaded.'}"
            )

        sem = self.get_semaphore()
        async with sem:
            return await asyncio.to_thread(
                self._sync_segment_image,
                image_data,
                content_type,
                confidence,
                iou,
            )

    def _sync_segment_image(
        self,
        image_data: bytes,
        content_type: Optional[str],
        confidence: float,
        iou: float,
    ) -> DamageInferenceResponse:
        total_t0 = time.perf_counter()

        # 1. Preprocessing & decoding
        t_pre0 = time.perf_counter()
        image_bgr, orig_h, orig_w = validate_and_decode_image(image_data, content_type)
        t_pre = (time.perf_counter() - t_pre0) * 1000.0

        # 2. PyTorch model inference
        t_inf0 = time.perf_counter()
        try:
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

            results = self.model(
                image_bgr,
                device=self.device,
                imgsz=DEFAULT_IMGSZ,
                conf=confidence,
                iou=iou,
                verbose=False,
            )[0]

            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            t_inf = (time.perf_counter() - t_inf0) * 1000.0

        except torch.cuda.OutOfMemoryError as oom:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.error("CUDA Out of Memory during inference.")
            raise RuntimeError(
                "GPU out of memory during crack segmentation inference."
            ) from oom

        # 3. Postprocessing & Mask extraction
        t_post0 = time.perf_counter()
        detections: List[DetectionItem] = []
        binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        has_boxes = (
            hasattr(results, "boxes")
            and results.boxes is not None
            and len(results.boxes) > 0
        )
        has_masks = (
            hasattr(results, "masks")
            and results.masks is not None
            and len(results.masks) > 0
        )

        if has_boxes and has_masks:
            # results.masks.xy contains list of numpy arrays with polygon coordinates in original image space
            for i, (box, poly) in enumerate(zip(results.boxes, results.masks.xy)):
                cls_id = int(box.cls[0].item())
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                conf = round(float(box.conf[0].item()), 4)

                # Bounding box in [x1, y1, x2, y2]
                xyxy = box.xyxy[0].tolist()
                bbox = [
                    max(0, min(orig_w - 1, int(round(xyxy[0])))),
                    max(0, min(orig_h - 1, int(round(xyxy[1])))),
                    max(0, min(orig_w, int(round(xyxy[2])))),
                    max(0, min(orig_h, int(round(xyxy[3])))),
                ]

                # Rasterize individual polygon for area calculation
                indiv_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                poly_coords = None
                if len(poly) >= 3:
                    clamped_poly = poly.copy()
                    clamped_poly[:, 0] = np.clip(clamped_poly[:, 0], 0, orig_w)
                    clamped_poly[:, 1] = np.clip(clamped_poly[:, 1], 0, orig_h)
                    pts = np.round(clamped_poly).astype(np.int32)
                    cv2.fillPoly(binary_mask, [pts], 255)
                    cv2.fillPoly(indiv_mask, [pts], 255)
                    area_px = int(np.count_nonzero(indiv_mask > 0))
                    # Store compact polygon coordinates
                    poly_coords = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in clamped_poly]
                else:
                    area_px = 0

                detections.append(
                    DetectionItem(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=bbox,
                        area_pixels=area_px,
                        polygon=poly_coords,
                    )
                )

        # 4. Coverage calculation: Union of all crack pixels without double counting
        total_pixels = orig_h * orig_w
        unique_crack_pixels = int(np.count_nonzero(binary_mask > 0))
        crack_coverage_pct = round((unique_crack_pixels / total_pixels) * 100.0, 4)

        # 5. Visual output generation
        overlay_bgr = create_overlay(image_bgr, binary_mask)

        b64_original = encode_image_to_base64_png(image_bgr)
        b64_mask = encode_image_to_base64_png(binary_mask)
        b64_overlay = encode_image_to_base64_png(overlay_bgr)

        t_post = (time.perf_counter() - t_post0) * 1000.0
        total_duration = (time.perf_counter() - total_t0) * 1000.0

        warnings = []
        if self.device == "cpu":
            warnings.append("Inference was executed on CPU; GPU acceleration is unavailable.")
        if len(detections) == 0:
            warnings.append("No cracks detected above the confidence threshold.")

        return DamageInferenceResponse(
            status="completed",
            model="YOLO26s-seg",
            class_names=[self.class_names.get(0, "crack")],
            image_width=orig_w,
            image_height=orig_h,
            detections_count=len(detections),
            crack_coverage_percentage=crack_coverage_pct,
            detections=detections,
            images=ImagesPayload(
                original=b64_original,
                mask=b64_mask,
                overlay=b64_overlay,
            ),
            inference=InferenceMeta(
                device=self.device,
                duration_ms=round(total_duration, 2),
                preprocess_ms=round(t_pre, 2),
                inference_ms=round(t_inf, 2),
                postprocess_ms=round(t_post, 2),
            ),
            warnings=warnings,
        )


# Global service instance
inference_service = DamageInferenceService()
