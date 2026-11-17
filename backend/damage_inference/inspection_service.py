"""
Unified Multi-Model Inspection Service for YOLO Microservice.
Runs on CUDA GPU (Port 8001, Python 3.12 / PyTorch 2.6.0+cu124).
Orchestrates:
  1. YOLO26s-seg (Crack Segmentation)
  2. YOLO26s (Industrial Defect Detection: corrosion, metal_crack, mechanical_slippage)
  3. YOLOv8s-Worldv2 (Open-Vocabulary Building & Equipment Component Localization)
  4. ResNet18 (CODEBRIM Multi-Label Concrete Damage Classifier)
  5. EasyOCR (Industrial Nameplate & Tag Text Extraction)
"""
import io
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO, YOLOWorld

from damage_inference.config import (
    BACKEND_DIR,
    PROJECT_ROOT,
    CHECKPOINT_PATH as CRACK_MODEL_PATH,
    DEFAULT_IMGSZ,
    DEFAULT_CONF,
    DEFAULT_IOU,
)

# Ensure Ultralytics resolves weights/clip/ relative to project root regardless of cwd
try:
    import ultralytics.utils
    project_weights = PROJECT_ROOT / "weights"
    if project_weights.exists():
        ultralytics.utils.WEIGHTS_DIR = project_weights
except Exception:
    pass
from damage_inference.image_utils import (
    validate_and_decode_image,
    encode_image_to_base64_png,
)
from damage_inference.classifier_service import CodebrimClassifierService
from damage_inference.schemas import (
    MicroserviceInspectionResponse,
    ComponentItem,
    DefectItem,
    OCRItem,
    ClassificationSummary,
    ImagesPayload,
    InferenceMeta,
)

logger = logging.getLogger("damage_inference.inspection_service")
logging.basicConfig(level=logging.INFO)

# Model paths (resolves relative to BACKEND_DIR with fallback)
INDUSTRIAL_MODEL_PATH = (BACKEND_DIR / "models" / "damage" / "industrial_defect_yolo26s.pt").resolve()
if not INDUSTRIAL_MODEL_PATH.exists():
    INDUSTRIAL_MODEL_PATH = Path("backend/models/damage/industrial_defect_yolo26s.pt")

YOLO_WORLD_PATH = (BACKEND_DIR / "models" / "yolov8s-worldv2.pt").resolve()
if not YOLO_WORLD_PATH.exists():
    YOLO_WORLD_PATH = Path("backend/models/yolov8s-worldv2.pt")

CODEBRIM_MODEL_PATH = (BACKEND_DIR / "models" / "damage" / "codebrim_multilabel_best.pt").resolve()
if not CODEBRIM_MODEL_PATH.exists():
    CODEBRIM_MODEL_PATH = Path("backend/models/damage/codebrim_multilabel_best.pt")

BUILDING_PROMPTS = ["beam", "column", "wall", "slab", "ceiling", "floor", "window", "door"]
INDUSTRIAL_PROMPTS = ["pipe", "flange", "valve", "tank", "motor", "pump", "gauge", "panel"]


class InspectionService:
    """Thread-safe multi-model GPU inspection orchestrator."""

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.crack_model: Optional[YOLO] = None
        self.industrial_model: Optional[YOLO] = None
        self.world_model: Optional[YOLOWorld] = None
        self.classifier_service: Optional[CodebrimClassifierService] = None
        self.ocr_reader: Optional[Any] = None
        self.is_initialized: bool = False

    def initialize(self) -> bool:
        """Load and warm up vision models on GPU."""
        logger.info(f"Initializing InspectionService on device: {self.device}")
        try:
            # 1. Model 1: Crack Segmentation
            if CRACK_MODEL_PATH.exists():
                logger.info(f"Loading Crack Segmentation model: {CRACK_MODEL_PATH}")
                self.crack_model = YOLO(str(CRACK_MODEL_PATH))
                _ = self.crack_model(np.zeros((320, 320, 3), dtype=np.uint8), device=self.device, verbose=False)
            else:
                logger.warning(f"Crack model not found at {CRACK_MODEL_PATH}")

            # 2. Model 2: Industrial Defect Detection
            if INDUSTRIAL_MODEL_PATH.exists():
                logger.info(f"Loading Industrial Defect model: {INDUSTRIAL_MODEL_PATH}")
                self.industrial_model = YOLO(str(INDUSTRIAL_MODEL_PATH))
                _ = self.industrial_model(np.zeros((320, 320, 3), dtype=np.uint8), device=self.device, verbose=False)
            else:
                logger.warning(f"Industrial model not found at {INDUSTRIAL_MODEL_PATH}")

            # 3. Model 3: YOLO-World Open-Vocabulary Detector
            if YOLO_WORLD_PATH.exists():
                logger.info(f"Loading YOLO-World model: {YOLO_WORLD_PATH}")
                self.world_model = YOLOWorld(str(YOLO_WORLD_PATH))
            else:
                logger.warning(f"YOLO-World model not found at {YOLO_WORLD_PATH}")

            # 4. Model 4: CODEBRIM ResNet18 Classifier
            if CODEBRIM_MODEL_PATH.exists():
                logger.info(f"Loading CODEBRIM Classifier: {CODEBRIM_MODEL_PATH}")
                self.classifier_service = CodebrimClassifierService(
                    checkpoint_path=CODEBRIM_MODEL_PATH,
                    device=self.device,
                    strict_cuda=torch.cuda.is_available()
                )
                self.classifier_service.initialize()
            else:
                logger.warning(f"CODEBRIM model not found at {CODEBRIM_MODEL_PATH}")

            self.is_initialized = True
            logger.info("InspectionService core vision models successfully initialized.")
            return True
        except Exception as exc:
            logger.exception(f"InspectionService initialization error: {str(exc)}")
            return False

    def _get_ocr_reader(self):
        """Lazy load EasyOCR reader on GPU."""
        if self.ocr_reader is None:
            logger.info("Initializing EasyOCR reader on GPU...")
            import easyocr
            self.ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
        return self.ocr_reader

    @staticmethod
    def _calculate_box_overlap(box_a: List[int], box_b: List[int]) -> float:
        """Calculate intersection over area of box_a."""
        x_left = max(box_a[0], box_b[0])
        y_top = max(box_a[1], box_b[1])
        x_right = min(box_a[2], box_b[2])
        y_bottom = min(box_a[3], box_b[3])

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        return intersection / float(area_a) if area_a > 0 else 0.0

    async def run_building_inspection(
        self,
        image_bytes: bytes,
        confidence: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
    ) -> MicroserviceInspectionResponse:
        """Run complete building inspection pipeline."""
        t_start = time.perf_counter()
        img_bgr, orig_h, orig_w = validate_and_decode_image(image_bytes)

        t_pre = time.perf_counter()
        warnings = []
        defects: List[DefectItem] = []
        components: List[ComponentItem] = []

        # 1. Concrete Multi-Label Classification (ResNet18)
        classification_summary = None
        if self.classifier_service is not None:
            clf_res = self.classifier_service.predict(img_bgr)
            classification_summary = ClassificationSummary(
                classes=clf_res["predicted_classes"],
                probabilities=clf_res["probabilities"],
                thresholds=clf_res["thresholds"]
            )
            # Add image-level classification findings (without fake bounding boxes)
            for c_name in clf_res["predicted_classes"]:
                if c_name.lower() != "crack":  # Crack is segmented below with localized geometry
                    defects.append(DefectItem(
                        category=c_name,
                        source_model="ResNet18-CODEBRIM",
                        finding_type="classification",
                        confidence=clf_res["probabilities"][c_name],
                        bbox=None,
                        area_pixels=None,
                        polygon=None,
                        associated_component="unknown"
                    ))

        # 2. Structural Component Detection (YOLO-World)
        if self.world_model is not None:
            self.world_model.set_classes(BUILDING_PROMPTS)
            world_res = self.world_model(
                img_bgr,
                conf=0.20,
                device=self.device,
                imgsz=DEFAULT_IMGSZ,
                verbose=False
            )[0]
            if world_res.boxes is not None and len(world_res.boxes) > 0:
                boxes = world_res.boxes.xyxy.cpu().numpy().astype(int)
                confs = world_res.boxes.conf.cpu().numpy()
                classes = world_res.boxes.cls.cpu().numpy().astype(int)
                for b, cf, cl in zip(boxes, confs, classes):
                    comp_name = BUILDING_PROMPTS[cl] if cl < len(BUILDING_PROMPTS) else "component"
                    components.append(ComponentItem(
                        name=comp_name,
                        confidence=round(float(cf), 3),
                        bbox=[int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                    ))

        # 3. Concrete Crack Segmentation (YOLO26s-seg)
        crack_coverage_pct = 0.0
        binary_mask_full = np.zeros((orig_h, orig_w), dtype=np.uint8)

        if self.crack_model is not None:
            crack_res = self.crack_model(
                img_bgr,
                conf=confidence,
                iou=iou,
                device=self.device,
                imgsz=DEFAULT_IMGSZ,
                verbose=False
            )[0]

            if crack_res.boxes is not None and len(crack_res.boxes) > 0:
                boxes = crack_res.boxes.xyxy.cpu().numpy().astype(int)
                confs = crack_res.boxes.conf.cpu().numpy()

                # Process masks if available
                if crack_res.masks is not None:
                    raw_masks = crack_res.masks.data.cpu().numpy()
                    for idx, (b, cf) in enumerate(zip(boxes, confs)):
                        m = raw_masks[idx]
                        m_resized = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                        bin_m = (m_resized > 0.5).astype(np.uint8) * 255
                        binary_mask_full = np.bitwise_or(binary_mask_full, bin_m)

                        # Find associated component
                        det_box = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                        assoc_comp = "unknown"
                        best_ov = 0.0
                        for comp in components:
                            ov = self._calculate_box_overlap(det_box, comp.bbox)
                            if ov > 0.05 and ov > best_ov:
                                best_ov = ov
                                assoc_comp = comp.name

                        # Extract contour polygon
                        contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        poly_pts = []
                        if contours:
                            largest_c = max(contours, key=cv2.contourArea)
                            poly_pts = [[float(p[0][0]), float(p[0][1])] for p in largest_c if len(p) > 0]

                        area_px = int(np.sum(bin_m > 0))
                        defects.append(DefectItem(
                            category="crack",
                            source_model="YOLO26s-seg",
                            finding_type="segmentation",
                            confidence=round(float(cf), 3),
                            bbox=det_box,
                            area_pixels=area_px,
                            polygon=poly_pts if len(poly_pts) >= 3 else None,
                            associated_component=assoc_comp
                        ))

                    total_crack_pixels = np.count_nonzero(binary_mask_full)
                    total_pixels = orig_h * orig_w
                    crack_coverage_pct = round((total_crack_pixels / total_pixels) * 100.0, 4)

        # 4. Generate Visual Overlay
        overlay_bgr = img_bgr.copy()

        # Draw component bounding boxes (subtle blue-gray dashed/thin)
        for comp in components:
            x1, y1, x2, y2 = comp.bbox
            cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (210, 160, 80), 2)
            cv2.putText(
                overlay_bgr,
                f"[{comp.name.upper()}] {comp.confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (210, 160, 80),
                2,
                cv2.LINE_AA
            )

        # Draw crack segmentation mask in vibrant red with transparency
        if np.any(binary_mask_full):
            color_mask = np.zeros_like(img_bgr)
            color_mask[binary_mask_full > 0] = [0, 0, 235]  # Bright red
            overlay_bgr = cv2.addWeighted(overlay_bgr, 0.72, color_mask, 0.28, 0)

            # Draw crack bounding boxes and labels
            for d in defects:
                if d.finding_type == "segmentation" and d.bbox:
                    x1, y1, x2, y2 = d.bbox
                    cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (0, 0, 220), 2)
                    cv2.putText(
                        overlay_bgr,
                        f"Crack: {d.confidence:.2f} ({d.associated_component})",
                        (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )

        # Encode images
        t_post = time.perf_counter()
        orig_b64 = encode_image_to_base64_png(img_bgr)
        mask_b64 = encode_image_to_base64_png(binary_mask_full)
        overlay_b64 = encode_image_to_base64_png(overlay_bgr)

        t_end = time.perf_counter()

        return MicroserviceInspectionResponse(
            status="completed",
            domain="building",
            image_width=orig_w,
            image_height=orig_h,
            classification=classification_summary,
            defects=defects,
            components=components,
            ocr_items=[],
            crack_coverage_percentage=crack_coverage_pct,
            images=ImagesPayload(
                original=orig_b64,
                mask=mask_b64,
                overlay=overlay_b64
            ),
            inference=InferenceMeta(
                device=self.device,
                duration_ms=round((t_end - t_start) * 1000, 2),
                preprocess_ms=round((t_pre - t_start) * 1000, 2),
                inference_ms=round((t_post - t_pre) * 1000, 2),
                postprocess_ms=round((t_end - t_post) * 1000, 2)
            ),
            warnings=warnings
        )

    async def run_industrial_inspection(
        self,
        image_bytes: bytes,
        confidence: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
    ) -> MicroserviceInspectionResponse:
        """Run complete industrial equipment inspection pipeline."""
        t_start = time.perf_counter()
        img_bgr, orig_h, orig_w = validate_and_decode_image(image_bytes)

        t_pre = time.perf_counter()
        warnings = []
        defects: List[DefectItem] = []
        components: List[ComponentItem] = []
        ocr_items: List[OCRItem] = []

        # 1. Equipment Component Detection (YOLO-World)
        if self.world_model is not None:
            self.world_model.set_classes(INDUSTRIAL_PROMPTS)
            world_res = self.world_model(
                img_bgr,
                conf=0.20,
                device=self.device,
                imgsz=DEFAULT_IMGSZ,
                verbose=False
            )[0]
            if world_res.boxes is not None and len(world_res.boxes) > 0:
                boxes = world_res.boxes.xyxy.cpu().numpy().astype(int)
                confs = world_res.boxes.conf.cpu().numpy()
                classes = world_res.boxes.cls.cpu().numpy().astype(int)
                for b, cf, cl in zip(boxes, confs, classes):
                    comp_name = INDUSTRIAL_PROMPTS[cl] if cl < len(INDUSTRIAL_PROMPTS) else "equipment"
                    components.append(ComponentItem(
                        name=comp_name,
                        confidence=round(float(cf), 3),
                        bbox=[int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                    ))

        # 2. Industrial Defect Detection (YOLO26s)
        total_defect_area_px = 0
        if self.industrial_model is not None:
            ind_res = self.industrial_model(
                img_bgr,
                conf=confidence,
                iou=iou,
                device=self.device,
                imgsz=DEFAULT_IMGSZ,
                verbose=False
            )[0]
            if ind_res.boxes is not None and len(ind_res.boxes) > 0:
                boxes = ind_res.boxes.xyxy.cpu().numpy().astype(int)
                confs = ind_res.boxes.conf.cpu().numpy()
                classes = ind_res.boxes.cls.cpu().numpy().astype(int)
                names = self.industrial_model.names

                for b, cf, cl in zip(boxes, confs, classes):
                    cat_name = names.get(cl, "defect")
                    det_box = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                    box_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                    total_defect_area_px += box_area

                    # Associate with closest/overlapping equipment component
                    assoc_comp = "unknown"
                    best_ov = 0.0
                    for comp in components:
                        ov = self._calculate_box_overlap(det_box, comp.bbox)
                        if ov > 0.05 and ov > best_ov:
                            best_ov = ov
                            assoc_comp = comp.name

                    defects.append(DefectItem(
                        category=cat_name,
                        source_model="YOLO26s-Industrial",
                        finding_type="bounding_box",
                        confidence=round(float(cf), 3),
                        bbox=det_box,
                        area_pixels=box_area,
                        polygon=None,
                        associated_component=assoc_comp
                    ))

        # 3. Optical Character Recognition (EasyOCR)
        try:
            reader = self._get_ocr_reader()
            ocr_results = reader.readtext(img_bgr)
            for ocr_entry in ocr_results:
                poly, text, ocr_conf = ocr_entry
                if len(text.strip()) > 1 and ocr_conf > 0.25:
                    pts = np.array(poly, dtype=int)
                    x1 = int(np.min(pts[:, 0]))
                    y1 = int(np.min(pts[:, 1]))
                    x2 = int(np.max(pts[:, 0]))
                    y2 = int(np.max(pts[:, 1]))
                    ocr_items.append(OCRItem(
                        text=text.strip(),
                        confidence=round(float(ocr_conf), 3),
                        bbox=[x1, y1, x2, y2]
                    ))
        except Exception as ocr_err:
            logger.warning(f"EasyOCR extraction exception: {str(ocr_err)}")
            warnings.append(f"OCR engine warning: {str(ocr_err)}")

        # 4. Generate Visual Overlay
        overlay_bgr = img_bgr.copy()

        # Component boxes (cyan)
        for comp in components:
            x1, y1, x2, y2 = comp.bbox
            cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(
                overlay_bgr,
                f"[{comp.name.upper()}] {comp.confidence:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 0),
                2,
                cv2.LINE_AA
            )

        # Defect boxes (color-coded)
        color_map = {
            "corrosion": (0, 140, 255),          # Amber / Orange
            "metal_crack": (0, 0, 255),           # Red
            "mechanical_slippage": (255, 0, 255)  # Magenta
        }
        for d in defects:
            if d.bbox:
                x1, y1, x2, y2 = d.bbox
                c_color = color_map.get(d.category.lower(), (0, 0, 255))
                cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), c_color, 2)
                cv2.putText(
                    overlay_bgr,
                    f"{d.category}: {d.confidence:.2f} ({d.associated_component})",
                    (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    c_color,
                    2,
                    cv2.LINE_AA
                )

        # OCR text boxes (green dashed/thin)
        for o in ocr_items:
            x1, y1, x2, y2 = o.bbox
            cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (0, 220, 100), 1)
            cv2.putText(
                overlay_bgr,
                f"TXT: {o.text}",
                (x1, min(orig_h - 5, y2 + 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 100),
                1,
                cv2.LINE_AA
            )

        # Mask representation for industrial (bounding box mask union)
        ind_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for d in defects:
            if d.bbox:
                x1, y1, x2, y2 = d.bbox
                ind_mask[y1:y2, x1:x2] = 255

        rel_defect_coverage = round((total_defect_area_px / (orig_h * orig_w)) * 100.0, 4)

        t_post = time.perf_counter()
        orig_b64 = encode_image_to_base64_png(img_bgr)
        mask_b64 = encode_image_to_base64_png(ind_mask)
        overlay_b64 = encode_image_to_base64_png(overlay_bgr)
        t_end = time.perf_counter()

        return MicroserviceInspectionResponse(
            status="completed",
            domain="industrial",
            image_width=orig_w,
            image_height=orig_h,
            classification=None,
            defects=defects,
            components=components,
            ocr_items=ocr_items,
            crack_coverage_percentage=rel_defect_coverage,
            images=ImagesPayload(
                original=orig_b64,
                mask=mask_b64,
                overlay=overlay_b64
            ),
            inference=InferenceMeta(
                device=self.device,
                duration_ms=round((t_end - t_start) * 1000, 2),
                preprocess_ms=round((t_pre - t_start) * 1000, 2),
                inference_ms=round((t_post - t_pre) * 1000, 2),
                postprocess_ms=round((t_end - t_post) * 1000, 2)
            ),
            warnings=warnings
        )


inspection_service = InspectionService()
