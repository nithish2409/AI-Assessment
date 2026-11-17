"""
CodebrimClassifierService: Reusable GPU-accelerated Multi-Label Concrete Damage Classifier.
Loads fine-tuned ResNet18 checkpoint (codebrim_multilabel_best.pt), executes CUDA inference,
and applies calibrated per-class thresholds for concrete defect detection.
"""
import io
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

logger = logging.getLogger("damage_inference.classifier_service")
logging.basicConfig(level=logging.INFO)

PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parent
DEFAULT_CHECKPOINT_PATH = (BACKEND_DIR / "models" / "damage" / "codebrim_multilabel_best.pt").resolve()
if not DEFAULT_CHECKPOINT_PATH.exists():
    DEFAULT_CHECKPOINT_PATH = Path("backend/models/damage/codebrim_multilabel_best.pt")

DEFAULT_CLASSES = [
    "Crack",
    "Spallation",
    "CorrosionStain",
    "ExposedBars",
    "Efflorescence"
]
DEFAULT_THRESHOLDS = [0.50, 0.50, 0.50, 0.50, 0.50]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


class CodebrimClassifierService:
    """Thread-safe, CUDA-enforced ResNet18 multi-label classifier service."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT_PATH,
        device: str = "cuda:0",
        strict_cuda: bool = True
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device_str = device
        self.strict_cuda = strict_cuda
        self.device: Optional[torch.device] = None
        self.model: Optional[nn.Module] = None
        self.is_loaded: bool = False
        self.classes: List[str] = DEFAULT_CLASSES
        self.optimal_thresholds: List[float] = DEFAULT_THRESHOLDS
        self.thresholds_by_class: Dict[str, float] = {c: t for c, t in zip(DEFAULT_CLASSES, DEFAULT_THRESHOLDS)}
        self.metadata: Dict[str, Any] = {}
        self.transform: Optional[transforms.Compose] = None

    def initialize(self) -> bool:
        """Load checkpoint, verify weights, and move to CUDA GPU."""
        logger.info(f"Initializing CodebrimClassifierService with checkpoint: {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        # CUDA verification
        if self.strict_cuda:
            assert torch.cuda.is_available(), "CRITICAL: CUDA GPU unavailable. Strict CUDA execution policy violated."
            self.device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")

        # Load checkpoint payload
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.metadata = {
            "model_name": checkpoint.get("model_name"),
            "architecture": checkpoint.get("architecture", "ResNet18"),
            "best_epoch": checkpoint.get("best_epoch"),
            "validation_metrics": checkpoint.get("validation_metrics", {}),
            "hardware": checkpoint.get("hardware", {}),
        }

        if "classes" in checkpoint:
            self.classes = checkpoint["classes"]
        if "optimal_thresholds" in checkpoint:
            self.optimal_thresholds = checkpoint["optimal_thresholds"]
            self.thresholds_by_class = {c: t for c, t in zip(self.classes, self.optimal_thresholds)}

        # Build ResNet18 model
        model = resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(self.classes))

        # Load state dict
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        # Strict CUDA verification
        if self.strict_cuda or "cuda" in str(self.device):
            assert next(model.parameters()).is_cuda, "Model parameters are not on CUDA!"
            assert all(p.is_cuda for p in model.parameters()), "Not all parameters are on CUDA!"

        self.model = model
        self.is_loaded = True

        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        logger.info(f"CodebrimClassifierService initialized successfully on {self.device}. Classes: {self.classes}")
        return True

    def _prepare_image(self, image_input: Union[np.ndarray, Image.Image, Path, str, bytes]) -> Image.Image:
        """Standardize image input to RGB PIL Image."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return Image.open(path).convert("RGB")
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                # Grayscale to RGB
                return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB))
            elif image_input.ndim == 3:
                if image_input.shape[2] == 4:
                    return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB))
                elif image_input.shape[2] == 3:
                    # Assume BGR if from cv2, convert to RGB
                    return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            raise ValueError(f"Unsupported numpy image shape: {image_input.shape}")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    @torch.inference_mode()
    def predict(
        self,
        image_input: Union[np.ndarray, Image.Image, Path, str, bytes],
        custom_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Run single image classification and return calibrated defect predictions."""
        if not self.is_loaded:
            self.initialize()

        start_time = time.perf_counter()
        pil_img = self._prepare_image(image_input)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Forward pass on GPU
        logits = self.model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Resolve thresholds
        thresholds_dict = {}
        for c in self.classes:
            if custom_thresholds and c in custom_thresholds:
                thresholds_dict[c] = float(custom_thresholds[c])
            else:
                thresholds_dict[c] = float(self.thresholds_by_class.get(c, 0.5))

        # Format output
        probabilities = {}
        binary_predictions = {}
        predicted_classes = []

        for i, c_name in enumerate(self.classes):
            prob = float(probs[i])
            thresh = thresholds_dict[c_name]
            is_detected = prob >= thresh
            probabilities[c_name] = round(prob, 4)
            binary_predictions[c_name] = bool(is_detected)
            if is_detected:
                predicted_classes.append(c_name)

        return {
            "predicted_classes": predicted_classes,
            "has_defects": len(predicted_classes) > 0,
            "defect_count": len(predicted_classes),
            "probabilities": probabilities,
            "thresholds": thresholds_dict,
            "binary_predictions": binary_predictions,
            "device": str(self.device),
            "is_cuda": next(self.model.parameters()).is_cuda,
            "inference_time_ms": inference_time_ms
        }

    @torch.inference_mode()
    def predict_batch(
        self,
        image_inputs: List[Union[np.ndarray, Image.Image, Path, str, bytes]],
        custom_thresholds: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Run batch inference on multiple images."""
        if not self.is_loaded:
            self.initialize()

        if not image_inputs:
            return []

        start_time = time.perf_counter()
        tensors = []
        for inp in image_inputs:
            pil_img = self._prepare_image(inp)
            tensors.append(self.transform(pil_img))

        batch_tensor = torch.stack(tensors).to(self.device)
        logits = self.model(batch_tensor)
        probs_batch = torch.sigmoid(logits).cpu().numpy()

        total_batch_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        per_item_time_ms = round(total_batch_time_ms / len(image_inputs), 2)

        results = []
        for idx in range(len(image_inputs)):
            probs = probs_batch[idx]
            thresholds_dict = {}
            for c in self.classes:
                if custom_thresholds and c in custom_thresholds:
                    thresholds_dict[c] = float(custom_thresholds[c])
                else:
                    thresholds_dict[c] = float(self.thresholds_by_class.get(c, 0.5))

            probabilities = {}
            binary_predictions = {}
            predicted_classes = []

            for i, c_name in enumerate(self.classes):
                prob = float(probs[i])
                thresh = thresholds_dict[c_name]
                is_detected = prob >= thresh
                probabilities[c_name] = round(prob, 4)
                binary_predictions[c_name] = bool(is_detected)
                if is_detected:
                    predicted_classes.append(c_name)

            results.append({
                "predicted_classes": predicted_classes,
                "has_defects": len(predicted_classes) > 0,
                "defect_count": len(predicted_classes),
                "probabilities": probabilities,
                "thresholds": thresholds_dict,
                "binary_predictions": binary_predictions,
                "device": str(self.device),
                "is_cuda": next(self.model.parameters()).is_cuda,
                "inference_time_ms": per_item_time_ms
            })

        return results
