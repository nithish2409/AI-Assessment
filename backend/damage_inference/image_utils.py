"""
Image preprocessing and visualization utilities for YOLO damage inference.
"""
import base64
import io
import cv2
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Optional
from damage_inference.config import (
    MAX_UPLOAD_SIZE,
    MAX_PIXELS,
    ALLOWED_CONTENT_TYPES,
)


def validate_and_decode_image(
    data: bytes, content_type: Optional[str] = None
) -> Tuple[np.ndarray, int, int]:
    """
    Validate raw upload bytes, correct EXIF orientation, and decode into a 3-channel BGR numpy array.

    Returns:
        (image_bgr, height, width)

    Raises:
        ValueError: If file is empty, oversized, invalid format, corrupted, or exceeds 20MP limit.
    """
    if not data:
        raise ValueError("Uploaded image is empty.")

    if len(data) > MAX_UPLOAD_SIZE:
        raise ValueError("Maximum file size is 10 MB.")

    if content_type:
        clean_type = content_type.lower().split(";")[0].strip()
        if clean_type not in ALLOWED_CONTENT_TYPES:
            raise ValueError(
                f"Unsupported image format '{content_type}'. "
                f"Allowed formats are JPEG, PNG, and WebP."
            )

    # Decode and handle EXIF rotation using Pillow
    try:
        pil_image = Image.open(io.BytesIO(data))
        pil_image.load()
        # Decompression bomb check
        width, height = pil_image.size
        if width * height > MAX_PIXELS:
            raise ValueError("Image exceeds the 20-megapixel limit.")

        # Respect camera orientation
        pil_image = ImageOps.exif_transpose(pil_image)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Invalid or corrupted image: {str(exc)}") from exc

    # Convert to RGB (handles RGBA, Palette, Grayscale)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    rgb_array = np.array(pil_image)
    if rgb_array is None or rgb_array.size == 0:
        raise ValueError("Invalid or corrupted image: failed to decode pixel buffer.")

    # Convert RGB to BGR for OpenCV compatibility
    image_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError("Invalid image dimensions.")

    return image_bgr, h, w


def encode_image_to_base64_png(image: np.ndarray) -> str:
    """Encode a BGR or 1-channel binary mask into a base64-encoded PNG string."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image to PNG format.")
    return base64.b64encode(buffer).decode("utf-8")


def create_overlay(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    alpha: float = 0.45,
    highlight_color: Tuple[int, int, int] = (0, 0, 255),  # BGR: Red
) -> np.ndarray:
    """
    Generate visual overlay with semi-transparent crack highlighting.
    Preserves original dimensions and returns unchanged copy if mask is empty.
    """
    overlay = image_bgr.copy()
    mask_bool = binary_mask > 0
    if not np.any(mask_bool):
        return overlay

    color_layer = np.zeros_like(image_bgr)
    color_layer[mask_bool] = highlight_color

    # Blend highlighted pixels with original pixels
    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, color_layer, alpha, 0)
    overlay[mask_bool] = blended[mask_bool]

    return overlay
