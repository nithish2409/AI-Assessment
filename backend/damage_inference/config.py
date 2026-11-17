"""
Configuration settings for the YOLO Damage Inference Microservice.
"""
import os
from pathlib import Path
from typing import Set

# Path resolution independent of current working directory
PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Server networking
HOST: str = os.getenv("DAMAGE_INFERENCE_HOST", "127.0.0.1")
PORT: int = int(os.getenv("DAMAGE_INFERENCE_PORT", "8001"))

# Checkpoint paths
CHECKPOINT_PATH: Path = Path(
    os.getenv(
        "DAMAGE_CHECKPOINT_PATH",
        str(BACKEND_DIR / "models" / "damage" / "crack_segmentation_best.pt")
    )
).resolve()

METADATA_PATH: Path = Path(
    os.getenv(
        "DAMAGE_METADATA_PATH",
        str(BACKEND_DIR / "models" / "damage" / "metadata.json")
    )
).resolve()

# Inference defaults
DEFAULT_IMGSZ: int = int(os.getenv("DAMAGE_IMGSZ", "640"))
DEFAULT_CONF: float = float(os.getenv("DAMAGE_CONF", "0.25"))
DEFAULT_IOU: float = float(os.getenv("DAMAGE_IOU", "0.45"))
MAX_CONCURRENT_INFERENCES: int = int(os.getenv("DAMAGE_MAX_CONCURRENCY", "2"))

# Image upload limits
MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB
MAX_PIXELS: int = 20_000_000             # 20 Megapixels
ALLOWED_CONTENT_TYPES: Set[str] = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
