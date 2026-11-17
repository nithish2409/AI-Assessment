"""
Unit and integration tests for YOLO damage inference service and image processing utilities.
"""
import io
import cv2
import numpy as np
import pytest
from pathlib import Path

# Add project root and backend to sys.path
import sys
backend_path = Path(__file__).resolve().parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from damage_inference.image_utils import (
    validate_and_decode_image,
    encode_image_to_base64_png,
    create_overlay,
)


def create_sample_image(width=300, height=200, color=(100, 100, 100)):
    img = np.full((height, width, 3), color, dtype=np.uint8)
    # Draw a line simulating a crack
    cv2.line(img, (20, 20), (width - 20, height - 20), (30, 30, 30), 2)
    success, encoded = cv2.imencode(".png", img)
    assert success
    return encoded.tobytes()


def test_validate_and_decode_valid():
    raw = create_sample_image(width=320, height=240)
    img_bgr, h, w = validate_and_decode_image(raw, "image/png")
    assert h == 240
    assert w == 320
    assert img_bgr.shape == (240, 320, 3)


def test_validate_and_decode_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_and_decode_image(b"", "image/jpeg")


def test_validate_and_decode_unsupported_type():
    raw = create_sample_image()
    with pytest.raises(ValueError, match="Unsupported image format"):
        validate_and_decode_image(raw, "application/pdf")


def test_validate_and_decode_corrupted():
    with pytest.raises(ValueError, match="Invalid or corrupted"):
        validate_and_decode_image(b"CORRUPTED_BYTES_HERE", "image/jpeg")


def test_validate_and_decode_exceeds_max_pixels():
    # Construct an image exceeding 20 Megapixels
    # 5000 x 4500 = 22.5 MP
    # Pillow verifies dimensions before allocating buffer
    from PIL import Image
    im = Image.new("RGB", (5000, 4500))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    with pytest.raises(ValueError, match="20-megapixel"):
        validate_and_decode_image(buf.getvalue(), "image/jpeg")


def test_encode_image_to_base64_png():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    b64_str = encode_image_to_base64_png(img)
    assert isinstance(b64_str, str)
    assert len(b64_str) > 50

    # Verify decodability
    import base64
    raw = base64.b64decode(b64_str)
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (100, 100, 3)


def test_create_overlay_non_empty():
    orig = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 255  # 20x20 square crack

    overlay = create_overlay(orig, mask, alpha=0.5, highlight_color=(0, 0, 255))
    assert overlay.shape == (100, 100, 3)
    # Outside mask, should be unchanged
    assert np.array_equal(overlay[0, 0], [128, 128, 128])
    # Inside mask, should be tinted red
    assert overlay[25, 25, 2] > overlay[25, 25, 0]


def test_create_overlay_empty_mask():
    orig = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    overlay = create_overlay(orig, mask)
    # Exactly identical
    assert np.array_equal(overlay, orig)


def test_crack_coverage_union_no_double_counting():
    """Verify overlapping polygon masks do not inflate coverage percentage."""
    h, w = 100, 100
    total_pixels = h * w
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    # Polygon 1: 0 to 20 (area 400)
    cv2.rectangle(binary_mask, (0, 0), (20, 20), 255, -1)
    # Polygon 2: 10 to 30 (area 400, overlap with poly1 is 10x20 = 200)
    cv2.rectangle(binary_mask, (10, 0), (30, 20), 255, -1)

    unique_crack_pixels = int(np.count_nonzero(binary_mask > 0))
    # Expected union area: 30 * 20 = 600 pixels (not 400 + 400 = 800)
    # (Note: cv2.rectangle (0,0) to (30,20) covers 31*21 = 651)
    assert unique_crack_pixels < 800
    coverage_pct = (unique_crack_pixels / total_pixels) * 100.0
    assert 5.0 <= coverage_pct <= 7.0


# ---------------------------------------------------------------------------
# PyTorch / Ultralytics Live Model Inference Tests (executed when torch exists)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def yolo_service():
    torch = pytest.importorskip("torch")
    from damage_inference.service import DamageInferenceService
    from damage_inference.config import CHECKPOINT_PATH

    service = DamageInferenceService(checkpoint_path=CHECKPOINT_PATH)
    initialized = service.initialize()
    assert initialized, f"Model initialization failed: {service.init_error}"
    return service


def test_yolo_checkpoint_loaded(yolo_service):
    readiness = yolo_service.get_readiness()
    assert readiness.ready is True
    assert readiness.model == "YOLO26s-seg"
    assert "crack" in readiness.classes.values()


def test_yolo_empty_detection_on_clean_image(yolo_service):
    clean = np.ones((640, 640, 3), dtype=np.uint8) * 200
    _, buf = cv2.imencode(".png", clean)
    import asyncio
    res = asyncio.run(yolo_service.segment_image(buf.tobytes(), "image/png"))
    assert res.status == "completed"
    assert res.detections_count == 0
    assert res.crack_coverage_percentage == 0.0
    assert len(res.warnings) > 0


def test_yolo_real_test_crack_image(yolo_service):
    test_img_path = backend_path / "datasets" / "damage" / "processed" / "images" / "test" / "1616.rf.c868709931a671796794fdbb95352c5a.jpg"
    if not test_img_path.exists():
        pytest.skip("Test image not found.")

    raw = test_img_path.read_bytes()
    import asyncio
    res = asyncio.run(yolo_service.segment_image(raw, "image/jpeg", confidence=0.25, iou=0.45))
    assert res.status == "completed"
    assert res.detections_count > 0
    assert res.crack_coverage_percentage > 0.0
    assert res.image_width > 0
    assert res.image_height > 0
    assert len(res.images.original) > 0
    assert len(res.images.mask) > 0
    assert len(res.images.overlay) > 0
    assert res.inference.duration_ms > 0
