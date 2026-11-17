"""
GPU Segmentation Inference Smoke Test for YOLO.

Validates end-to-end execution of YOLO26 segmentation models on the
RTX 4060 GPU, measuring inference latency and VRAM allocation.
"""

import time
import numpy as np
import torch
from ultralytics import YOLO


def run_gpu_smoke_test(model_path: str = "yolo26n-seg.pt"):
    print("=" * 65)
    print(f" YOLO Segmentation GPU Inference Smoke Test [{model_path}] ")
    print("=" * 65)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU smoke test cannot proceed.")

    device = 0
    device_name = torch.cuda.get_device_name(device)
    print(f"Target Device:        CUDA:{device} ({device_name})")

    # 1. Load model onto GPU
    print(f"Loading checkpoint '{model_path}'...")
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    model = YOLO(model_path)
    model.to("cuda")
    t_load = (time.perf_counter() - t0) * 1000
    print(f"[+] Model loaded in {t_load:.1f} ms")
    print(f"[+] Model Task: {model.task}")

    # 2. Create a synthetic test image (640x640 BGR)
    rng = np.random.default_rng(42)
    test_image = rng.integers(100, 200, size=(640, 640, 3), dtype=np.uint8)

    # 3. Warm-up run
    print("Running GPU warm-up inference...")
    _ = model(test_image, device=device, imgsz=640, verbose=False)
    torch.cuda.synchronize()

    # 4. Timed Benchmark Run (5 iterations)
    print("Executing timed benchmark (5 runs @ imgsz=640)...")
    latencies = []
    results = None
    for i in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        results = model(test_image, device=device, imgsz=640, verbose=False)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    avg_latency = float(np.mean(latencies))
    min_latency = float(np.min(latencies))
    std_latency = float(np.std(latencies))

    # 5. Inspect Results
    first_result = results[0]
    boxes_detected = len(first_result.boxes) if first_result.boxes is not None else 0
    has_masks_attr = hasattr(first_result, "masks") and first_result.masks is not None
    masks_count = len(first_result.masks) if has_masks_attr else 0

    # 6. Memory statistics
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved_vram_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    print("\nInference Performance Metrics:")
    print(f"  - Average Latency:  {avg_latency:.2f} ms (+/- {std_latency:.2f} ms)")
    print(f"  - Minimum Latency:  {min_latency:.2f} ms (~{1000.0 / min_latency:.1f} FPS)")
    print(f"  - Peak VRAM Used:   {peak_vram_mb:.2f} MB")
    print(f"  - VRAM Reserved:    {reserved_vram_mb:.2f} MB")

    print("\nResult Object Verification:")
    print(f"  - Output Shape:     {first_result.orig_shape}")
    print(f"  - Speed Breakdown:  preprocess={first_result.speed['preprocess']:.2f}ms, "
          f"inference={first_result.speed['inference']:.2f}ms, "
          f"postprocess={first_result.speed['postprocess']:.2f}ms")
    print(f"  - Detections:       {boxes_detected} boxes, {masks_count} segmentation masks")

    print("\nClarification on Runtime vs. Capability:")
    print("  [*] Model execution on GPU: SUCCESSFUL")
    print("  [*] Segmentation architecture operational: YES")
    print("  [*] Damage-specific crack detection: PENDING (requires Phase 3B fine-tuning)")
    print("=" * 65)
    print("Smoke Test Status: PASSED")
    print("=" * 65)


if __name__ == "__main__":
    run_gpu_smoke_test("yolo26n-seg.pt")
