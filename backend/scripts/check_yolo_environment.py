"""
Environment and GPU Acceleration Verification Script for YOLO Training.

Reports installed package versions, CUDA availability, device attributes,
and runs a CUDA tensor execution benchmark.
"""

import sys
import cv2
import torch
import torchvision
import ultralytics


def check_environment():
    print("=" * 60)
    print(" YOLO Environment and GPU Verification ")
    print("=" * 60)

    # 1. Package Versions
    print(f"Python Version:       {sys.version.split()[0]} ({sys.executable})")
    print(f"PyTorch Version:      {torch.__version__}")
    print(f"torchvision Version:  {torchvision.__version__}")
    print(f"Ultralytics Version:  {ultralytics.__version__}")
    print(f"OpenCV Version:       {cv2.__version__} (headless: {hasattr(cv2, 'AKAZE_create')})")

    # 2. CUDA & Device Information
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available:       {cuda_available}")

    if not cuda_available:
        print("[-] ERROR: CUDA is not available to PyTorch!")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    cuda_runtime = torch.version.cuda
    device_props = torch.cuda.get_device_properties(current_device)
    total_memory_gb = device_props.total_memory / (1024 ** 3)
    compute_capability = f"{device_props.major}.{device_props.minor}"

    print(f"CUDA Runtime Version: {cuda_runtime}")
    print(f"GPU Device Count:     {device_count}")
    print(f"Current GPU (0):      {device_name}")
    print(f"Total VRAM:           {total_memory_gb:.2f} GB")
    print(f"Compute Capability:   {compute_capability}")

    # 3. Functional GPU Tensor Operation
    print("\nRunning GPU Tensor Smoke Test...")
    torch.cuda.empty_cache()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    c = torch.matmul(a, b)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)

    print(f"[+] Matrix multiplication (2048x2048 float32) succeeded in {elapsed_ms:.2f} ms")
    print(f"[+] Memory allocated: {allocated_mb:.2f} MB | Reserved: {reserved_mb:.2f} MB")
    print("=" * 60)
    print("Environment Verification: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    check_environment()
