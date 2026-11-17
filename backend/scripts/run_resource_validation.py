"""
Sequential GPU Resource Validation Script for Phase 1.
Measures empirical VRAM allocation, reservation, latency, and loading success
for each staged model sequentially on CUDA:0.
"""
import gc
import time
from pathlib import Path
import cv2
import numpy as np
import torch

def reset_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

def measure_model_1_crack():
    reset_cuda()
    print("--- 1. Testing crack_segmentation_best.pt (YOLO26s-seg) ---")
    from ultralytics import YOLO
    
    ckpt = Path("backend/models/damage/crack_segmentation_best.pt")
    t0 = time.perf_counter()
    model = YOLO(str(ckpt))
    load_time = (time.perf_counter() - t0) * 1000
    
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Warmup
    _ = model(test_img, device="cuda:0", verbose=False)
    torch.cuda.synchronize()
    
    # Timed inference
    latencies = []
    for _ in range(5):
        t_start = time.perf_counter()
        _ = model(test_img, device="cuda:0", verbose=False)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t_start) * 1000)
    
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
    avg_lat = np.mean(latencies)
    
    print(f"  Load Time:      {load_time:.2f} ms")
    print(f"  Avg Latency:    {avg_lat:.2f} ms (5 runs)")
    print(f"  Peak Allocated: {peak_alloc:.2f} MB")
    print(f"  Peak Reserved:  {peak_res:.2f} MB")
    del model
    reset_cuda()
    return {"name": "crack_segmentation_best.pt", "load_ms": load_time, "latency_ms": avg_lat, "alloc_mb": peak_alloc, "res_mb": peak_res}

def measure_model_2_industrial():
    reset_cuda()
    print("\n--- 2. Testing industrial_defect_yolo26s.pt (YOLO26s-detect) ---")
    from ultralytics import YOLO
    
    ckpt = Path("backend/models/damage/industrial_defect_yolo26s.pt")
    t0 = time.perf_counter()
    model = YOLO(str(ckpt))
    load_time = (time.perf_counter() - t0) * 1000
    
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model(test_img, device="cuda:0", verbose=False)
    torch.cuda.synchronize()
    
    latencies = []
    for _ in range(5):
        t_start = time.perf_counter()
        _ = model(test_img, device="cuda:0", verbose=False)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t_start) * 1000)
        
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
    avg_lat = np.mean(latencies)
    
    print(f"  Load Time:      {load_time:.2f} ms")
    print(f"  Avg Latency:    {avg_lat:.2f} ms (5 runs)")
    print(f"  Peak Allocated: {peak_alloc:.2f} MB")
    print(f"  Peak Reserved:  {peak_res:.2f} MB")
    del model
    reset_cuda()
    return {"name": "industrial_defect_yolo26s.pt", "load_ms": load_time, "latency_ms": avg_lat, "alloc_mb": peak_alloc, "res_mb": peak_res}

def measure_model_3_yoloworld():
    reset_cuda()
    print("\n--- 3. Testing yolov8s-worldv2.pt (YOLO-World v2 Small) ---")
    from ultralytics import YOLOWorld
    
    ckpt = Path("backend/models/yolov8s-worldv2.pt")
    t0 = time.perf_counter()
    model = YOLOWorld(str(ckpt))
    model.set_classes(["wall", "column", "beam", "window", "door"])
    load_time = (time.perf_counter() - t0) * 1000
    
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model(test_img, device="cuda:0", verbose=False)
    torch.cuda.synchronize()
    
    latencies = []
    for _ in range(5):
        t_start = time.perf_counter()
        _ = model(test_img, device="cuda:0", verbose=False)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t_start) * 1000)
        
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
    avg_lat = np.mean(latencies)
    
    print(f"  Load Time:      {load_time:.2f} ms")
    print(f"  Avg Latency:    {avg_lat:.2f} ms (5 runs)")
    print(f"  Peak Allocated: {peak_alloc:.2f} MB")
    print(f"  Peak Reserved:  {peak_res:.2f} MB")
    del model
    reset_cuda()
    return {"name": "yolov8s-worldv2.pt", "load_ms": load_time, "latency_ms": avg_lat, "alloc_mb": peak_alloc, "res_mb": peak_res}

def measure_model_4_easyocr():
    reset_cuda()
    print("\n--- 4. Testing EasyOCR (CRAFT + CRNN on CUDA) ---")
    import easyocr
    
    t0 = time.perf_counter()
    reader = easyocr.Reader(['en'], gpu=True)
    load_time = (time.perf_counter() - t0) * 1000
    
    test_img = np.full((120, 400, 3), 255, dtype=np.uint8)
    cv2.putText(test_img, "INSPECTION TEST", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Warmup
    _ = reader.readtext(test_img)
    torch.cuda.synchronize()
    
    latencies = []
    for _ in range(5):
        t_start = time.perf_counter()
        _ = reader.readtext(test_img)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t_start) * 1000)
        
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
    avg_lat = np.mean(latencies)
    
    print(f"  Load Time:      {load_time:.2f} ms")
    print(f"  Avg Latency:    {avg_lat:.2f} ms (5 runs)")
    print(f"  Peak Allocated: {peak_alloc:.2f} MB")
    print(f"  Peak Reserved:  {peak_res:.2f} MB")
    del reader
    reset_cuda()
    return {"name": "EasyOCR (GPU)", "load_ms": load_time, "latency_ms": avg_lat, "alloc_mb": peak_alloc, "res_mb": peak_res}

def measure_model_5_resnet18():
    reset_cuda()
    print("\n--- 5. Testing ResNet18 (Torchvision ImageNet Baseline) ---")
    import torchvision.models as models
    
    t0 = time.perf_counter()
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = model.to("cuda:0").eval()
    load_time = (time.perf_counter() - t0) * 1000
    
    dummy_x = torch.randn(1, 3, 224, 224, device="cuda:0")
    with torch.no_grad():
        _ = model(dummy_x)
    torch.cuda.synchronize()
    
    latencies = []
    for _ in range(5):
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_x)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t_start) * 1000)
        
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
    avg_lat = np.mean(latencies)
    
    print(f"  Load Time:      {load_time:.2f} ms")
    print(f"  Avg Latency:    {avg_lat:.2f} ms (5 runs)")
    print(f"  Peak Allocated: {peak_alloc:.2f} MB")
    print(f"  Peak Reserved:  {peak_res:.2f} MB")
    del model
    reset_cuda()
    return {"name": "ResNet18 (ImageNet)", "load_ms": load_time, "latency_ms": avg_lat, "alloc_mb": peak_alloc, "res_mb": peak_res}

if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")
    
    r1 = measure_model_1_crack()
    r2 = measure_model_2_industrial()
    r3 = measure_model_3_yoloworld()
    r4 = measure_model_4_easyocr()
    r5 = measure_model_5_resnet18()
    
    print("\n=== SUMMARY OF EMPIRICAL RESOURCE MEASUREMENTS ===")
    print(f"{'Model':<30} | {'Load (ms)':<10} | {'Latency (ms)':<12} | {'Alloc (MB)':<10} | {'Res (MB)':<10}")
    print("-" * 82)
    for r in [r1, r2, r3, r4, r5]:
        print(f"{r['name']:<30} | {r['load_ms']:<10.1f} | {r['latency_ms']:<12.1f} | {r['alloc_mb']:<10.1f} | {r['res_mb']:<10.1f}")
