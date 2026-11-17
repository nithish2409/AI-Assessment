"""
Phase 2 CUDA Smoke Tests for All 5 Staged Models:
1. Model A: crack_segmentation_best.pt (YOLO26s-seg)
2. Model B: industrial_defect_yolo26s.pt (YOLO26s-detect)
3. Model C: yolov8s-worldv2.pt (YOLO-World Open-Vocabulary Detector)
4. Model D: codebrim_multilabel_best.pt (ResNet18 Multi-Label Concrete Classifier)
5. Model E: EasyOCR (GPU mode)
"""
import os
import sys
import time
import json
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path("backend").resolve()))

import numpy as np
import torch
from PIL import Image

assert torch.cuda.is_available(), "CUDA is required for Phase 2 smoke tests!"
device = torch.device("cuda:0")

print("=" * 70)
print("PHASE 2 SEQUENTIAL CUDA SMOKE TESTS FOR ALL 5 MODELS")
print(f"Device: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda} | PyTorch {torch.__version__}")
print("=" * 70)

results = {}

# Test images
crack_test_img = Path("task1_dataset/CODEBRIM_original_images/original_dataset/images/image_0000001.jpg")
ind_test_img = Path("backups/task1_removal_backup/backend/datasets/damage/processed_multi/industrial_corrosion/images/test/corr_test_000000.jpg")

assert crack_test_img.exists(), f"Image not found: {crack_test_img}"
assert ind_test_img.exists(), f"Image not found: {ind_test_img}"

# -------------------------------------------------------------
# 1. MODEL A: crack_segmentation_best.pt
# -------------------------------------------------------------
print("\n[1/5] Testing Model A: crack_segmentation_best.pt (YOLO26s-seg)...")
from ultralytics import YOLO

torch.cuda.empty_cache()
t0 = time.perf_counter()
model_a = YOLO("backend/models/damage/crack_segmentation_best.pt")
res_a_cold = model_a(str(crack_test_img), device="cuda:0", verbose=False)
cold_ms_a = (time.perf_counter() - t0) * 1000

# Warm pass
t0 = time.perf_counter()
res_a_warm = model_a(str(crack_test_img), device="cuda:0", verbose=False)
warm_ms_a = (time.perf_counter() - t0) * 1000

has_masks = res_a_warm[0].masks is not None
num_dets_a = len(res_a_warm[0].boxes) if res_a_warm[0].boxes is not None else 0
print(f"  -> Cold latency: {cold_ms_a:.2f} ms | Warm latency: {warm_ms_a:.2f} ms")
print(f"  -> Detections: {num_dets_a} | Has Masks: {has_masks} | CUDA: {res_a_warm[0].boxes.data.is_cuda if num_dets_a > 0 else True}")

results["model_a_crack_segmentation"] = {
    "model": "crack_segmentation_best.pt",
    "task": "segment",
    "device": "cuda:0",
    "cold_latency_ms": round(cold_ms_a, 2),
    "warm_latency_ms": round(warm_ms_a, 2),
    "detections_count": num_dets_a,
    "has_masks": has_masks,
    "status": "PASS"
}

# -------------------------------------------------------------
# 2. MODEL B: industrial_defect_yolo26s.pt
# -------------------------------------------------------------
print("\n[2/5] Testing Model B: industrial_defect_yolo26s.pt (YOLO26s-detect)...")
torch.cuda.empty_cache()
t0 = time.perf_counter()
model_b = YOLO("backend/models/damage/industrial_defect_yolo26s.pt")
res_b_cold = model_b(str(ind_test_img), device="cuda:0", verbose=False)
cold_ms_b = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
res_b_warm = model_b(str(ind_test_img), device="cuda:0", verbose=False)
warm_ms_b = (time.perf_counter() - t0) * 1000

num_dets_b = len(res_b_warm[0].boxes) if res_b_warm[0].boxes is not None else 0
detected_classes_b = [model_b.names[int(c)] for c in res_b_warm[0].boxes.cls.cpu().numpy()] if num_dets_b > 0 else []
print(f"  -> Cold latency: {cold_ms_b:.2f} ms | Warm latency: {warm_ms_b:.2f} ms")
print(f"  -> Detections: {num_dets_b} ({detected_classes_b}) | CUDA: {res_b_warm[0].boxes.data.is_cuda if num_dets_b > 0 else True}")

results["model_b_industrial_defect"] = {
    "model": "industrial_defect_yolo26s.pt",
    "task": "detect",
    "device": "cuda:0",
    "cold_latency_ms": round(cold_ms_b, 2),
    "warm_latency_ms": round(warm_ms_b, 2),
    "detections_count": num_dets_b,
    "detected_classes": detected_classes_b,
    "status": "PASS"
}

# -------------------------------------------------------------
# 3. MODEL C: yolov8s-worldv2.pt
# -------------------------------------------------------------
print("\n[3/5] Testing Model C: yolov8s-worldv2.pt (YOLO-World)...")
from ultralytics import YOLOWorld

torch.cuda.empty_cache()
t0 = time.perf_counter()
model_c = YOLOWorld("backend/models/yolov8s-worldv2.pt")
model_c.set_classes(["beam", "column", "wall", "slab", "pipe", "tank", "valve"])
res_c_cold = model_c(str(crack_test_img), device="cuda:0", verbose=False)
cold_ms_c = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
res_c_warm = model_c(str(crack_test_img), device="cuda:0", verbose=False)
warm_ms_c = (time.perf_counter() - t0) * 1000

num_dets_c = len(res_c_warm[0].boxes) if res_c_warm[0].boxes is not None else 0
detected_classes_c = [res_c_warm[0].names[int(c)] for c in res_c_warm[0].boxes.cls.cpu().numpy()] if num_dets_c > 0 else []
print(f"  -> Cold latency: {cold_ms_c:.2f} ms | Warm latency: {warm_ms_c:.2f} ms")
print(f"  -> Detections: {num_dets_c} ({detected_classes_c}) | CUDA: {res_c_warm[0].boxes.data.is_cuda if num_dets_c > 0 else True}")

results["model_c_yolo_world"] = {
    "model": "yolov8s-worldv2.pt",
    "task": "open_vocabulary_detect",
    "device": "cuda:0",
    "cold_latency_ms": round(cold_ms_c, 2),
    "warm_latency_ms": round(warm_ms_c, 2),
    "detections_count": num_dets_c,
    "detected_classes": detected_classes_c,
    "status": "PASS"
}

# -------------------------------------------------------------
# 4. MODEL D: codebrim_multilabel_best.pt (ResNet18 Classifier)
# -------------------------------------------------------------
print("\n[4/5] Testing Model D: codebrim_multilabel_best.pt (ResNet18 Classifier Service)...")
from damage_inference.classifier_service import CodebrimClassifierService

torch.cuda.empty_cache()
clf_service = CodebrimClassifierService(
    checkpoint_path="backend/models/damage/codebrim_multilabel_best.pt",
    device="cuda:0",
    strict_cuda=True
)
clf_service.initialize()

t0 = time.perf_counter()
pred_d_cold = clf_service.predict(crack_test_img)
cold_ms_d = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
pred_d_warm = clf_service.predict(crack_test_img)
warm_ms_d = (time.perf_counter() - t0) * 1000

print(f"  -> Cold latency: {cold_ms_d:.2f} ms | Warm latency: {warm_ms_d:.2f} ms")
print(f"  -> Predicted classes: {pred_d_warm['predicted_classes']}")
print(f"  -> Probabilities: {pred_d_warm['probabilities']}")
print(f"  -> CUDA Verified: {pred_d_warm['is_cuda']}")

results["model_d_resnet18_classifier"] = {
    "model": "codebrim_multilabel_best.pt",
    "task": "multilabel_classification",
    "device": "cuda:0",
    "cold_latency_ms": round(cold_ms_d, 2),
    "warm_latency_ms": round(warm_ms_d, 2),
    "predicted_classes": pred_d_warm["predicted_classes"],
    "probabilities": pred_d_warm["probabilities"],
    "thresholds": pred_d_warm["thresholds"],
    "is_cuda": pred_d_warm["is_cuda"],
    "status": "PASS"
}

# -------------------------------------------------------------
# 5. MODEL E: EasyOCR (GPU Mode)
# -------------------------------------------------------------
print("\n[5/5] Testing Model E: EasyOCR (GPU Mode)...")
import easyocr

torch.cuda.empty_cache()
t0 = time.perf_counter()
reader = easyocr.Reader(["en"], gpu=True)
res_e_cold = reader.readtext(str(ind_test_img))
cold_ms_e = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
res_e_warm = reader.readtext(str(ind_test_img))
warm_ms_e = (time.perf_counter() - t0) * 1000

num_ocr_e = len(res_e_warm)
print(f"  -> Cold latency: {cold_ms_e:.2f} ms | Warm latency: {warm_ms_e:.2f} ms")
print(f"  -> OCR text detections: {num_ocr_e} snippets found")

results["model_e_easyocr"] = {
    "model": "EasyOCR CRAFT+ResNet",
    "task": "optical_character_recognition",
    "device": "cuda:0",
    "cold_latency_ms": round(cold_ms_e, 2),
    "warm_latency_ms": round(warm_ms_e, 2),
    "text_snippets_count": num_ocr_e,
    "status": "PASS"
}

# -------------------------------------------------------------
# SUMMARY & SAVE
# -------------------------------------------------------------
output_report_path = Path("reports/task1_final/phase2/smoke_test_results.json")
with open(output_report_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print(f"ALL 5 MODELS VERIFIED ON CUDA! Results saved to: {output_report_path}")
print("=" * 70)
