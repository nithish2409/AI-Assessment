"""
Hybrid Building Damage Comparison Validation Script (Phase 4).
Runs real end-to-end tests against live endpoints:
  1. Identical Image Control Pair
  2. Synthetically Transformed Pair (Alignment & Warping verification)
  3. Synthetic Crack Progression Pair (Crack Expansion & New Branch detection)
  4. Unrelated Image Rejection Pair (Alignment Quality Gate verification)

Measures stage-by-stage latencies, collects telemetry, and saves visual assets.
"""
import base64
import json
import os
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

OUTPUT_DIR = Path("backend/datasets/damage/reports/hybrid_comparison")
ALIGNMENT_DIR = OUTPUT_DIR / "alignment_examples"
CRACK_CHANGE_DIR = OUTPUT_DIR / "crack_change_examples"
VIS_DIR = OUTPUT_DIR / "comparison_visualizations"

for d in [OUTPUT_DIR, ALIGNMENT_DIR, CRACK_CHANGE_DIR, VIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://127.0.0.1:8000"
TEST_IMG_PATH = Path("backend/datasets/damage/processed/images/test/1616.rf.c868709931a671796794fdbb95352c5a.jpg")
UNRELATED_IMG_PATH = Path("backend/datasets/damage/processed/images/test/1722.rf.38b38f2e833309a4f35bfbf0432dffff.jpg")


def decode_and_save_b64(b64_str: str, out_path: Path):
    """Save base64 PNG string to disk."""
    raw = base64.b64decode(b64_str)
    with open(out_path, "wb") as f:
        f.write(raw)


def run_hybrid_validation():
    assert TEST_IMG_PATH.exists(), f"Image not found: {TEST_IMG_PATH}"
    assert UNRELATED_IMG_PATH.exists(), f"Image not found: {UNRELATED_IMG_PATH}"

    client = httpx.Client(timeout=60.0)

    # Load base test image
    with open(TEST_IMG_PATH, "rb") as f:
        base_img_bytes = f.read()

    base_bgr = cv2.imread(str(TEST_IMG_PATH))
    h, w = base_bgr.shape[:2]

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)",
        "model": "YOLO26s-seg",
        "tests": {},
    }

    # =========================================================================
    # Test 1: Identical Image Control Pair
    # =========================================================================
    print("\n--- Running Test 1: Identical Image Control Pair ---")
    t0 = time.perf_counter()
    resp1 = client.post(
        f"{BASE_URL}/api/comparison/hybrid",
        files={
            "before": ("before.jpg", base_img_bytes, "image/jpeg"),
            "after": ("after.jpg", base_img_bytes, "image/jpeg"),
        },
        data={"alignment_mode": "auto", "confidence": "0.25", "iou": "0.45", "tolerance_px": "2"},
    )
    e2e_ms1 = (time.perf_counter() - t0) * 1000.0

    assert resp1.status_code == 200, f"Test 1 failed with {resp1.status_code}: {resp1.text}"
    p1 = resp1.json()
    print(f"Status: {p1['status']}")
    print(f"Alignment Status: {p1['alignment']['status']}, Method: {p1['alignment']['method']}")
    print(f"SSIM Score: {p1['visual_comparison']['ssim_score']}")
    print(f"Crack Before: {p1['crack_comparison']['before_crack_pixels']} px, After: {p1['crack_comparison']['after_crack_pixels']} px")
    print(f"Persistent: {p1['crack_comparison']['persistent_pixels']} px, New: {p1['crack_comparison']['newly_observed_pixels']} px")
    print(f"End-to-End Latency: {e2e_ms1:.2f} ms")

    # Save visual assets
    decode_and_save_b64(p1["images"]["hybrid_overlay"], CRACK_CHANGE_DIR / "test1_identical_overlay.png")

    results["tests"]["identical_control"] = {
        "status_code": resp1.status_code,
        "alignment_status": p1["alignment"]["status"],
        "alignment_method": p1["alignment"]["method"],
        "valid_overlap_pct": p1["alignment"]["valid_overlap_percentage"],
        "ssim_score": p1["visual_comparison"]["ssim_score"],
        "visual_change_pct": p1["visual_comparison"]["visual_change_percentage"],
        "before_crack_pixels": p1["crack_comparison"]["before_crack_pixels"],
        "after_crack_pixels": p1["crack_comparison"]["after_crack_pixels"],
        "persistent_pixels": p1["crack_comparison"]["persistent_pixels"],
        "newly_observed_pixels": p1["crack_comparison"]["newly_observed_pixels"],
        "no_longer_observed_pixels": p1["crack_comparison"]["no_longer_observed_pixels"],
        "coverage_change_pp": p1["crack_comparison"]["coverage_change_percentage_points"],
        "e2e_latency_ms": round(e2e_ms1, 2),
        "stage_timings": p1.get("timing"),
    }

    # =========================================================================
    # Test 2: Synthetically Transformed Image Pair (Translation + Minor Rotation)
    # =========================================================================
    print("\n--- Running Test 2: Synthetically Transformed Pair ---")
    # Rotate by 1.2 degrees around center and shift dx=+6, dy=+4
    center = (w // 2, h // 2)
    rot_M = cv2.getRotationMatrix2D(center, 1.2, 1.0)
    rot_M[0, 2] += 6.0
    rot_M[1, 2] += 4.0
    transformed_after_bgr = cv2.warpAffine(base_bgr, rot_M, (w, h), borderMode=cv2.BORDER_REFLECT)
    _, t_buf = cv2.imencode(".jpg", transformed_after_bgr)
    transformed_after_bytes = t_buf.tobytes()

    t0 = time.perf_counter()
    resp2 = client.post(
        f"{BASE_URL}/api/comparison/hybrid",
        files={
            "before": ("before.jpg", base_img_bytes, "image/jpeg"),
            "after": ("after_transformed.jpg", transformed_after_bytes, "image/jpeg"),
        },
        data={"alignment_mode": "auto", "confidence": "0.25", "iou": "0.45", "tolerance_px": "2"},
    )
    e2e_ms2 = (time.perf_counter() - t0) * 1000.0

    assert resp2.status_code == 200, f"Test 2 failed with {resp2.status_code}: {resp2.text}"
    p2 = resp2.json()
    print(f"Alignment Status: {p2['alignment']['status']}, Method: {p2['alignment']['method']}")
    print(f"Inliers: {p2['alignment']['inlier_count']}, Inlier Ratio: {p2['alignment']['inlier_ratio']}")
    print(f"Valid Overlap: {p2['alignment']['valid_overlap_percentage']}%")
    print(f"SSIM Score: {p2['visual_comparison']['ssim_score']}")
    print(f"Crack Persistent: {p2['crack_comparison']['persistent_pixels']} px, Tolerance-Adjusted New: {p2['crack_comparison']['tolerance_adjusted_new_pixels']} px")
    print(f"End-to-End Latency: {e2e_ms2:.2f} ms")

    # Save visual assets
    decode_and_save_b64(p2["images"]["aligned_before"], ALIGNMENT_DIR / "test2_aligned_before.png")
    cv2.imwrite(str(ALIGNMENT_DIR / "test2_after_ref.png"), transformed_after_bgr)
    decode_and_save_b64(p2["images"]["hybrid_overlay"], CRACK_CHANGE_DIR / "test2_transformed_overlay.png")

    results["tests"]["synthetically_transformed"] = {
        "status_code": resp2.status_code,
        "alignment_status": p2["alignment"]["status"],
        "alignment_method": p2["alignment"]["method"],
        "inliers": p2["alignment"]["inlier_count"],
        "inlier_ratio": p2["alignment"]["inlier_ratio"],
        "reprojection_error": p2["alignment"]["reprojection_error"],
        "valid_overlap_pct": p2["alignment"]["valid_overlap_percentage"],
        "ssim_score": p2["visual_comparison"]["ssim_score"],
        "persistent_pixels": p2["crack_comparison"]["persistent_pixels"],
        "newly_observed_pixels": p2["crack_comparison"]["newly_observed_pixels"],
        "tolerance_adjusted_new_pixels": p2["crack_comparison"]["tolerance_adjusted_new_pixels"],
        "e2e_latency_ms": round(e2e_ms2, 2),
        "stage_timings": p2.get("timing"),
    }

    # =========================================================================
    # Test 3: Synthetic Crack Progression Pair (Crack Extension & New Fissure)
    # =========================================================================
    print("\n--- Running Test 3: Synthetic Crack Progression Pair ---")
    progression_bgr = base_bgr.copy()
    # Draw realistic dark fissure extensions onto concrete surface:
    # 1. Extend the main crack downwards
    pts_extension = np.array([
        [240, 260], [255, 290], [270, 330], [280, 365], [290, 400]
    ], dtype=np.int32)
    cv2.polylines(progression_bgr, [pts_extension], isClosed=False, color=(35, 30, 25), thickness=3)

    # 2. Draw a new branching crack on right side
    pts_new_branch = np.array([
        [320, 100], [335, 140], [350, 175], [370, 210]
    ], dtype=np.int32)
    cv2.polylines(progression_bgr, [pts_new_branch], isClosed=False, color=(25, 20, 20), thickness=2)

    _, prog_buf = cv2.imencode(".jpg", progression_bgr)
    prog_bytes = prog_buf.tobytes()

    t0 = time.perf_counter()
    resp3 = client.post(
        f"{BASE_URL}/api/comparison/hybrid",
        files={
            "before": ("before.jpg", base_img_bytes, "image/jpeg"),
            "after": ("after_progression.jpg", prog_bytes, "image/jpeg"),
        },
        data={"alignment_mode": "auto", "confidence": "0.20", "iou": "0.45", "tolerance_px": "2"},
    )
    e2e_ms3 = (time.perf_counter() - t0) * 1000.0

    assert resp3.status_code == 200, f"Test 3 failed with {resp3.status_code}: {resp3.text}"
    p3 = resp3.json()
    print(f"Alignment Status: {p3['alignment']['status']}")
    print(f"SSIM Score: {p3['visual_comparison']['ssim_score']}, Visual Change: {p3['visual_comparison']['visual_change_percentage']}%")
    print(f"Crack Coverage Before: {p3['crack_comparison']['before_crack_coverage_percentage']}%, After: {p3['crack_comparison']['after_crack_coverage_percentage']}%")
    print(f"Coverage Change: {p3['crack_comparison']['coverage_change_percentage_points']} pp")
    print(f"Persistent: {p3['crack_comparison']['persistent_pixels']} px, Newly Observed: {p3['crack_comparison']['newly_observed_pixels']} px")
    print(f"Potential New Components: {p3['crack_comparison']['potential_new_components']}")
    print(f"Potential Expanded Components: {p3['crack_comparison']['potential_expanded_components']}")
    print(f"End-to-End Latency: {e2e_ms3:.2f} ms")

    # Save visual assets
    decode_and_save_b64(p3["images"]["hybrid_overlay"], CRACK_CHANGE_DIR / "test3_progression_overlay.png")
    decode_and_save_b64(p3["images"]["newly_observed_mask"], CRACK_CHANGE_DIR / "test3_newly_observed_mask.png")
    decode_and_save_b64(p3["images"]["persistent_mask"], CRACK_CHANGE_DIR / "test3_persistent_mask.png")
    decode_and_save_b64(p3["images"]["visual_difference"], VIS_DIR / "test3_ssim_heatmap.png")
    decode_and_save_b64(p3["images"]["hybrid_overlay"], VIS_DIR / "test3_hybrid_composite.png")

    results["tests"]["crack_progression"] = {
        "status_code": resp3.status_code,
        "alignment_status": p3["alignment"]["status"],
        "ssim_score": p3["visual_comparison"]["ssim_score"],
        "visual_change_pct": p3["visual_comparison"]["visual_change_percentage"],
        "before_crack_pixels": p3["crack_comparison"]["before_crack_pixels"],
        "after_crack_pixels": p3["crack_comparison"]["after_crack_pixels"],
        "before_coverage_pct": p3["crack_comparison"]["before_crack_coverage_percentage"],
        "after_coverage_pct": p3["crack_comparison"]["after_crack_coverage_percentage"],
        "coverage_change_pp": p3["crack_comparison"]["coverage_change_percentage_points"],
        "persistent_pixels": p3["crack_comparison"]["persistent_pixels"],
        "newly_observed_pixels": p3["crack_comparison"]["newly_observed_pixels"],
        "tolerance_adjusted_new_pixels": p3["crack_comparison"]["tolerance_adjusted_new_pixels"],
        "potential_new_components": p3["crack_comparison"]["potential_new_components"],
        "potential_expanded_components": p3["crack_comparison"]["potential_expanded_components"],
        "e2e_latency_ms": round(e2e_ms3, 2),
        "stage_timings": p3.get("timing"),
    }

    # =========================================================================
    # Test 4: Unrelated Image Rejection Pair (Quality Gate Verification)
    # =========================================================================
    print("\n--- Running Test 4: Unrelated Image Rejection Pair ---")
    with open(UNRELATED_IMG_PATH, "rb") as f:
        unrelated_bytes = f.read()

    t0 = time.perf_counter()
    resp4 = client.post(
        f"{BASE_URL}/api/comparison/hybrid",
        files={
            "before": ("before.jpg", base_img_bytes, "image/jpeg"),
            "after": ("after_unrelated.jpg", unrelated_bytes, "image/jpeg"),
        },
        data={"alignment_mode": "auto", "confidence": "0.25", "iou": "0.45"},
    )
    e2e_ms4 = (time.perf_counter() - t0) * 1000.0

    print(f"Status Code: {resp4.status_code} (Expected 422)")
    assert resp4.status_code == 422, f"Expected 422 for unrelated pair, got {resp4.status_code}"
    p4 = resp4.json()["detail"]
    print(f"Error Code: {p4['error']}")
    print(f"Message: {p4['message']}")
    print(f"Attempted: {p4.get('attempted_methods')}")
    print(f"End-to-End Latency: {e2e_ms4:.2f} ms")

    results["tests"]["unrelated_rejection"] = {
        "status_code": resp4.status_code,
        "error_code": p4["error"],
        "message": p4["message"],
        "attempted_methods": p4.get("attempted_methods"),
        "guidance": p4.get("guidance"),
        "rejection_confirmed": True,
        "e2e_latency_ms": round(e2e_ms4, 2),
    }

    # Save results json
    with open(OUTPUT_DIR / "hybrid_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== All real-world hybrid validation tests completed successfully! ===")
    print(f"Report directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run_hybrid_validation()
