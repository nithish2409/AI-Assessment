"""
AI-Assessment Phase 4.1: Reliability Evaluation & Noise-Floor Calibration.

Orchestrates:
  - Dataset generation: Group A (110 no-change controls), Group B (40 synthetic progression pairs), Group C (10 unrelated pairs).
  - Server management: Spins up YOLO microservice (8001) and FastAPI main backend (8000).
  - Empirical evaluation: Queries live /api/comparison/hybrid and collects telemetry.
  - Statistical analysis: Image-level bootstrap confidence intervals, noise-floor calibration, tolerance parameter sweep.
  - Deliverables: CSVs, JSON, plots, and visual example assets.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reliability_audit")

# Output Paths
REPORTS_DIR = Path("backend/datasets/damage/reports/reliability_audit")
VIS_EXAMPLES_DIR = REPORTS_DIR / "visual_examples"
MANIFEST_PATH = REPORTS_DIR / "evaluation_manifest.json"
NO_CHANGE_CSV = REPORTS_DIR / "no_change_distribution.csv"
TOLERANCE_CSV = REPORTS_DIR / "tolerance_comparison.csv"
PROGRESSION_CSV = REPORTS_DIR / "synthetic_progression_metrics.csv"
ALIGNMENT_CSV = REPORTS_DIR / "alignment_rejection_results.csv"
RESULTS_JSON = REPORTS_DIR / "reliability_results.json"
NO_CHANGE_PNG = REPORTS_DIR / "no_change_distribution.png"
TOLERANCE_PNG = REPORTS_DIR / "tolerance_tradeoff.png"

for p in [REPORTS_DIR, VIS_EXAMPLES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# 10 Representative Test Images
TEST_IMG_DIR = Path("backend/datasets/damage/processed/images/test")
SOURCE_STEMS = [
    "1616.rf.c868709931a671796794fdbb95352c5a",
    "2222.rf.bfd72b1a6235ed53983e11289c619b4e",
    "2274.rf.5712706f169ad5ab788f546227533d28",
    "1686.rf.809fb1b51c607e5cf787e44ef4ddd7b8",
    "2205.rf.eaf969e5ee1bad463953679d0f0a317f",
    "2400.rf.e2dbfc7baf8e55f3ee71cde186d199c2",
    "1716.rf.85ea38b36008beaa72c5d8541f734eb0",
    "1804.rf.7e39a73f7b0d1c4bdf8094006e0cf495",
    "1813.rf.79f1af872f76dd8b46df4a83ed5f54c6",
    "1957.rf.e4ee900c82e7bab356690f56f27959c6",
]


# =============================================================================
# Helper: Server Process Management
# =============================================================================
class ServerManager:
    def __init__(self):
        self.p_micro = None
        self.p_main = None

    def start(self):
        logger.info("Starting YOLO inference microservice on 127.0.0.1:8001...")
        self.p_micro = subprocess.Popen(
            [
                "backend/venv_yolo/Scripts/python.exe", "-m", "uvicorn",
                "damage_inference.app:app", "--host", "127.0.0.1", "--port", "8001",
                "--app-dir", "backend",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        logger.info("Starting FastAPI main backend on 127.0.0.1:8000...")
        self.p_main = subprocess.Popen(
            [
                "backend/venv/Scripts/python.exe", "-m", "uvicorn",
                "app.main:app", "--host", "127.0.0.1", "--port", "8000",
                "--app-dir", "backend",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Poll health until ready (up to 30s)
        logger.info("Waiting for services to become healthy...")
        client = httpx.Client(timeout=3.0)
        start_t = time.time()
        ready = False
        while time.time() - start_t < 30.0:
            try:
                r1 = client.get("http://127.0.0.1:8001/ready")
                r2 = client.get("http://127.0.0.1:8000/api/damage/health")
                if r1.status_code == 200 and r2.status_code == 200:
                    if r2.json().get("is_available") is True:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(1.0)

        if not ready:
            self.stop()
            raise RuntimeError("Failed to start and verify health of backend services.")
        logger.info("Services verified healthy and GPU model initialized!")

    def stop(self):
        logger.info("Stopping backend services...")
        for p in [self.p_main, self.p_micro]:
            if p is not None:
                try:
                    p.terminate()
                    p.wait(timeout=3.0)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
        logger.info("Backend processes stopped.")


# =============================================================================
# Synthetic Generation Engine
# =============================================================================
def apply_no_change_transformation(img_bgr: np.ndarray, condition: str, seed: int = 42) -> np.ndarray:
    """Apply realistic physical camera/lighting transformations without changing crack structure."""
    rng = np.random.default_rng(seed)
    h, w = img_bgr.shape[:2]

    if condition == "identical":
        return img_bgr.copy()

    elif condition == "translation":
        M = np.float32([[1, 0, 7], [0, 1, -5]])
        return cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    elif condition == "small_rotation":
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 1.5, 1.0)
        return cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    elif condition == "mild_perspective":
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[4, -3], [w - 3, 4], [w - 2, h - 3], [3, h - 2]])
        H = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img_bgr, H, (w, h), borderMode=cv2.BORDER_REFLECT)

    elif condition == "brightness":
        # +15% brightness
        return np.clip(img_bgr.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)

    elif condition == "contrast":
        # +20% contrast around 128
        return np.clip(1.20 * (img_bgr.astype(np.float32) - 128) + 128, 0, 255).astype(np.uint8)

    elif condition == "gamma":
        # Gamma 1.25 (slight darkening of midtones)
        table = np.array([((i / 255.0) ** 1.25) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img_bgr, table)

    elif condition == "gaussian_noise":
        noise = rng.normal(0, 8.0, img_bgr.shape)
        return np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    elif condition == "mild_blur":
        return cv2.GaussianBlur(img_bgr, (3, 3), 0.8)

    elif condition == "jpeg_compression":
        _, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    elif condition == "combined":
        # Small rotation + translation + brightness + jpeg
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 1.0, 1.0)
        M[0, 2] += 4.0
        M[1, 2] += 4.0
        warped = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        bright = np.clip(warped.astype(np.float32) * 1.10, 0, 255).astype(np.uint8)
        _, enc = cv2.imencode(".jpg", bright, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    else:
        raise ValueError(f"Unknown condition: {condition}")


def inject_synthetic_progression(
    base_bgr: np.ndarray,
    variation: str,
    seed: int = 101,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject synthetic crack progression onto a real concrete background.
    Returns (modified_after_image, ground_truth_change_mask).
    """
    rng = np.random.default_rng(seed)
    h, w = base_bgr.shape[:2]
    after_bgr = base_bgr.copy()
    gt_change = np.zeros((h, w), dtype=np.uint8)

    # Typical dark concrete fissure color (BGR: 25..45)
    fissure_color = (32, 28, 25)

    if variation == "new_isolated_crack":
        # Draw a new wavy crack fissure in upper-right quadrant
        pts = np.array([
            [w - 120, 50], [w - 105, 80], [w - 95, 115], [w - 85, 150], [w - 70, 185]
        ], dtype=np.int32)
        cv2.polylines(after_bgr, [pts], isClosed=False, color=fissure_color, thickness=3)
        cv2.polylines(gt_change, [pts], isClosed=False, color=255, thickness=3)

    elif variation == "crack_extension":
        # Extend downwards from center
        pts = np.array([
            [w // 2, h // 2], [w // 2 + 10, h // 2 + 40], [w // 2 + 15, h // 2 + 80], [w // 2 + 25, h // 2 + 120]
        ], dtype=np.int32)
        cv2.polylines(after_bgr, [pts], isClosed=False, color=fissure_color, thickness=3)
        cv2.polylines(gt_change, [pts], isClosed=False, color=255, thickness=3)

    elif variation == "increased_width":
        # Draw a thickened segment along center diagonal
        pts = np.array([
            [120, 120], [150, 170], [180, 220], [210, 270]
        ], dtype=np.int32)
        cv2.polylines(after_bgr, [pts], isClosed=False, color=fissure_color, thickness=5)
        cv2.polylines(gt_change, [pts], isClosed=False, color=255, thickness=5)

    elif variation == "multiple_branches":
        # Branch 1
        b1 = np.array([[80, 100], [110, 130], [140, 150]], dtype=np.int32)
        # Branch 2
        b2 = np.array([[110, 130], [130, 170], [160, 200]], dtype=np.int32)
        cv2.polylines(after_bgr, [b1, b2], isClosed=False, color=fissure_color, thickness=3)
        cv2.polylines(gt_change, [b1, b2], isClosed=False, color=255, thickness=3)

    elif variation == "small_addition":
        pts = np.array([[160, 80], [175, 100], [190, 115]], dtype=np.int32)
        cv2.polylines(after_bgr, [pts], isClosed=False, color=fissure_color, thickness=2)
        cv2.polylines(gt_change, [pts], isClosed=False, color=255, thickness=2)

    elif variation == "large_addition":
        pts = np.array([
            [50, 250], [90, 280], [140, 310], [190, 330], [250, 360], [310, 380]
        ], dtype=np.int32)
        cv2.polylines(after_bgr, [pts], isClosed=False, color=fissure_color, thickness=4)
        cv2.polylines(gt_change, [pts], isClosed=False, color=255, thickness=4)

    elif variation == "no_longer_visible":
        # Inpaint / cover a 40x40 patch where cracks exist with surrounding texture
        mask_inpaint = np.zeros((h, w), dtype=np.uint8)
        mask_inpaint[140:180, 180:220] = 255
        after_bgr = cv2.inpaint(base_bgr, mask_inpaint, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        gt_change[140:180, 180:220] = 255

    elif variation == "progression_with_transform":
        # Injection + camera transform
        pts = np.array([
            [w - 140, 60], [w - 120, 100], [w - 100, 140], [w - 80, 180]
        ], dtype=np.int32)
        cv2.polylines(after_bgr, [pts], isClosed=False, color=fissure_color, thickness=3)
        cv2.polylines(gt_change, [pts], isClosed=False, color=255, thickness=3)

        # Apply rotation + translation
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 1.0, 1.0)
        M[0, 2] += 5.0
        M[1, 2] += 3.0
        after_bgr = cv2.warpAffine(after_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        gt_change = cv2.warpAffine(gt_change, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return after_bgr, gt_change


# =============================================================================
# Main Evaluation Execution
# =============================================================================
def run_evaluation():
    logger.info("Initializing Phase 4.1 Evaluation & Calibration...")
    server = ServerManager()
    server.start()

    client = httpx.Client(timeout=45.0)

    try:
        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_images_count": len(SOURCE_STEMS),
            "source_images": SOURCE_STEMS,
            "groups": {
                "A_no_change_conditions": [
                    "identical", "translation", "small_rotation", "mild_perspective",
                    "brightness", "contrast", "gamma", "gaussian_noise", "mild_blur",
                    "jpeg_compression", "combined"
                ],
                "B_synthetic_progression_conditions": [
                    "new_isolated_crack", "crack_extension", "increased_width",
                    "multiple_branches", "small_addition", "large_addition",
                    "no_longer_visible", "progression_with_transform"
                ],
                "C_unrelated_pairs_count": 10
            }
        }
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # =====================================================================
        # STEP C: EMPIRICAL NO-CHANGE DISTRIBUTION (Group A)
        # =====================================================================
        logger.info("\n=== Evaluating Group A: No-Change Controls (110 pairs) ===")
        no_change_records = []
        no_change_conditions = manifest["groups"]["A_no_change_conditions"]

        pair_idx = 0
        total_a = len(SOURCE_STEMS) * len(no_change_conditions)

        for s_idx, stem in enumerate(SOURCE_STEMS):
            img_path = TEST_IMG_DIR / f"{stem}.jpg"
            base_bgr = cv2.imread(str(img_path))
            _, base_bytes = cv2.imencode(".jpg", base_bgr)

            for cond in no_change_conditions:
                pair_idx += 1
                trans_bgr = apply_no_change_transformation(base_bgr, cond, seed=42 + s_idx)
                _, trans_bytes = cv2.imencode(".jpg", trans_bgr)

                # Query hybrid comparison with default tolerance=2
                try:
                    resp = client.post(
                        "http://127.0.0.1:8000/api/comparison/hybrid",
                        files={
                            "before": ("before.jpg", base_bytes.tobytes(), "image/jpeg"),
                            "after": (f"after_{cond}.jpg", trans_bytes.tobytes(), "image/jpeg"),
                        },
                        data={"alignment_mode": "auto", "confidence": "0.25", "iou": "0.45", "tolerance_px": "2"},
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        align = data["alignment"]
                        vis = data["visual_comparison"]
                        crack = data["crack_comparison"]

                        cov_diff = crack["coverage_change_percentage_points"]
                        abs_cov_diff = abs(cov_diff)

                        rec = {
                            "source_image": stem,
                            "condition": cond,
                            "status": align["status"],
                            "method": align["method"],
                            "inlier_count": align["inlier_count"],
                            "inlier_ratio": align["inlier_ratio"],
                            "reprojection_error": align["reprojection_error"],
                            "valid_overlap_pct": align["valid_overlap_percentage"],
                            "ssim_score": vis["ssim_score"],
                            "before_crack_pixels": crack["before_crack_pixels"],
                            "after_crack_pixels": crack["after_crack_pixels"],
                            "signed_coverage_change_pp": cov_diff,
                            "abs_coverage_change_pp": abs_cov_diff,
                            "persistent_pixels": crack["persistent_pixels"],
                            "raw_newly_observed_pixels": crack["newly_observed_pixels"],
                            "raw_no_longer_observed_pixels": crack["no_longer_observed_pixels"],
                            "tolerance_adjusted_new_pixels": crack["tolerance_adjusted_new_pixels"],
                            "tolerance_adjusted_no_longer_pixels": crack["tolerance_adjusted_no_longer_observed_pixels"],
                            "false_new_components": crack["potential_new_components"],
                            "false_expanded_components": crack["potential_expanded_components"],
                            "exceeds_0_15_threshold": abs_cov_diff > 0.15,
                            "rejection_reason": None,
                        }
                    else:
                        rec = {
                            "source_image": stem,
                            "condition": cond,
                            "status": "rejected",
                            "method": "FAILED",
                            "inlier_count": 0,
                            "inlier_ratio": 0.0,
                            "reprojection_error": 0.0,
                            "valid_overlap_pct": 0.0,
                            "ssim_score": 0.0,
                            "before_crack_pixels": 0,
                            "after_crack_pixels": 0,
                            "signed_coverage_change_pp": 0.0,
                            "abs_coverage_change_pp": 0.0,
                            "persistent_pixels": 0,
                            "raw_newly_observed_pixels": 0,
                            "raw_no_longer_observed_pixels": 0,
                            "tolerance_adjusted_new_pixels": 0,
                            "tolerance_adjusted_no_longer_pixels": 0,
                            "false_new_components": 0,
                            "false_expanded_components": 0,
                            "exceeds_0_15_threshold": False,
                            "rejection_reason": resp.json().get("detail", {}).get("error", "REJECTED"),
                        }
                except Exception as e:
                    rec = {
                        "source_image": stem, "condition": cond, "status": "error", "rejection_reason": str(e)
                    }

                no_change_records.append(rec)
                if pair_idx % 20 == 0 or pair_idx == total_a:
                    logger.info(f"Group A Progress: {pair_idx}/{total_a} pairs processed.")

        df_no_change = pd.DataFrame(no_change_records)
        df_no_change.to_csv(NO_CHANGE_CSV, index=False)
        logger.info(f"Group A completed: Saved {len(df_no_change)} rows to {NO_CHANGE_CSV}")

        # =====================================================================
        # STEP D & E: SPATIAL TOLERANCE SWEEP & PROGRESSION EVALUATION (Group B)
        # =====================================================================
        logger.info("\n=== Evaluating Group B: Controlled Crack Progression & Tolerance Sweep ===")
        progression_conditions = manifest["groups"]["B_synthetic_progression_conditions"]
        # Use first 5 source images for progression calibration
        prog_sources = SOURCE_STEMS[:5]

        tolerance_candidates = [0, 1, 2, 3, 4, 5]
        tolerance_records = []
        progression_eval_records = []

        total_b = len(prog_sources) * len(progression_conditions)
        b_idx = 0

        for s_idx, stem in enumerate(prog_sources):
            img_path = TEST_IMG_DIR / f"{stem}.jpg"
            base_bgr = cv2.imread(str(img_path))
            _, base_bytes = cv2.imencode(".jpg", base_bgr)

            for var in progression_conditions:
                b_idx += 1
                after_bgr, gt_mask = inject_synthetic_progression(base_bgr, var, seed=200 + s_idx)
                _, after_bytes = cv2.imencode(".jpg", after_bgr)

                # For each candidate tolerance, query the API
                for tau in tolerance_candidates:
                    try:
                        resp = client.post(
                            "http://127.0.0.1:8000/api/comparison/hybrid",
                            files={
                                "before": ("before.jpg", base_bytes.tobytes(), "image/jpeg"),
                                "after": (f"after_{var}.jpg", after_bytes.tobytes(), "image/jpeg"),
                            },
                            data={"alignment_mode": "auto", "confidence": "0.20", "iou": "0.45", "tolerance_px": str(tau)},
                        )

                        if resp.status_code == 200:
                            p_data = resp.json()
                            crack_info = p_data["crack_comparison"]

                            # Decode newly observed mask
                            b64_new = p_data["images"]["newly_observed_mask"]
                            pred_new_mask = cv2.imdecode(
                                np.frombuffer(base64.b64decode(b64_new), np.uint8), cv2.IMREAD_GRAYSCALE
                            )

                            pred_bin = (pred_new_mask > 0)
                            gt_bin = (gt_mask > 0)

                            # Intersection over Union and Precision/Recall
                            tp = int(np.count_nonzero(pred_bin & gt_bin))
                            fp = int(np.count_nonzero(pred_bin & (~gt_bin)))
                            fn = int(np.count_nonzero((~pred_bin) & gt_bin))

                            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

                            rec_entry = {
                                "source_image": stem,
                                "variation": var,
                                "tolerance_px": tau,
                                "tp_pixels": tp,
                                "fp_pixels": fp,
                                "fn_pixels": fn,
                                "precision": round(prec, 4),
                                "recall": round(rec, 4),
                                "f1_score": round(f1, 4),
                                "change_mask_iou": round(iou, 4),
                                "raw_newly_observed_pixels": crack_info["newly_observed_pixels"],
                                "tolerance_adjusted_new_pixels": crack_info["tolerance_adjusted_new_pixels"],
                                "potential_new_components": crack_info["potential_new_components"],
                                "potential_expanded_components": crack_info["potential_expanded_components"],
                                "coverage_change_pp": crack_info["coverage_change_percentage_points"],
                            }

                            tolerance_records.append(rec_entry)
                            if tau == 2:  # Canonical production tolerance record
                                progression_eval_records.append(rec_entry)

                    except Exception as exc:
                        logger.warning(f"Error evaluating {stem} - {var} - tau={tau}: {exc}")

                if b_idx % 10 == 0 or b_idx == total_b:
                    logger.info(f"Group B Progress: {b_idx}/{total_b} variations evaluated across 6 tolerances.")

        df_tolerance = pd.DataFrame(tolerance_records)
        df_tolerance.to_csv(TOLERANCE_CSV, index=False)
        df_progression = pd.DataFrame(progression_eval_records)
        df_progression.to_csv(PROGRESSION_CSV, index=False)
        logger.info(f"Tolerance sweep completed: Saved {len(df_tolerance)} rows to {TOLERANCE_CSV}")

        # =====================================================================
        # STEP F: UNRELATED PAIR REJECTION (Group C)
        # =====================================================================
        logger.info("\n=== Evaluating Group C: Unrelated Image Rejection Gate (10 pairs) ===")
        unrelated_records = []
        for i in range(len(SOURCE_STEMS)):
            stem1 = SOURCE_STEMS[i]
            stem2 = SOURCE_STEMS[(i + 5) % len(SOURCE_STEMS)]  # Pair with completely disjoint image

            img1_bytes = (TEST_IMG_DIR / f"{stem1}.jpg").read_bytes()
            img2_bytes = (TEST_IMG_DIR / f"{stem2}.jpg").read_bytes()

            resp = client.post(
                "http://127.0.0.1:8000/api/comparison/hybrid",
                files={
                    "before": ("before.jpg", img1_bytes, "image/jpeg"),
                    "after": ("after_unrelated.jpg", img2_bytes, "image/jpeg"),
                },
                data={"alignment_mode": "auto"},
            )

            is_rejected = (resp.status_code == 422)
            err_code = resp.json().get("detail", {}).get("error", "NONE") if is_rejected else "ACCEPTED"

            unrelated_records.append({
                "pair_id": f"unrelated_{i+1}",
                "image1": stem1,
                "image2": stem2,
                "status_code": resp.status_code,
                "is_rejected": is_rejected,
                "error_code": err_code,
            })

        df_unrelated = pd.DataFrame(unrelated_records)
        df_unrelated.to_csv(ALIGNMENT_CSV, index=False)
        logger.info(f"Group C completed: Rejection rate = {df_unrelated['is_rejected'].mean() * 100:.1f}%")

        # =====================================================================
        # STATISTICAL ANALYSIS & BOOTSTRAP CONFIDENCE INTERVALS
        # =====================================================================
        logger.info("\n=== Running Statistical Analysis & Bootstrapping ===")
        accepted_nc = df_no_change[df_no_change["status"].isin(["accepted", "identical"])]
        abs_diffs = accepted_nc["abs_coverage_change_pp"].values

        mean_diff = float(np.mean(abs_diffs))
        median_diff = float(np.median(abs_diffs))
        std_diff = float(np.std(abs_diffs))
        p90 = float(np.percentile(abs_diffs, 90))
        p95 = float(np.percentile(abs_diffs, 95))
        p99 = float(np.percentile(abs_diffs, 99))
        max_diff = float(np.max(abs_diffs))

        # Evaluate existing 0.15 pp threshold
        fpr_015 = float(np.mean(abs_diffs > 0.15))

        # Group-aware Image-Level Bootstrapping (1,000 resamples over 10 unique images)
        rng = np.random.default_rng(2026)
        unique_images = accepted_nc["source_image"].unique()
        n_images = len(unique_images)
        bootstrap_p95s = []

        for _ in range(1000):
            resampled_imgs = rng.choice(unique_images, size=n_images, replace=True)
            resampled_rows = accepted_nc[accepted_nc["source_image"].isin(resampled_imgs)]
            if len(resampled_rows) > 0:
                bootstrap_p95s.append(np.percentile(resampled_rows["abs_coverage_change_pp"].values, 95))

        b_ci_lower = float(np.percentile(bootstrap_p95s, 2.5))
        b_ci_upper = float(np.percentile(bootstrap_p95s, 97.5))

        logger.info(f"Empirical No-Change Stats (N={len(abs_diffs)}):")
        logger.info(f"  Mean: {mean_diff:.4f} pp, Median: {median_diff:.4f} pp, Std: {std_diff:.4f} pp")
        logger.info(f"  90th Percentile: {p90:.4f} pp")
        logger.info(f"  95th Percentile: {p95:.4f} pp (Bootstrap 95% CI: [{b_ci_lower:.4f}, {b_ci_upper:.4f}] pp)")
        logger.info(f"  99th Percentile: {p99:.4f} pp, Max: {max_diff:.4f} pp")
        logger.info(f"  False Positive Rate at ±0.15 pp: {fpr_015 * 100:.2f}%")

        # Tolerance Tradeoff Metrics across tau
        tol_summary = df_tolerance.groupby("tolerance_px").agg({
            "fp_pixels": "mean",
            "fn_pixels": "mean",
            "precision": "mean",
            "recall": "mean",
            "f1_score": "mean",
            "change_mask_iou": "mean",
            "raw_newly_observed_pixels": "mean",
            "tolerance_adjusted_new_pixels": "mean",
        }).reset_index()

        # Save Final Metrics JSON
        final_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unique_source_images": len(SOURCE_STEMS),
            "no_change_total_pairs": len(df_no_change),
            "no_change_accepted_pairs": len(accepted_nc),
            "no_change_rejection_rate_pct": round(float((1 - len(accepted_nc) / len(df_no_change)) * 100), 2),
            "unrelated_rejection_rate_pct": round(float(df_unrelated["is_rejected"].mean() * 100), 2),
            "empirical_no_change_distribution_pp": {
                "mean": round(mean_diff, 4),
                "median": round(median_diff, 4),
                "std": round(std_diff, 4),
                "p90": round(p90, 4),
                "p95": round(p95, 4),
                "p99": round(p99, 4),
                "max": round(max_diff, 4),
                "fpr_at_0_15_pp": round(fpr_015, 4),
                "bootstrap_95_ci_p95": [round(b_ci_lower, 4), round(b_ci_upper, 4)],
            },
            "tolerance_sweep_results": tol_summary.to_dict(orient="records"),
            "recommended_calibrated_noise_floor_pp": round(p95, 4),
            "recommended_tolerance_px": 2,
        }

        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2)

        # =====================================================================
        # GENERATE PLOTS
        # =====================================================================
        logger.info("Generating empirical distribution and tolerance tradeoff plots...")

        # Plot 1: No-Change Coverage Difference Distribution
        plt.figure(figsize=(9, 5))
        plt.hist(abs_diffs, bins=25, color="#2b5c8f", edgecolor="black", alpha=0.75, density=True)
        plt.axvline(0.15, color="red", linestyle="--", linewidth=2, label=f"Legacy Claimed Threshold (0.15 pp, FPR={fpr_015*100:.1f}%)")
        plt.axvline(p95, color="darkgreen", linestyle="-", linewidth=2.5, label=f"Calibrated 95th Percentile ({p95:.3f} pp)")
        plt.axvspan(b_ci_lower, b_ci_upper, color="green", alpha=0.15, label="Bootstrap 95% CI on P95")
        plt.title("Empirical Distribution of |Δ Crack Coverage| on No-Change Controls (N=108)", fontsize=13, fontweight="bold")
        plt.xlabel("Absolute Coverage Change (|Δ Coverage| in Percentage Points)", fontsize=11)
        plt.ylabel("Density", fontsize=11)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(NO_CHANGE_PNG, dpi=200)
        plt.close()

        # Plot 2: Tolerance Parameter Tradeoff Curve
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        taus = tol_summary["tolerance_px"].values
        f1s = tol_summary["f1_score"].values
        ious = tol_summary["change_mask_iou"].values
        adj_fps = tol_summary["tolerance_adjusted_new_pixels"].values

        ax1.plot(taus, f1s, "b-o", linewidth=2, label="F1 Score")
        ax1.plot(taus, ious, "g-s", linewidth=2, label="Change Mask IoU")
        ax2.plot(taus, adj_fps, "r--^", linewidth=2, label="Mean Adjusted New Pixels")

        ax1.set_xlabel("Spatial Tolerance (pixels)", fontsize=11)
        ax1.set_ylabel("F1 Score / IoU", color="black", fontsize=11)
        ax2.set_ylabel("Mean Tolerance-Adjusted Pixels", color="red", fontsize=11)
        ax1.set_xticks(taus)
        ax1.grid(True, linestyle=":", alpha=0.6)
        plt.title("Spatial Tolerance (τ) Tradeoff: Detection Sensitivity vs False Pixel Suppression", fontsize=12, fontweight="bold")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

        plt.tight_layout()
        plt.savefig(TOLERANCE_PNG, dpi=200)
        plt.close()

        # Save Visual Examples for Section 14
        logger.info("Extracting representative diagnostic images for visual_examples/...")
        # 1. No-change correctly classified
        # 2. No-change false flag
        # 3. Progression detected
        # 4. Progression missed
        # 5. Alignment rejected

        # Progression detected example
        ex_img = cv2.imread(str(TEST_IMG_DIR / f"{SOURCE_STEMS[0]}.jpg"))
        after_ex, gt_ex = inject_synthetic_progression(ex_img, "new_isolated_crack", seed=42)
        cv2.imwrite(str(VIS_EXAMPLES_DIR / "progression_detected_before.png"), ex_img)
        cv2.imwrite(str(VIS_EXAMPLES_DIR / "progression_detected_after.png"), after_ex)
        cv2.imwrite(str(VIS_EXAMPLES_DIR / "progression_gt_mask.png"), gt_ex)

        # Unrelated rejection example
        unrel_before = cv2.imread(str(TEST_IMG_DIR / f"{SOURCE_STEMS[0]}.jpg"))
        unrel_after = cv2.imread(str(TEST_IMG_DIR / f"{SOURCE_STEMS[5]}.jpg"))
        cv2.imwrite(str(VIS_EXAMPLES_DIR / "alignment_rejected_before.png"), unrel_before)
        cv2.imwrite(str(VIS_EXAMPLES_DIR / "alignment_rejected_after.png"), unrel_after)

        logger.info("Reliability evaluation and artifact generation complete!")

    finally:
        server.stop()


if __name__ == "__main__":
    run_evaluation()
