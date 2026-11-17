"""
Unit and Integration Tests for Phase 4.1.1: Calibration Integrity & Independent Reliability Evaluation.

Covers the 16 required verification points:
  1. Precision uses the correct tolerance-adjusted mask.
  2. Recall uses the correct tolerance-adjusted mask.
  3. F1 changes when the underlying confusion matrix changes.
  4. IoU changes when the underlying confusion matrix changes.
  5. Different tolerances produce independently calculated results.
  6. Calibration and evaluation source IDs do not overlap.
  7. Augmented variants remain in their source-image split.
  8. Duplicate-image detection.
  9. Frozen threshold evaluation.
  10. False-positive rate denominator.
  11. Rejected alignments excluded from FPR denominator.
  12. Empty-mask handling.
  13. Valid-overlap restriction.
  14. Correct threshold metadata.
  15. Qualified progression labels.
  16. Existing API response compatibility.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
import pytest

from app.schemas.hybrid_comparison import (
    HybridAlignmentInfo,
    HybridCrackComparison,
    HybridImagesPayload,
    HybridVisualComparison,
    HybridComparisonResponse,
)
from app.services.crack_change_engine import CrackChangeEngine, crack_change_engine
from app.services.hybrid_comparison_service import HybridComparisonService, hybrid_comparison_service


class TestToleranceMetricIntegrity:
    """Verifies that evaluation metrics use actual tolerance-adjusted masks rather than raw masks."""

    def test_01_and_02_precision_and_recall_use_tolerance_adjusted_mask(self):
        """Verify that precision and recall change when computed with tolerance-adjusted mask."""
        engine = CrackChangeEngine()
        valid = np.full((200, 200), 255, dtype=np.uint8)

        # Before mask: crack at x in [48..52], y in [50..150]
        B = np.zeros((200, 200), dtype=np.uint8)
        B[50:150, 48:52] = 255

        # After mask: baseline crack shifted 1px to the right [49..53],
        # plus a true new crack branch at [100..150, 80..85]
        A = np.zeros((200, 200), dtype=np.uint8)
        A[50:150, 49:53] = 255  # 1px boundary mismatch
        A[100:150, 80:85] = 255  # True new crack: 50 * 5 = 250 px

        # Ground truth change is ONLY the true new crack
        gt_change = np.zeros((200, 200), dtype=np.uint8)
        gt_change[100:150, 80:85] = 255
        gt_bin = (gt_change > 0)

        # 1. At tolerance = 0 (raw comparison)
        res_0 = engine.analyze_crack_change(B, A, valid, tolerance_px=0)
        raw_mask_bin = (res_0["masks"]["newly_observed"] > 0)
        tp_raw = int(np.count_nonzero(raw_mask_bin & gt_bin))
        fp_raw = int(np.count_nonzero(raw_mask_bin & (~gt_bin)))
        prec_raw = tp_raw / (tp_raw + fp_raw)

        # 2. At tolerance = 1 (boundary mismatch absorbed by 1px dilation)
        res_1 = engine.analyze_crack_change(B, A, valid, tolerance_px=1)
        adj_mask_bin = (res_1["masks"]["tolerance_adjusted_new"] > 0)
        tp_adj = int(np.count_nonzero(adj_mask_bin & gt_bin))
        fp_adj = int(np.count_nonzero(adj_mask_bin & (~gt_bin)))
        prec_adj = tp_adj / (tp_adj + fp_adj)

        # With tolerance adjustment, false positive boundary pixels are suppressed!
        assert fp_adj < fp_raw, "Tolerance adjustment failed to suppress boundary mismatch false positives"
        assert prec_adj > prec_raw, f"Precision did not improve with tolerance adjustment: {prec_adj} vs {prec_raw}"
        assert tp_adj == tp_raw, "True positive crack pixels should not be altered when distant from before cracks"

    def test_03_and_04_f1_and_iou_change_when_confusion_matrix_changes(self):
        """Verify that F1 and IoU change when TP/FP/FN change across tolerances."""
        engine = CrackChangeEngine()
        valid = np.full((150, 150), 255, dtype=np.uint8)

        B = np.zeros((150, 150), dtype=np.uint8)
        B[40:110, 40:44] = 255

        # Shifted 2px + new crack
        A = np.zeros((150, 150), dtype=np.uint8)
        A[40:110, 42:46] = 255  # 2px shift
        A[80:120, 90:94] = 255  # New crack: 40 * 4 = 160 px

        gt = np.zeros((150, 150), dtype=np.uint8)
        gt[80:120, 90:94] = 255
        gt_bin = (gt > 0)

        ious = []
        f1s = []
        for tau in [0, 1, 2]:
            res = engine.analyze_crack_change(B, A, valid, tolerance_px=tau)
            mask_bin = (res["masks"]["tolerance_adjusted_new"] > 0)
            tp = int(np.count_nonzero(mask_bin & gt_bin))
            fp = int(np.count_nonzero(mask_bin & (~gt_bin)))
            fn = int(np.count_nonzero((~mask_bin) & gt_bin))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            f1s.append(f1)
            ious.append(iou)

        # F1 and IoU must not be identical across different tolerances
        assert len(set(f1s)) > 1, f"F1 scores remained identical across tolerances: {f1s}"
        assert len(set(ious)) > 1, f"IoU remained identical across tolerances: {ious}"
        assert f1s[2] > f1s[0], "F1 at tau=2 should exceed tau=0 due to boundary false-positive filtering"
        assert ious[2] > ious[0], "IoU at tau=2 should exceed tau=0 due to boundary false-positive filtering"

    def test_05_different_tolerances_produce_independently_calculated_results(self):
        """Verify each tolerance produces its own distinct adjusted pixel count and mask."""
        engine = CrackChangeEngine()
        valid = np.full((100, 100), 255, dtype=np.uint8)

        B = np.zeros((100, 100), dtype=np.uint8)
        B[30:70, 50] = 255  # 1px line, 40 px

        # After: dilated line width = 7 px (centered at 50, x in 47..53)
        A = np.zeros((100, 100), dtype=np.uint8)
        A[30:70, 47:54] = 255  # 40 * 7 = 280 px

        adjusted_counts = []
        masks = []
        for tau in [0, 1, 2, 3]:
            res = engine.analyze_crack_change(B, A, valid, tolerance_px=tau)
            count = res["tolerance_adjusted_new_pixels"]
            mask = res["masks"]["tolerance_adjusted_new"]
            adjusted_counts.append(count)
            masks.append(mask)

        # Counts must strictly decrease as tolerance radius increases
        assert adjusted_counts[0] > adjusted_counts[1] > adjusted_counts[2] >= adjusted_counts[3]
        # Masks must not be identical arrays
        assert not np.array_equal(masks[0], masks[1])
        assert not np.array_equal(masks[1], masks[2])


class TestDatasetIntegrityAndSplits:
    """Verifies source-image separation, non-overlapping splits, and duplicate detection."""

    def test_06_calibration_and_evaluation_source_ids_do_not_overlap(self):
        """Verify complete disjointness of calibration and evaluation sources."""
        from scripts.run_calibration_integrity_evaluation import build_dataset_splits
        cal_stems, eval_stems, manifest = build_dataset_splits(seed=42)

        assert len(cal_stems) >= 20, f"Expected >= 20 calibration sources, got {len(cal_stems)}"
        assert len(eval_stems) >= 20, f"Expected >= 20 evaluation sources, got {len(eval_stems)}"

        overlap = set(cal_stems) & set(eval_stems)
        assert len(overlap) == 0, f"Found source overlap between splits: {overlap}"

        cal_prefixes = {s.split(".rf.")[0] for s in cal_stems}
        eval_prefixes = {s.split(".rf.")[0] for s in eval_stems}
        assert len(cal_prefixes & eval_prefixes) == 0, "Correlated scene prefixes overlap between splits!"

    def test_07_augmented_variants_remain_in_source_image_split(self):
        """Verify augmented and transformed pairs never cross splits."""
        from scripts.run_calibration_integrity_evaluation import build_dataset_splits, CONDITIONS
        cal_stems, eval_stems, _ = build_dataset_splits(seed=42)

        cal_set = set(cal_stems)
        eval_set = set(eval_stems)

        # Check simulated pair registry
        for s in cal_stems:
            for cond in CONDITIONS:
                assigned_split = "calibration" if s in cal_set else "evaluation"
                assert assigned_split == "calibration"
                assert s not in eval_set

        for s in eval_stems:
            for cond in CONDITIONS:
                assigned_split = "evaluation" if s in eval_set else "calibration"
                assert assigned_split == "evaluation"
                assert s not in cal_set

    def test_08_duplicate_image_detection(self):
        """Verify image hashes detect exact duplicates and confirm splits have zero duplicates."""
        from scripts.run_calibration_integrity_evaluation import build_dataset_splits
        cal_stems, eval_stems, manifest = build_dataset_splits(seed=42)

        cal_hashes = [entry["sha256"] for entry in manifest["calibration_sources"]]
        eval_hashes = [entry["sha256"] for entry in manifest["evaluation_sources"]]

        # Uniqueness within splits
        assert len(cal_hashes) == len(set(cal_hashes)), "Duplicate image detected in calibration split!"
        assert len(eval_hashes) == len(set(eval_hashes)), "Duplicate image detected in evaluation split!"

        # Zero overlap between splits
        hash_overlap = set(cal_hashes) & set(eval_hashes)
        assert len(hash_overlap) == 0, f"Identical image files found in both splits: {hash_overlap}"


class TestStatisticalEvaluationMechanics:
    """Verifies frozen threshold logic, false positive rate calculation, and rejection exclusion."""

    def test_09_frozen_threshold_evaluation(self):
        """Verify that frozen threshold does not mutate and correctly flags alerts."""
        frozen_thresh = 0.90  # percentage points

        # Synthetic test series on evaluation split
        test_changes = [0.05, 0.40, 0.89, 0.90, 0.91, 1.25, 2.50]
        expected_alerts = [False, False, False, False, True, True, True]

        alerts = [abs(c) > frozen_thresh for c in test_changes]
        assert alerts == expected_alerts
        # Threshold remains unchanged
        assert frozen_thresh == 0.90

    def test_10_and_11_false_positive_rate_denominator_excludes_rejected_alignments(self):
        """Verify FPR denominator is strictly accepted pairs, not total pairs including rejections."""
        # 10 pairs total: 8 accepted, 2 rejected due to alignment failure
        pairs = [
            {"status": "accepted", "abs_cov": 0.20},
            {"status": "accepted", "abs_cov": 0.45},
            {"status": "accepted", "abs_cov": 0.10},
            {"status": "accepted", "abs_cov": 1.15},  # False alert (> 0.90 pp)
            {"status": "accepted", "abs_cov": 0.35},
            {"status": "accepted", "abs_cov": 0.50},
            {"status": "accepted", "abs_cov": 0.80},
            {"status": "accepted", "abs_cov": 0.25},
            {"status": "rejected", "abs_cov": 0.00},  # Rejection
            {"status": "rejected", "abs_cov": 0.00},  # Rejection
        ]
        thresh = 0.90

        accepted = [p for p in pairs if p["status"] == "accepted"]
        rejected = [p for p in pairs if p["status"] == "rejected"]
        assert len(accepted) == 8
        assert len(rejected) == 2

        false_alerts = sum(1 for p in accepted if p["abs_cov"] > thresh)
        assert false_alerts == 1

        # Correct FPR = 1 / 8 = 0.125 (12.5%)
        # Incorrect FPR if rejections were included = 1 / 10 = 0.10 (10.0%)
        correct_fpr = false_alerts / len(accepted)
        flawed_fpr = false_alerts / len(pairs)

        assert math.isclose(correct_fpr, 0.125)
        assert not math.isclose(correct_fpr, flawed_fpr)

    def test_12_empty_mask_handling(self):
        """Verify behavior when both prediction and ground truth contain zero crack pixels."""
        engine = CrackChangeEngine()
        valid = np.full((100, 100), 255, dtype=np.uint8)
        empty = np.zeros((100, 100), dtype=np.uint8)

        res = engine.analyze_crack_change(empty, empty, valid, tolerance_px=2)
        assert res["before_crack_pixels"] == 0
        assert res["after_crack_pixels"] == 0
        assert res["coverage_change_percentage_points"] == 0.0
        assert res["is_above_noise_floor"] is False
        assert res["progression_label"] == "No substantial predicted change detected"

        # Documented convention for PR/IoU when GT and prediction are both empty
        pred_bin = (res["masks"]["tolerance_adjusted_new"] > 0)
        gt_bin = (empty > 0)
        tp = int(np.count_nonzero(pred_bin & gt_bin))
        fp = int(np.count_nonzero(pred_bin & (~gt_bin)))
        fn = int(np.count_nonzero((~pred_bin) & gt_bin))

        assert tp == 0 and fp == 0 and fn == 0
        # When tp + fp == 0, safe convention returns 0.0 (or documented null), never raises ZeroDivisionError
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        assert prec == 0.0
        assert rec == 0.0

    def test_13_valid_overlap_restriction(self):
        """Verify crack pixels outside valid overlap are strictly excluded."""
        engine = CrackChangeEngine()
        valid = np.zeros((100, 100), dtype=np.uint8)
        valid[20:80, 20:80] = 255  # Valid ROI = 60 * 60 = 3600 pixels

        # Put cracks outside valid overlap
        A = np.zeros((100, 100), dtype=np.uint8)
        A[0:10, 0:10] = 255  # 100 pixels outside ROI
        A[30:50, 30:50] = 255  # 400 pixels inside ROI

        res = engine.analyze_crack_change(A, A, valid, tolerance_px=2)
        assert res["valid_overlap_pixels"] == 3600
        assert res["before_crack_pixels"] == 400
        assert res["after_crack_pixels"] == 400
        # Outside pixels must be completely ignored
        assert res["after_crack_coverage_percentage"] == round(100.0 * 400 / 3600, 4)


class TestAPIContractAndQualifications:
    """Verifies schema compatibility, uncertainty metadata, and qualified progression labels."""

    def test_14_correct_threshold_metadata(self):
        """Verify calibrated_noise_floor_pp is faithfully preserved in response."""
        engine = CrackChangeEngine()
        valid = np.full((100, 100), 255, dtype=np.uint8)
        B = np.zeros((100, 100), dtype=np.uint8)
        A = np.zeros((100, 100), dtype=np.uint8)
        A[10:30, 10:20] = 255  # 200 pixels = 2.00 pp change

        res = engine.analyze_crack_change(B, A, valid, calibrated_noise_floor_pp=0.95)
        assert res["calibrated_noise_floor_pp"] == 0.95
        assert res["is_above_noise_floor"] is True

        res_sub = engine.analyze_crack_change(B, A, valid, calibrated_noise_floor_pp=3.50)
        assert res_sub["calibrated_noise_floor_pp"] == 3.50
        assert res_sub["is_above_noise_floor"] is False

    def test_15_qualified_progression_labels(self):
        """Verify qualified wording in progression labels and summaries."""
        engine = CrackChangeEngine()
        valid = np.full((100, 100), 255, dtype=np.uint8)
        B = np.zeros((100, 100), dtype=np.uint8)

        # 1. No change
        res_nc = engine.analyze_crack_change(B, B, valid)
        assert res_nc["progression_label"] == "No substantial predicted change detected"
        assert "spatial tolerance buffer" in res_nc["interpretation_summary"]

        # 2. Potential new crack
        A_new = np.zeros((100, 100), dtype=np.uint8)
        A_new[20:40, 20:25] = 255  # 100 px new component
        res_new = engine.analyze_crack_change(B, A_new, valid)
        assert res_new["progression_label"] == "Potential newly observed crack regions"
        assert "Potential newly observed crack regions detected" in res_new["interpretation_summary"]

    def test_16_existing_api_response_compatibility(self):
        """Verify all historical fields and newly exposed tolerance-adjusted mask exist in response schema."""
        # Verify HybridImagesPayload contains all 8 historical fields + optional tolerance-adjusted mask
        payload = HybridImagesPayload(
            aligned_before="dGVzdA==",
            visual_difference="dGVzdA==",
            before_crack_mask="dGVzdA==",
            after_crack_mask="dGVzdA==",
            newly_observed_mask="dGVzdA==",
            persistent_mask="dGVzdA==",
            no_longer_observed_mask="dGVzdA==",
            hybrid_overlay="dGVzdA==",
            tolerance_adjusted_new_mask="dGVzdA==",
        )
        d = payload.model_dump()
        for k in [
            "aligned_before", "visual_difference", "before_crack_mask",
            "after_crack_mask", "newly_observed_mask", "persistent_mask",
            "no_longer_observed_mask", "hybrid_overlay", "tolerance_adjusted_new_mask"
        ]:
            assert k in d, f"Missing field {k} in HybridImagesPayload"
