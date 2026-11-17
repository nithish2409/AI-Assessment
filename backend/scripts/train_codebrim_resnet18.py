"""
GPU-Only ResNet18 Multi-Label Fine-Tuning Pipeline for CODEBRIM.
Strictly requires CUDA GPU execution (NVIDIA GeForce RTX 4060 Laptop GPU).
Executes two-stage transfer learning:
  Stage A (Epochs 1-3): Head-only training (backbone frozen)
  Stage B (Epochs 4-15): Layer4 fine-tuning + head with differential learning rates
Includes validation threshold tuning, metrics logging, and checkpoint saving.
"""
import os
import gc
import json
import time
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
# Pure NumPy metrics implementation (identical to scikit-learn)
def precision_score(y_true, y_pred, average=None, zero_division=0):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if average == "micro":
        tp = np.logical_and(y_true, y_pred).sum()
        fp = np.logical_and(~y_true, y_pred).sum()
        return float(tp / (tp + fp)) if (tp + fp) > 0 else float(zero_division)
    elif average == "macro":
        scores = [precision_score(y_true[:, c], y_pred[:, c], zero_division=zero_division) for c in range(y_true.shape[1])]
        return float(np.mean(scores))
    else:
        tp = np.logical_and(y_true, y_pred).sum()
        fp = np.logical_and(~y_true, y_pred).sum()
        return float(tp / (tp + fp)) if (tp + fp) > 0 else float(zero_division)

def recall_score(y_true, y_pred, average=None, zero_division=0):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if average == "micro":
        tp = np.logical_and(y_true, y_pred).sum()
        fn = np.logical_and(y_true, ~y_pred).sum()
        return float(tp / (tp + fn)) if (tp + fn) > 0 else float(zero_division)
    elif average == "macro":
        scores = [recall_score(y_true[:, c], y_pred[:, c], zero_division=zero_division) for c in range(y_true.shape[1])]
        return float(np.mean(scores))
    else:
        tp = np.logical_and(y_true, y_pred).sum()
        fn = np.logical_and(y_true, ~y_pred).sum()
        return float(tp / (tp + fn)) if (tp + fn) > 0 else float(zero_division)

def f1_score(y_true, y_pred, average=None, zero_division=0):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if average == "micro":
        tp = np.logical_and(y_true, y_pred).sum()
        fp = np.logical_and(~y_true, y_pred).sum()
        fn = np.logical_and(y_true, ~y_pred).sum()
        denom = 2 * tp + fp + fn
        return float(2 * tp / denom) if denom > 0 else float(zero_division)
    elif average == "macro":
        scores = [f1_score(y_true[:, c], y_pred[:, c], zero_division=zero_division) for c in range(y_true.shape[1])]
        return float(np.mean(scores))
    else:
        tp = np.logical_and(y_true, y_pred).sum()
        fp = np.logical_and(~y_true, y_pred).sum()
        fn = np.logical_and(y_true, ~y_pred).sum()
        denom = 2 * tp + fp + fn
        return float(2 * tp / denom) if denom > 0 else float(zero_division)

def average_precision_score(y_true, y_scores):
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    if np.sum(y_true) == 0:
        return 0.0
    desc_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_idx]
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    recalls = tp / np.sum(y_true)
    precisions = tp / (tp + fp)
    recalls_shifted = np.concatenate(([0.0], recalls[:-1]))
    delta_r = recalls - recalls_shifted
    return float(np.sum(delta_r * precisions))

def hamming_loss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(np.mean(y_true != y_pred))

# ---------------------------------------------------------
# 1. MANDATORY CUDA INITIALIZATION & POLICY
# ---------------------------------------------------------
assert torch.cuda.is_available(), "CRITICAL: CUDA GPU unavailable. CPU execution is strictly prohibited."

DEVICE = torch.device("cuda:0")
torch.cuda.set_device(DEVICE)

print("=" * 70)
print("GPU-ONLY PYTORCH TRAINING INITIALIZATION")
print("=" * 70)
print(f"PyTorch Version:         {torch.__version__}")
print(f"CUDA Runtime Version:    {torch.version.cuda}")
print(f"Active GPU Device:       {torch.cuda.get_device_name(0)}")
print(f"Compute Capability:      {torch.cuda.get_device_capability(0)}")
print(f"Total Dedicated VRAM:    {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
print(f"Initial VRAM Allocated:  {torch.cuda.memory_allocated(DEVICE) / (1024**2):.2f} MB")
print("=" * 70)

CLASSES = [
    "Crack",
    "Spallation",
    "CorrosionStain",
    "ExposedBars",
    "Efflorescence"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0  # Safe Windows DataLoader setting
SEED = 42

# ---------------------------------------------------------
# 2. DATASET SPLITS & LEAKAGE-AWARE PARTITIONS
# ---------------------------------------------------------
def prepare_splits(base_dir: Path, output_split_dir: Path):
    output_split_dir.mkdir(parents=True, exist_ok=True)
    img_dir = base_dir / "images"
    ann_dir = base_dir / "annotations"
    
    xml_files = sorted(list(ann_dir.glob("*.xml")))
    print(f"Processing {len(xml_files)} verified XML annotation files...")
    
    records = []
    for x in xml_files:
        tree = ET.parse(x)
        root = tree.getroot()
        vec = [0] * len(CLASSES)
        for obj in root.findall("object"):
            defect = obj.find("Defect")
            if defect is not None:
                for idx, c in enumerate(CLASSES):
                    val = defect.find(c)
                    if val is not None and val.text.strip() == "1":
                        vec[idx] = 1
        idx_num = int(x.stem.split("_")[1])
        records.append({
            "stem": x.stem,
            "filename": f"{x.stem}.jpg",
            "image_path": str((img_dir / f"{x.stem}.jpg").resolve()),
            "index": idx_num,
            "labels": vec
        })
    
    # Contiguous sequence session grouping (block size 16)
    BLOCK_SIZE = 16
    for i, r in enumerate(records):
        r["group_id"] = i // BLOCK_SIZE
        
    num_groups = (len(records) + BLOCK_SIZE - 1) // BLOCK_SIZE
    rng = np.random.RandomState(SEED)
    shuffled_groups = rng.permutation(num_groups)
    
    n_train_groups = int(round(num_groups * 0.70))
    n_val_groups = int(round(num_groups * 0.15))
    
    train_g = set(shuffled_groups[:n_train_groups])
    val_g = set(shuffled_groups[n_train_groups:n_train_groups + n_val_groups])
    test_g = set(shuffled_groups[n_train_groups + n_val_groups:])
    
    assert len(train_g & val_g) == 0 and len(train_g & test_g) == 0 and len(val_g & test_g) == 0
    
    train_recs = [r for r in records if r["group_id"] in train_g]
    val_recs = [r for r in records if r["group_id"] in val_g]
    test_recs = [r for r in records if r["group_id"] in test_g]
    
    # Save split manifests
    for name, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        p = output_split_dir / f"{name}_manifest.json"
        with open(p, "w") as f:
            json.dump(recs, f, indent=2)
            
    summary = {
        "total_annotated_records": len(records),
        "train_count": len(train_recs),
        "val_count": len(val_recs),
        "test_count": len(test_recs),
        "num_groups": num_groups,
        "group_block_size": BLOCK_SIZE,
        "classes": CLASSES,
        "random_seed": SEED,
        "split_manifest_hashes": {
            "train": hashlib.sha256(open(output_split_dir / "train_manifest.json", "rb").read()).hexdigest(),
            "val": hashlib.sha256(open(output_split_dir / "val_manifest.json", "rb").read()).hexdigest(),
            "test": hashlib.sha256(open(output_split_dir / "test_manifest.json", "rb").read()).hexdigest(),
        }
    }
    with open(output_split_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Splits saved: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")
    return train_recs, val_recs, test_recs, summary

# ---------------------------------------------------------
# 3. PYTORCH DATASET DEFINITION
# ---------------------------------------------------------
class CodebrimDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform
        
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        item = self.records[idx]
        img_path = item["image_path"]
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
            
        if self.transform is not None:
            img = self.transform(img)
            
        labels = torch.tensor(item["labels"], dtype=torch.float32)
        return img, labels

# ---------------------------------------------------------
# 4. TRAINING & EVALUATION FUNCTIONS
# ---------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    count = 0
    
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Verify GPU placement
        assert images.is_cuda and labels.is_cuda, "CRITICAL: Input tensors not on CUDA!"
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            assert outputs.is_cuda, "CRITICAL: Model output not on CUDA!"
            loss = criterion(outputs, labels)
            assert loss.is_cuda, "CRITICAL: Loss tensor not on CUDA!"
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        count += images.size(0)
        
    return total_loss / count

@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_logits = []
    
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        total_loss += loss.item() * images.size(0)
        all_logits.append(outputs.detach().cpu())
        all_targets.append(labels.detach().cpu())
        
    avg_loss = total_loss / len(dataloader.dataset)
    logits = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    return avg_loss, probs, targets

def optimize_thresholds(probs, targets):
    """Find per-class optimal threshold in [0.20, 0.80] maximizing F1 on validation set."""
    best_thresholds = []
    candidates = np.linspace(0.20, 0.80, 25)
    for c in range(targets.shape[1]):
        y_true = targets[:, c]
        best_f1 = -1.0
        best_t = 0.50
        for t in candidates:
            y_pred = (probs[:, c] >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds.append(round(float(best_t), 3))
    return best_thresholds

def compute_multilabel_metrics(probs, targets, thresholds):
    preds = np.zeros_like(probs, dtype=int)
    for c, t in enumerate(thresholds):
        preds[:, c] = (probs[:, c] >= t).astype(int)
        
    micro_f1 = f1_score(targets, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    micro_p = precision_score(targets, preds, average="micro", zero_division=0)
    macro_p = precision_score(targets, preds, average="macro", zero_division=0)
    micro_r = recall_score(targets, preds, average="micro", zero_division=0)
    macro_r = recall_score(targets, preds, average="macro", zero_division=0)
    h_loss = hamming_loss(targets, preds)
    
    per_class = {}
    aps = []
    for c, name in enumerate(CLASSES):
        y_t = targets[:, c]
        y_p = preds[:, c]
        pr = probs[:, c]
        
        ap = average_precision_score(y_t, pr) if np.sum(y_t) > 0 else 0.0
        aps.append(ap)
        per_class[name] = {
            "threshold": thresholds[c],
            "precision": float(round(precision_score(y_t, y_p, zero_division=0), 4)),
            "recall": float(round(recall_score(y_t, y_p, zero_division=0), 4)),
            "f1": float(round(f1_score(y_t, y_p, zero_division=0), 4)),
            "average_precision": float(round(ap, 4)),
            "support": int(np.sum(y_t))
        }
        
    macro_map = float(round(np.mean(aps), 4))
    
    return {
        "macro_f1": float(round(macro_f1, 4)),
        "micro_f1": float(round(micro_f1, 4)),
        "macro_precision": float(round(macro_p, 4)),
        "micro_precision": float(round(micro_p, 4)),
        "macro_recall": float(round(macro_r, 4)),
        "micro_recall": float(round(micro_r, 4)),
        "macro_map": macro_map,
        "hamming_loss": float(round(h_loss, 4)),
        "per_class": per_class
    }

# ---------------------------------------------------------
# 5. MAIN TRAINING WORKFLOW
# ---------------------------------------------------------
def main():
    start_total_time = time.perf_counter()
    
    # Paths
    base_dataset_dir = Path("task1_dataset/CODEBRIM_original_images/original_dataset")
    split_dir = Path("backend/datasets/codebrim_splits")
    model_save_dir = Path("backend/models/damage")
    report_dir = Path("reports/task1_final/phase2")
    
    model_save_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Prepare splits
    train_recs, val_recs, test_recs, split_summary = prepare_splits(base_dataset_dir, split_dir)
    
    # 2. Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    train_ds = CodebrimDataset(train_recs, transform=train_transform)
    val_ds = CodebrimDataset(val_recs, transform=eval_transform)
    test_ds = CodebrimDataset(test_recs, transform=eval_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
    
    # 3. Model Architecture
    print("\nInstantiating ResNet18 with ImageNet Pretrained Weights...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASSES))
    
    # Move to GPU
    model = model.to(DEVICE)
    assert next(model.parameters()).is_cuda, "CRITICAL: Model is not on CUDA!"
    assert all(p.is_cuda for p in model.parameters()), "CRITICAL: Not all model parameters are on CUDA!"
    print("GPU Model Placement Verified: All parameters reside on cuda:0.")
    
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    scaler = torch.amp.GradScaler("cuda")
    
    # ---------------------------------------------------------
    # STAGE A: Classification Head Training (Epochs 1-3)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE A: Training Classification Head Only (Backbone Frozen, 3 Epochs)")
    print("=" * 70)
    
    # Freeze all backbone layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
        
    optimizer_stage_a = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-2)
    
    history = []
    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_thresholds = [0.5] * len(CLASSES)
    best_state_dict = None
    
    total_epochs = 15
    stage_a_epochs = 3
    patience = 4
    patience_counter = 0
    
    for epoch in range(1, total_epochs + 1):
        epoch_start = time.perf_counter()
        
        # Transition to Stage B at epoch 4
        if epoch == stage_a_epochs + 1:
            print("\n" + "=" * 70)
            print("STAGE B: Unfreezing Layer4 for Fine-Tuning with Differential Learning Rates")
            print("=" * 70)
            # Unfreeze layer4
            for param in model.layer4.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.layer4.parameters(), "lr": 1e-5},
                {"params": model.fc.parameters(), "lr": 1e-4}
            ], weight_decay=1e-2)
            active_stage = "Stage B (Layer4 + Head)"
        elif epoch <= stage_a_epochs:
            optimizer = optimizer_stage_a
            active_stage = "Stage A (Head Only)"
            
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        
        # Validation
        val_loss, val_probs, val_targets = evaluate_model(model, val_loader, criterion, DEVICE)
        
        # Tune thresholds on validation set
        epoch_thresholds = optimize_thresholds(val_probs, val_targets)
        val_metrics = compute_multilabel_metrics(val_probs, val_targets, epoch_thresholds)
        
        epoch_duration = time.perf_counter() - epoch_start
        peak_alloc = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
        peak_res = torch.cuda.max_memory_reserved(DEVICE) / (1024**2)
        current_lr = optimizer.param_groups[0]["lr"]
        
        history.append({
            "epoch": epoch,
            "stage": active_stage,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "val_macro_map": val_metrics["macro_map"],
            "learning_rate": current_lr,
            "duration_s": round(epoch_duration, 2),
            "peak_vram_alloc_mb": round(peak_alloc, 2),
            "peak_vram_res_mb": round(peak_res, 2)
        })
        
        print(f"Epoch {epoch:02d}/{total_epochs:02d} [{active_stage}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Macro F1: {val_metrics['macro_f1']:.4f} | Val Micro F1: {val_metrics['micro_f1']:.4f} | "
              f"Time: {epoch_duration:.2f}s | VRAM: {peak_alloc:.1f}MB")
        
        # Checkpoint selection based on validation macro F1
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_thresholds = epoch_thresholds
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics
            patience_counter = 0
            print(f"  >>> New Best Model at Epoch {epoch} (Val Macro F1: {best_val_macro_f1:.4f})")
        else:
            patience_counter += 1
            if epoch > stage_a_epochs and patience_counter >= patience:
                print(f"\nEarly stopping triggered at Epoch {epoch} (No improvement for {patience} epochs).")
                break
                
    total_training_duration = time.perf_counter() - start_total_time
    print(f"\nTraining completed in {total_training_duration:.2f} seconds ({total_training_duration/60:.2f} minutes).")
    
    # ---------------------------------------------------------
    # 6. SAVE BEST CHECKPOINT WITH COMPLETE METADATA
    # ---------------------------------------------------------
    best_checkpoint_path = model_save_dir / "codebrim_multilabel_best.pt"
    checkpoint_payload = {
        "model_name": "ResNet18 CODEBRIM Multi-Label Concrete Damage Classifier",
        "architecture": "ResNet18",
        "task": "multilabel_classification",
        "num_classes": len(CLASSES),
        "classes": CLASSES,
        "input_resolution": [IMAGE_SIZE, IMAGE_SIZE],
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD
        },
        "optimal_thresholds": best_thresholds,
        "best_epoch": best_epoch,
        "total_epochs_trained": len(history),
        "training_duration_seconds": round(total_training_duration, 2),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "peak_vram_mb": round(torch.cuda.max_memory_allocated(DEVICE) / (1024**2), 2)
        },
        "validation_metrics": best_val_metrics,
        "split_summary": split_summary,
        "model_state_dict": best_state_dict
    }
    
    torch.save(checkpoint_payload, best_checkpoint_path)
    ckpt_hash = hashlib.sha256(open(best_checkpoint_path, "rb").read()).hexdigest()
    print(f"\nBest Checkpoint Saved: {best_checkpoint_path}")
    print(f"Checkpoint Size:        {best_checkpoint_path.stat().st_size} bytes ({best_checkpoint_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"Checkpoint SHA256:      {ckpt_hash}")
    
    # ---------------------------------------------------------
    # 7. GPU-ONLY HELD-OUT TEST SET EVALUATION
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("GPU-ONLY HELD-OUT TEST EVALUATION (Using Frozen Validation Thresholds)")
    print("=" * 70)
    
    # Reload best weights onto GPU
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state_dict.items()})
    model = model.to(DEVICE)
    model.eval()
    
    assert next(model.parameters()).is_cuda, "CRITICAL: Evaluated model not on CUDA!"
    assert all(p.is_cuda for p in model.parameters()), "CRITICAL: Model parameters not on CUDA!"
    
    test_loss, test_probs, test_targets = evaluate_model(model, test_loader, criterion, DEVICE)
    test_metrics = compute_multilabel_metrics(test_probs, test_targets, best_thresholds)
    
    print(f"Test Set Loss:          {test_loss:.4f}")
    print(f"Test Set Macro F1:      {test_metrics['macro_f1']:.4f}")
    print(f"Test Set Micro F1:      {test_metrics['micro_f1']:.4f}")
    print(f"Test Set Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"Test Set Macro Recall:    {test_metrics['macro_recall']:.4f}")
    print(f"Test Set Macro mAP:     {test_metrics['macro_map']:.4f}")
    print(f"Test Set Hamming Loss:  {test_metrics['hamming_loss']:.4f}")
    
    print("\nPer-Class Held-Out Test Evaluation:")
    per_class_rows = []
    for c_name in CLASSES:
        m = test_metrics["per_class"][c_name]
        print(f"  {c_name:<16} | Thresh={m['threshold']:.3f} | Prec={m['precision']:.4f} | "
              f"Rec={m['recall']:.4f} | F1={m['f1']:.4f} | AP={m['average_precision']:.4f} | Support={m['support']}")
        per_class_rows.append({
            "class_name": c_name,
            "threshold": m["threshold"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1"],
            "average_precision": m["average_precision"],
            "support": m["support"]
        })
        
    # Save per-class metrics CSV
    pd.DataFrame(per_class_rows).to_csv(report_dir / "per_class_metrics.csv", index=False)
    
    # Save validation & test JSON
    with open(report_dir / "validation_metrics.json", "w") as f:
        json.dump(best_val_metrics, f, indent=2)
    with open(report_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
        
    # Save training history CSV
    df_history = pd.DataFrame(history)
    df_history.to_csv(report_dir / "training_history.csv", index=False)
    
    # ---------------------------------------------------------
    # 8. GENERATE TRAINING & VALIDATION CURVES PLOT
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(df_history["epoch"], df_history["train_loss"], label="Train Loss (BCE)", marker="o", color="#2563EB")
    ax1.plot(df_history["epoch"], df_history["val_loss"], label="Val Loss (BCE)", marker="s", color="#DC2626")
    ax1.axvline(x=stage_a_epochs + 0.5, color="gray", linestyle="--", label="Stage B Start (Layer4 Unfrozen)")
    ax1.scatter([best_epoch], [df_history.loc[df_history['epoch'] == best_epoch, 'val_loss'].values[0]],
                color="#059669", s=100, zorder=5, label=f"Best Model (Epoch {best_epoch})")
    ax1.set_title("ResNet18 Fine-Tuning Loss Curves", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()
    
    # F1 curves
    ax2.plot(df_history["epoch"], df_history["val_macro_f1"], label="Val Macro F1", marker="^", color="#7C3AED")
    ax2.plot(df_history["epoch"], df_history["val_micro_f1"], label="Val Micro F1", marker="d", color="#D97706")
    ax2.plot(df_history["epoch"], df_history["val_macro_map"], label="Val Macro mAP", marker="x", color="#059669")
    ax2.axvline(x=stage_a_epochs + 0.5, color="gray", linestyle="--", label="Stage B Start")
    ax2.scatter([best_epoch], [best_val_macro_f1], color="#DC2626", s=100, zorder=5, label=f"Best Macro F1 ({best_val_macro_f1:.4f})")
    ax2.set_title("ResNet18 Validation Performance Metrics", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    curves_path = report_dir / "training_curves.png"
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Training curves saved to: {curves_path}")
    
    # ---------------------------------------------------------
    # 9. SAMPLE PREDICTIONS EXPORT
    # ---------------------------------------------------------
    sample_preds = []
    for i in range(min(10, len(test_recs))):
        rec = test_recs[i]
        pr = test_probs[i]
        t = test_targets[i]
        pred_bin = [(pr[c] >= best_thresholds[c]) for c in range(len(CLASSES))]
        
        gt_classes = [CLASSES[c] for c in range(len(CLASSES)) if t[c] == 1]
        pd_classes = [CLASSES[c] for c in range(len(CLASSES)) if pred_bin[c]]
        
        sample_preds.append({
            "image": rec["filename"],
            "ground_truth": gt_classes if gt_classes else ["None (Clean Background)"],
            "predicted": pd_classes if pd_classes else ["None (Clean Background)"],
            "probabilities": {CLASSES[c]: float(round(pr[c], 4)) for c in range(len(CLASSES))},
            "correct": set(gt_classes) == set(pd_classes)
        })
        
    with open(report_dir / "sample_predictions.json", "w") as f:
        json.dump(sample_preds, f, indent=2)
        
    print(f"Sample predictions exported to: {report_dir / 'sample_predictions.json'}")
    print("\nPhase 2 GPU Fine-Tuning Execution Finished Successfully!")

if __name__ == "__main__":
    main()
