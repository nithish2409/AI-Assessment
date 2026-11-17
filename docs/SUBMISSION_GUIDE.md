# GitHub Submission & Evaluator Guide

## 1. Overview

This document provides instructions for technical evaluators to clone, configure, execute, and verify the **AI-Assessment** project from GitHub.

---

## 2. Prerequisites & System Requirements

### Hardware Requirements
- **Recommended GPU**: NVIDIA RTX 3060 / 4060 or higher with $\ge 6\text{ GB}$ dedicated VRAM (evaluated on RTX 4060 Laptop GPU, 8 GB VRAM).
- **System Memory**: $\ge 16\text{ GB}$ RAM.
- **Operating System**: Windows 11 / Windows 10 (or Linux with CUDA 12+).

### Software Requirements
- **Git** with **Git LFS** installed (`git-lfs >= 3.0`).
- **Python**:
  - Python 3.12 (required for GPU Microservice with PyTorch + CUDA).
  - Python 3.14 (or Python 3.12+) for the FastAPI Gateway.
- **Node.js**: $\ge 18.0$ (tested on Node.js v24.2.0) & npm.
- **NVIDIA CUDA Toolkit**: 12.4 compatible driver.

---

## 3. Cloning the Repository (with Git LFS)

Because large model weights (notably `weights/clip/ViT-B-32.pt` at 337 MB) are tracked via Git LFS, ensure Git LFS is initialized before cloning:

```bash
# 1. Initialize Git LFS on your system (one-time setup)
git lfs install

# 2. Clone the repository
git clone https://github.com/<username>/AI-Assessment.git
cd AI-Assessment

# 3. Pull all LFS pointer binaries
git lfs pull
```

### Model Checksum Verification
To confirm that model checkpoints downloaded without corruption, verify their SHA-256 hashes:

```powershell
# Windows PowerShell
Get-FileHash backend/models/damage/crack_segmentation_best.pt, backend/models/damage/industrial_defect_yolo26s.pt, backend/models/damage/codebrim_multilabel_best.pt, backend/models/yolov8s-worldv2.pt, backend/models/news_classifier.joblib, weights/clip/ViT-B-32.pt -Algorithm SHA256 | Format-List
```

Expected Hashes:
- `crack_segmentation_best.pt`: `cffe2ec06b921dc6374a2714644dd5c907c3eeded6bdd81988cd5e7f0130ec32`
- `industrial_defect_yolo26s.pt`: `3013631424d4535b63efed04ea4d8968a7130421db7d43be1801c871dfea7685`
- `codebrim_multilabel_best.pt`: `07d0dd2546b004ec9f47b5fac1569f3ea41ad3e91d0a5c8af1e90a1379df8e80`
- `yolov8s-worldv2.pt`: `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792`
- `news_classifier.joblib`: `70c31204238c01f13f06c4efdc63ab322a857b8a6f64965f5df038443a11fc9b`
- `ViT-B-32.pt`: `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`

---

## 4. Environment Setup

### 4.1 Backend Gateway Environment (`backend/venv`)
```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..
```

### 4.2 GPU Inference Microservice Environment (`backend/venv_yolo`)
```powershell
cd backend
# Use Python 3.12 for CUDA 12.4 PyTorch wheels
py -3.12 -m venv venv_yolo
.\venv_yolo\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements_yolo.txt
deactivate
cd ..
```

### 4.3 Frontend Environment (`frontend/`)
```powershell
cd frontend
npm install
cd ..
```

---

## 5. Application Execution Instructions

To run the complete system, start the three services in separate terminal windows:

### Terminal 1 — Dedicated GPU Microservice (Port 8001)
From repository root:
```powershell
.\backend\venv_yolo\Scripts\python.exe -m uvicorn damage_inference.app:app --app-dir backend --host 127.0.0.1 --port 8001
```
*(Or from `backend/`: `cd backend; .\venv_yolo\Scripts\python.exe -m uvicorn damage_inference.app:app --host 127.0.0.1 --port 8001`)*  
*Expected log:* `INFO: InspectionService core vision models successfully initialized.`

### Terminal 2 — FastAPI Gateway (Port 8000)
From repository root:
```powershell
.\backend\venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```
*(Or from `backend/`: `cd backend; .\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`)*  
*Expected log:* `INFO: Application startup complete.`

### Terminal 3 — React / Vite Frontend (Port 5173)
```powershell
cd frontend
npm run dev
```
*Expected log:* `VITE ready in ~180 ms. Local: http://localhost:5173/`

### Application Navigation & Direct URLs
Open your web browser and navigate to:
```
http://localhost:5173/
```
Direct links to key tasks:
- **Main Dashboard**: `http://localhost:5173/`
- **Task 1: AI Visual Inspection & Severity**: `http://localhost:5173/inspection`
- **Task 2: Before/After Image Comparison**: `http://localhost:5173/image-comparison` (or `/comparison`)
- **Additional Feature: Hybrid Crack Progression**: `http://localhost:5173/hybrid-analysis` (or `/hybrid`)
- **Task 3: NLP News Classification**: `http://localhost:5173/news-classification` (or `/news`)
- **System Health Diagnostics**: `http://localhost:5173/system-status`

---

## 6. How to Run Test Suites

Evaluators can reproduce all automated test suites locally:

### 1. Backend Regression Tests (230 tests)
```powershell
cd backend
.\venv\Scripts\python.exe -m pytest tests
cd ..
```
*Expected output:* `226 passed, 4 skipped in ~12s`

### 2. GPU Microservice Tests (32 tests)
```powershell
cd backend
.\venv_yolo\Scripts\python.exe -m pytest tests/test_damage_service.py tests/test_damage_geometry.py
cd ..
```
*Expected output:* `32 passed in ~6s`

### 3. Frontend Unit Tests (30 tests)
```powershell
cd frontend
npm test
cd ..
```
*Expected output:* `tests 30, pass 30, fail 0`

### 4. Frontend Production Build
```powershell
cd frontend
npm run build
cd ..
```
*Expected output:* `✓ built in ~700ms`

---

## 7. Common Troubleshooting Steps

| Symptom | Cause | Solution |
|---|---|---|
| **`HTTP 503 Damage Service Unavailable`** | GPU microservice on port 8001 is not running or still loading models. | Ensure Terminal 1 is started and shows `InspectionService core vision models successfully initialized`. |
| **`CUDA out of memory`** | Another application is consuming GPU VRAM. | The microservice requires $\approx 2.5\text{ GB}$ VRAM. Close GPU-intensive applications or browser hardware acceleration. |
| **`Git LFS pointer file instead of actual weights`** | Repository was cloned without Git LFS. | Run `git lfs pull` in the repository root. Verify that `weights/clip/ViT-B-32.pt` is 337 MB (not 130 bytes). |
| **`PDF Report generation fails`** | ReportLab is missing in gateway environment. | Ensure `reportlab` is installed in `backend/venv` via `pip install reportlab`. |

---

## 8. Template Submission Email for Evaluators

```text
Subject: Technical Assessment Submission — AI & Computer Vision Engineer — [Candidate Name]

Dear Evaluation Team,

I have completed the AI Technical Assessment. The implementation includes all three required tasks as well as additional hybrid image comparison and progression analytics:

1. Task 1: AI Damage Detection, Multilabel Classification, Component Localization, Visual Severity Screening, and PDF/JSON Reporting.
2. Task 2: Before/After Image Registration, Homography Estimation, SSIM Analysis, and Hybrid Crack Progression Tracking.
3. Task 3: NLP News Topic Classification with TF-IDF and Logistic Regression.

Repository URL: https://github.com/<username>/AI-Assessment
Documentation:
- Setup & Execution Guide: docs/SUBMISSION_GUIDE.md
- Model Provenance & Hashes: docs/MODEL_INVENTORY.md
- Architecture & Topology: docs/ARCHITECTURE.md
- Empirical Test Evidence: docs/VERIFICATION_REPORT.md

Note: Model checkpoints are tracked via Git LFS. Please run `git lfs pull` after cloning.

All test suites (226 backend unit tests, 32 microservice tests, 30 frontend unit tests) pass cleanly.

Thank you for your consideration, and I look forward to discussing the architecture with your engineering team.

Best regards,
[Candidate Name]
```
