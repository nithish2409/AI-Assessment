# React Frontend Integration Guide — Phase 5 Dashboard

This document provides complete instructions for running, testing, and developing the unified React dashboard for the AI Assessment Platform on Windows 11.

---

## 1. System Architecture

The AI Assessment Platform comprises three decoupled tiers designed for maximum fault isolation and hardware efficiency:

```text
┌─────────────────────────────────────────────────────────────┐
│                 React Frontend (Port 5173)                  │
│       Vite 8 · React 19 · Tailwind CSS 4 · Axios            │
└──────────────────────────────┬──────────────────────────────┘
                               │
               Async HTTP (REST / Multipart)
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               Main FastAPI Backend (Port 8000)              │
│       Python 3.14.2 — backend/venv (Task 2 & Task 3)        │
│       • POST /api/damage/segment (Gateway Proxy)             │
│       • POST /api/comparison/compare (SIFT/AKAZE/ORB + SSIM)│
│       • POST /api/comparison/hybrid (Phase 4 Progression)   │
│       • POST /api/nlp/classify (TF-IDF + Logistic Reg.)     │
│       • GET  /api/health & GET /api/damage/health            │
└──────────────────────────────┬──────────────────────────────┘
                               │
               Internal Loopback Delegation (httpx)
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          YOLO Inference Microservice (Port 8001)            │
│       Python 3.12.2 — backend/venv_yolo (Task 1 & Phase 4)   │
│       • PyTorch 2.6.0+cu124 (RTX 4060 Laptop GPU, 8 GB)     │
│       • YOLO26s-seg Crack Segmentation Model                │
│       • Isolated from browser access                         │
└─────────────────────────────────────────────────────────────┘
```

### Architectural Key Points:
1. **Isolated Environments:** The main backend runs on Python 3.14.2 for general scientific computing and web serving, while the YOLO microservice runs on Python 3.12.2 with CUDA 12.4 acceleration on the NVIDIA RTX 4060 Laptop GPU.
2. **Gateway Pattern:** The browser client communicates exclusively with the main backend (port 8000). The internal microservice (port 8001) is never called directly from the client.
3. **Resilient Degradation:** If the YOLO microservice is offline, Task 2 (Image Comparison) and Task 3 (NLP) continue functioning without disruption. Calls to crack segmentation return structured 503 errors.

---

## 2. Directory Structure

```text
frontend/
├── dist/                               # Production build output
├── node_modules/                       # Node.js dependencies
├── public/                             # Static assets
├── src/
│   ├── assets/                         # SVG icons and visual logos
│   ├── components/
│   │   ├── ErrorState.jsx              # Structured error display with guidance
│   │   ├── FeatureChart.jsx            # Recharts bar chart for NLP features
│   │   ├── ImageComparisonViewer.jsx   # 11-asset viewer, tabs, split slider, lightbox
│   │   ├── ImageUpload.jsx             # Drag-and-drop accessible file uploader
│   │   ├── Layout.jsx                  # Responsive sidebar navigation & drawer
│   │   ├── LoadingState.jsx            # Stage-aware accessible loading spinner
│   │   ├── PageHeader.jsx              # Standard title, subtitle, and badge header
│   │   ├── ReliabilityNotice.jsx       # Noise floor & defect limitation notices
│   │   ├── ResultCard.jsx              # Reusable KPI and metric cards
│   │   ├── SegmentationOverlay.jsx     # YOLO crack visualizer with layer toggle
│   │   └── SystemHealthIndicator.jsx   # Live service connectivity diagnostic pill
│   ├── pages/
│   │   ├── CrackDetection.jsx          # Task 1: Building Crack Detection
│   │   ├── DashboardOverview.jsx       # High-level architecture & task overview
│   │   ├── FeatureAnalysis.jsx         # Task 3: NLP TF-IDF & BoW feature deep-dive
│   │   ├── HybridProgression.jsx       # Phase 4: Hybrid Crack Progression Analysis
│   │   ├── ImageComparison.jsx         # Task 2: Before/After Image Comparison
│   │   ├── NewsClassification.jsx      # Task 3: AG News text categorization
│   │   ├── SystemStatus.jsx            # Live health telemetry & diagnostics
│   │   └── TextSimilarity.jsx          # Task 3: Cosine text similarity explorer
│   ├── services/
│   │   └── api.js                      # Centralized Axios client & error parsers
│   ├── App.jsx                         # React Router v7 with dynamic code splitting
│   ├── index.css                       # Tailwind CSS 4 theme and global styles
│   └── main.jsx                        # React root entry point
├── tests/
│   ├── frontend_validation.test.js     # 20 automated frontend unit scenarios
│   └── e2e_real_verification.js        # Live API end-to-end verification script
├── package.json                        # Scripts and dependencies
└── vite.config.js                      # Vite 8 configuration
```

---

## 3. Environment Variables

Create or edit `.env` in the `frontend/` directory if connecting to a non-default API gateway:

```bash
# URL of the main FastAPI gateway (default: http://localhost:8000/api)
VITE_API_URL=http://localhost:8000/api
```

---

## 4. Quick Start: Running on Windows 11

To start the complete system, open three separate PowerShell terminal windows.

### Terminal 1: Start YOLO Inference Microservice (Port 8001)
```powershell
cd C:\Users\nithi\Projects\AI-Assessment\backend
.\venv_yolo\Scripts\python.exe -m uvicorn damage_inference.app:app --host 127.0.0.1 --port 8001
```
*Expected log output:*
```text
INFO:damage_inference.service:CUDA is available. Selected device: cuda:0 (NVIDIA GeForce RTX 4060 Laptop GPU)
INFO:damage_inference.service:YOLO Crack Segmentation service successfully initialized.
INFO:Uvicorn running on http://127.0.0.1:8001
```

### Terminal 2: Start Main FastAPI Backend (Port 8000)
```powershell
cd C:\Users\nithi\Projects\AI-Assessment\backend
$env:PYTHONPATH=".;app;.."
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
*Expected log output:*
```text
INFO:Application startup complete.
INFO:Uvicorn running on http://127.0.0.1:8000
```

### Terminal 3: Start React Frontend (Port 5173)
```powershell
cd C:\Users\nithi\Projects\AI-Assessment\frontend
npm run dev
```
*Expected log output:*
```text
  VITE v8.3.8  ready in 418 ms
  ➜  Local:   http://localhost:5173/
```

Open `http://localhost:5173/` in any modern web browser (Chrome, Edge, Firefox).

---

## 5. Automated Testing & Verification

### Running Frontend Validation Tests
The frontend includes an automated unit test suite covering all 20 required validation scenarios:
```powershell
cd C:\Users\nithi\Projects\AI-Assessment\frontend
npm test
```
*Expected output:*
```text
1..20
# tests 20
# suites 0
# pass 20
# fail 0
# duration_ms 307.2
```

### Running Production Build
To verify production minification and chunk generation:
```powershell
cd C:\Users\nithi\Projects\AI-Assessment\frontend
npm run build
```
*Expected output: All 8 route chunks built with zero warnings in < 700 ms.*

### Running Live End-to-End API Verification
With both backend services running, execute:
```powershell
cd C:\Users\nithi\Projects\AI-Assessment\frontend
node tests\e2e_real_verification.js
```
*Verifies genuine crack detection, zero-crack handling, valid pair alignment, unrelated image rejection, hybrid progression, and NLP classification.*

---

## 6. Supported Workflows

### 1. Task 1: Building Crack Detection (`/crack-detection`)
- **Workflow:** Upload a building concrete photograph (JPEG, PNG, WebP up to 10 MB). Adjust confidence (0.05–0.95) and IoU (0.10–0.90) sliders. Click "Detect Cracks".
- **Visualizations:** Toggle between Crack Overlay, Binary Mask, and Original Photograph.
- **Metrics:** Total crack count, crack coverage percentage, bounding box coordinates, and inference latency breakdown.
- **Limitation:** Only the `crack` defect class is supported. Zero cracks detected does not imply structural safety.

### 2. Task 2: Before/After Image Comparison (`/image-comparison`)
- **Workflow:** Upload a before and after inspection photograph. Select feature alignment detector (`auto`, `sift`, `akaze`, `orb`, `direct`). Click "Validate Pair" to check dimensions or "Compare Images" to compute SSIM.
- **Visualizations:** SSIM Difference Heatmap, Highlighted Changes, Binary Mask, Aligned Reference, and Before/After Split Slider.
- **Metrics:** Mean SSIM score, changed area percentage, changed region count, inlier ratio, and reprojection error.
- **Error Handling:** When alignment fails, attempted detectors, failure reasons, and guidance notes are displayed.

### 3. Phase 4: Hybrid Crack Progression Analysis (`/hybrid-analysis`)
- **Workflow:** Upload baseline and subsequent inspection photographs. Set alignment mode, confidence, IoU, and spatial tolerance radius (0–10 px, default 2 px). Click "Execute Hybrid Progression".
- **Visualizations:** 11-asset inspector (Combined Hybrid Overlay, Tolerance-Adjusted Mask, Newly Observed Mask, Persistent Mask, No-Longer-Observed Mask, After Mask, Warped Before Mask, SSIM Difference, Aligned Before, and Original Photographs) with Before/After Slider.
- **Legend:**
  - 🟡 **Yellow:** Persistent predicted crack regions
  - 🔴 **Red:** Potential newly observed cracks
  - 🔵 **Blue:** No-longer-observed crack regions
  - ⬛ **Dimmed:** Outside valid registration overlap
- **Progression Logic:** Displays categorical `progression_label`, `uncertainty_level`, and filters coverage change against the provisional **0.89 pp noise floor**.

### 4. Task 3: News Classification (`/news-classification`)
- **Workflow:** Input news headlines or select from preloaded sample articles. Click "Classify Article".
- **Output:** Predicted category (World, Sports, Business, Sci/Tech), model confidence, and probability distribution across all 4 classes.
- **Deep-Dive:** Seamless navigation to Feature Analysis (`/features`) and Text Similarity (`/similarity`).

### 5. System Status (`/system-status`)
- **Workflow:** Inspect live health and latency of the Main Backend (port 8000), YOLO Microservice (port 8001 via gateway), and NLP model. Click "Refresh Status" for instantaneous diagnostics.

---

## 7. Known Limitations & Methodological Constraints

1. **Class Specificity:** YOLO26s-seg is trained exclusively on concrete cracks. Spalling, exposed rebar, honeycomb voids, and efflorescence are not recognized.
2. **Provisional Noise Floor (0.89 pp):** Derived empirically from controlled image experiments to filter segmentation and alignment shifts. It is an engineering heuristic, not an absolute guarantee against false positives.
3. **No Structural Assessment:** The platform measures predicted pixel-level coverage and component topology. It does not measure physical crack width in millimeters and cannot certify structural safety or building integrity.
4. **Extreme Viewpoint Shifts:** Homography assumes an approximately planar surface. Extreme camera viewpoint changes (> 30° perspective tilt) or complete scene occlusion will result in alignment failure.

---

## 8. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **Backend Disconnected** in sidebar | Main FastAPI server not running on port 8000 | In Terminal 2, run `uvicorn app.main:app --port 8000`. |
| **YOLO Service Offline (503)** | Inference microservice not running on port 8001 | In Terminal 1, run `uvicorn damage_inference.app:app --port 8001` under `backend/venv_yolo`. |
| **ALIGNMENT_FAILED (422)** | Photos have insufficient shared visual features or viewpoint tilt > 30° | Upload photos of the same concrete surface with visible texture. If images are already pixel-aligned, select `alignment_mode="direct"`. |
| **Image exceeds 10 MB** | File exceeds maximum upload limit | Downsample photograph resolution or compress image under 10 MB before uploading. |
| **CORS Error in Browser Console** | Frontend running on non-5173 port | Ensure Vite runs on port 5173 (configured in `vite.config.js` and FastAPI CORS middleware). |
