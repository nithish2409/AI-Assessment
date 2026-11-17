# YOLO Crack Segmentation — FastAPI Integration Guide

This guide provides instructions for deploying and running the YOLO26s-seg crack segmentation microservice alongside the main FastAPI application.

---

## 1. Architecture Overview

The system utilizes an isolated microservice architecture to satisfy the dual Python environment requirements:

```text
React Frontend (Port 5173)
      │
      ▼
Main FastAPI Backend (Port 8000)
Python 3.14.2 — backend/venv
      │
      │  POST /api/damage/segment
      │  (Image validation, 10MB limit, 20MP limit)
      ▼
Damage API Router (`app/api/damage.py`)
      │
      │  Async HTTP Delegation (httpx)
      │  http://127.0.0.1:8001
      ▼
YOLO Inference Microservice (Port 8001)
Python 3.12.2 — backend/venv_yolo
      │
      │  CUDA:0 (RTX 4060 Laptop GPU, 8 GB VRAM)
      ▼
YOLO26s-seg Checkpoint (`models/damage/crack_segmentation_best.pt`)
      │
      │  Segmentation Masks, Visual Overlays, Area Union %
      ▼
Main FastAPI API Response (JSON + Base64 PNGs)
```

### Why Two Python Environments?
- **Main Backend (`backend/venv`):** Runs on Python 3.14.2 with FastAPI, OpenCV, scikit-image, and scikit-learn powering Task 2 (Image Alignment) and Task 3 (NLP News Classification). PyTorch with CUDA 12.4 does not have stable prebuilt binary wheel support on Python 3.14 on Windows.
- **Inference Microservice (`backend/venv_yolo`):** Runs on Python 3.12.2 with PyTorch 2.6.0+cu124, TorchVision 0.21.0+cu124, and Ultralytics 8.4.214, leveraging the NVIDIA GeForce RTX 4060 Laptop GPU for accelerated inference.

---

## 2. Starting the Services

Both services run locally on Windows using PowerShell. Open two separate terminal windows.

### Terminal 1: Start YOLO Inference Microservice (Port 8001)
```powershell
cd C:\Users\nithi\Projects\AI-Assessment\backend
.\venv_yolo\Scripts\python.exe -m uvicorn damage_inference.app:app --host 127.0.0.1 --port 8001
```
*Expected log output:*
```text
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:damage_inference.service:Initializing YOLO segmentation model from: ...\crack_segmentation_best.pt
INFO:damage_inference.service:CUDA is available. Selected device: cuda:0 (NVIDIA GeForce RTX 4060 Laptop GPU)
INFO:damage_inference.service:Executing warmup forward pass...
INFO:damage_inference.service:YOLO Crack Segmentation service successfully initialized.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8001
```

### Terminal 2: Start Main FastAPI Backend (Port 8000)
```powershell
cd C:\Users\nithi\Projects\AI-Assessment\backend
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
*Expected log output:*
```text
INFO:     Started server process [PID]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

## 3. Stopping the Services

To stop either service gracefully:
- Press `Ctrl + C` in the respective PowerShell terminal.
- Or terminate via PowerShell command:
  ```powershell
  # Find and terminate processes by port
  Get-Process -Id (Get-NetTCPConnection -LocalPort 8001).OwningProcess | Stop-Process -Force
  Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
  ```

---

## 4. API Specification: `POST /api/damage/segment`

### Request Format
- **URL:** `http://127.0.0.1:8000/api/damage/segment`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`

### Form Fields
| Field Name | Type | Required | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `image` | Binary File | **Yes** | — | Building photograph (JPEG, PNG, or WebP; $\le 10$ MB, $\le 20$ MP) |
| `confidence` | Float | No | `0.25` | Detection confidence threshold in $(0, 1]$ |
| `iou` | Float | No | `0.45` | NMS IoU overlap threshold in $(0, 1]$ |

### PowerShell Example Request
```powershell
$imgPath = "C:\Users\nithi\Projects\AI-Assessment\backend\datasets\damage\processed\images\test\1616.rf.c868709931a671796794fdbb95352c5a.jpg"
$form = @{
    image = Get-Item -Path $imgPath
    confidence = "0.25"
    iou = "0.45"
}
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/damage/segment" -Method Post -Form $form
$response | ConvertTo-Json -Depth 5
```

### Example JSON Response
```json
{
  "status": "completed",
  "model": "YOLO26s-seg",
  "class_names": ["crack"],
  "image_width": 416,
  "image_height": 416,
  "detections_count": 3,
  "crack_coverage_percentage": 4.0594,
  "detections": [
    {
      "class_id": 0,
      "class_name": "crack",
      "confidence": 0.5996,
      "bbox": [0, 205, 403, 413],
      "area_pixels": 6945,
      "polygon": [[0.0, 205.0], [403.0, 205.0], [403.0, 413.0], [0.0, 413.0]]
    },
    {
      "class_id": 0,
      "class_name": "crack",
      "confidence": 0.3991,
      "bbox": [0, 214, 68, 250],
      "area_pixels": 847,
      "polygon": [[0.0, 214.0], [68.0, 214.0], [68.0, 250.0], [0.0, 250.0]]
    },
    {
      "class_id": 0,
      "class_name": "crack",
      "confidence": 0.2745,
      "bbox": [61, 225, 392, 271],
      "area_pixels": 4251,
      "polygon": [[61.0, 225.0], [392.0, 225.0], [392.0, 271.0], [61.0, 271.0]]
    }
  ],
  "images": {
    "original": "iVBORw0KGgoAAAANSUhEUgAA...",
    "mask": "iVBORw0KGgoAAAANSUhEUgAA...",
    "overlay": "iVBORw0KGgoAAAANSUhEUgAA..."
  },
  "inference": {
    "device": "cuda:0",
    "duration_ms": 28.5,
    "preprocess_ms": 6.2,
    "inference_ms": 13.8,
    "postprocess_ms": 8.5
  },
  "warnings": []
}
```

---

## 5. Health & Diagnostic Endpoints

### 1. Main Backend Damage Status
- **Endpoint:** `GET http://127.0.0.1:8000/api/damage/health`
- **Output:** Reports whether the inference microservice is available, the configured URL, and model readiness status.

### 2. Internal Microservice Health & Readiness
- **Endpoint:** `GET http://127.0.0.1:8001/health` (Liveness)
- **Endpoint:** `GET http://127.0.0.1:8001/ready` (Model readiness probe)

---

## 6. Hardware Acceleration & CPU Fallback

- **GPU Acceleration (Default):** The microservice queries `torch.cuda.is_available()`. On the development system, it automatically binds to `cuda:0` (NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM).
- **Concurrency Protection:** A semaphore bounds concurrent GPU forward passes to 2 simultaneous requests to protect 8 GB VRAM from out-of-memory spikes.
- **CPU Fallback:** If CUDA is unavailable (e.g. running on non-GPU host), the service automatically falls back to CPU execution, logs a warning, and returns `"device": "cpu"` with an advisory note in the response `"warnings"` array.

---

## 7. Fault Isolation & Troubleshooting

### Scenario: Inference Microservice is Down
- The main FastAPI application remains **100% operational**.
- Task 3 (NLP News Classification) and Task 2 (Image Alignment & Comparison) continue functioning normally.
- Requests to `POST /api/damage/segment` return a structured HTTP 503 error:
  ```json
  {
    "detail": {
      "error": "DAMAGE_SERVICE_UNAVAILABLE",
      "message": "The damage segmentation inference microservice is currently unreachable.",
      "guidance": "Ensure the YOLO inference microservice is running under backend/venv_yolo on http://127.0.0.1:8001."
    }
  }
  ```

### Common Troubleshooting Steps
1. **`Connection refused` on port 8001:**
   - Confirm Terminal 1 has the microservice running (`.\venv_yolo\Scripts\python.exe -m uvicorn damage_inference.app:app --port 8001`).
2. **`CUDA Out of Memory`:**
   - The microservice automatically runs `torch.cuda.empty_cache()` upon catching `OutOfMemoryError`. Ensure no other intensive GPU workloads (e.g. full model training) are running concurrently.
3. **`422 Unprocessable Entity - Invalid image upload`:**
   - Verify the file is JPEG, PNG, or WebP. Verify image size is under 10 MB and total pixels are under 20 Megapixels.
