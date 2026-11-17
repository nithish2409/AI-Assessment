# Setup and verification

## Existing working installation

Keep the two Python environments separate. The inspected installation uses Python 3.14.2 for backend/venv and Python 3.12.2 for backend/venv_yolo. The YOLO requirements target PyTorch 2.6.0 with CUDA 12.4. Existing Python dependency pins and frontend lockfile have been preserved.

Start all services from the project root as shown in the root README. In particular, the vision inspection service uses project-root-relative model paths; launching it from backend/ can miss models.

## Setting up another machine

1. Install suitable Python versions, Node.js/npm compatible with the locked frontend dependencies, and the GPU driver/runtime required by the pinned CUDA build.
2. Create backend/venv and backend/venv_yolo using the corresponding Python versions.
3. Install the existing requirements into their respective environments:

```powershell
.\backend\venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\backend\venv_yolo\Scripts\python.exe -m pip install -r backend/requirements_yolo.txt
```

The inspected requirement lists omit two installed runtime packages. For PDF reports and OCR, respectively, install the versions observed in the working environments:

```powershell
.\backend\venv\Scripts\python.exe -m pip install reportlab==5.1.0
.\backend\venv_yolo\Scripts\python.exe -m pip install easyocr==1.7.2
```

These are documented observations, not a fresh-environment installation guarantee. YOLO-World may require its text-encoder dependencies and cached assets; EasyOCR can download recognition assets on first use. Retain the working environment until a separate fresh setup has been verified. Do not merge environments or overwrite their dependency pins.

4. Restore the model files listed in backend/models/README.md from your own verified copies.
5. In frontend/, run npm ci. Keep package-lock.json.
6. Start the services using the root README.

## Configuration

frontend/.env.example documents VITE_API_URL. Copy it to frontend/.env.local if needed; Vite reads it at startup/build time. Defaults already target http://localhost:8000/api. Never put secrets into VITE_ variables: they are included in browser code.

The main API defaults to the GPU service at http://127.0.0.1:8001 through DAMAGE_INFERENCE_URL. Backend environment variables are read from the process environment; no automatic .env loading is assumed. For example, set $env:DAMAGE_INFERENCE_URL in the same terminal before starting the API.

Default CORS origins are http://localhost:5173 and http://127.0.0.1:5173. Using another frontend origin requires an intentional backend configuration change.

## Verification

```powershell
# From frontend/
npm test
npm run build

# From the project root: representative backend regression suite
.\backend\venv\Scripts\python.exe -m pytest backend/tests/test_nlp.py backend/tests/test_alignment.py backend/tests/test_comparison.py backend/tests/test_direct_alignment_telemetry.py backend/tests/test_inspection_api.py backend/tests/test_visual_severity_engine.py -q -p no:cacheprovider -o pythonpath=backend
```

Backend tests require the corresponding installed packages and, for NLP tests, the local classifier artifact. GPU/live-service tests have additional requirements. A successful frontend build does not validate GPU model loading. Avoid running training/evaluation utilities just to start the app: some write datasets, reports, or checkpoints.
