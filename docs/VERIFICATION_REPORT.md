# Empirical Verification & Quality Assurance Report

## 1. Executive Summary

This report documents the verification results across all unit, integration, microservice, frontend, and end-to-end regression test suites.

All tests were executed against the cleaned submission copy on Windows 11 with an **NVIDIA GeForce RTX 4060 Laptop GPU** (8 GB dedicated VRAM, Driver 572.70, CUDA 12.4).

---

## 2. Test Suite Execution Telemetry

| Test Suite | Environment | Scope / Test Count | Passed | Skipped | Failed | Execution Time |
|---|---|---|---|---|---|---|
| **Backend Gateway Suite** | `backend/venv` (Python 3.14) | Alignment, Geometry, Severity Engine, Comparison API, Damage API, NLP API, Inspection API | **226** | **4** | **0** | **12.16 s** |
| **GPU Microservice Suite** | `backend/venv_yolo` (Python 3.12, CUDA) | YOLO Damage Service, Polygon Topology, Mask Geometry | **32** | **0** | **0** | **6.16 s** |
| **Frontend Unit Test Suite**| Node.js v24.2.0 (`npm test`) | API error parsing, State management, Progress indicators, Metric formatting, Accessibility | **30** | **0** | **0** | **0.39 s** |
| **Frontend Production Build**| Vite v8.3.8 (`npm run build`) | Complete single-page application bundle compilation | **2,543 modules** | N/A | **0 errors** | **0.73 s** |
| **End-to-End Real Image GPU Test** | Physical RTX 4060 GPU | Building & Industrial pipelines on real test photos | **2 / 2** | **0** | **0** | **10.58 s** |

---

## 3. End-to-End Inference Verification on Real Images

The multi-model pipelines were verified using real test photographs from the benchmark datasets:

### Building Structural Inspection Pipeline
- **Test Image**: `image_0000001.jpg` (Resolution: $1712\times 2572$ px, 4.4 Megapixels).
- **GPU Execution Time**: **3,508.48 ms**.
- **Model Detections**:
  - ResNet18: Classified `Efflorescence` ($p = 0.7664$, threshold $\tau = 0.3750$). Other classes evaluated: `Crack` ($0.4662$), `Spallation` ($0.0441$), `CorrosionStain` ($0.2868$), `ExposedBars` ($0.1147$).
  - YOLO26s-seg: 0 crack instances detected ($0.0000\%$ relative coverage).
  - YOLO-Worldv2: Localized structural components: `wall` (0.35 confidence), `door` (0.27 confidence).
- **Severity Evaluation**: **LOW** (`monitor`).
- **Explainable Rationale**:
  - *"Isolated calcite/mineral efflorescence staining detected without cracking or spallation."*
  - *"Indicates water transport or past moisture exposure without direct mechanical section loss."*
- **Report Generation**: Rendered **929,386 byte** ReportLab PDF document (`sample_building_report.pdf`, valid `%PDF-1.4`).

### Industrial Equipment Inspection Pipeline
- **Test Image**: `corr_test_000000.jpg` (Resolution: $640\times 640$ px).
- **GPU Execution Time**: **7,076.99 ms** (including full-frame GPU EasyOCR inference).
- **Model Detections**:
  - YOLO26s: Localized 4 surface corrosion patches (confidences: $0.69$, $0.67$, $0.56$, $0.36$; relative defect area: $0.0698\%$).
  - YOLO-Worldv2: Equipment components below threshold on close-up cropped test coupon.
  - EasyOCR: 0 text snippets detected (surface coupon devoid of text).
- **Severity Evaluation**: **LOW** (`monitor`).
- **Explainable Rationale**:
  - *"Minor superficial corrosion spotting (0.07%) detected."*
  - *"Isolated oxidation with limited surface footprint; schedule for routine monitoring."*
- **Report Generation**: Rendered **148,407 byte** ReportLab PDF document (`sample_industrial_report.pdf`, valid `%PDF-1.4`).

---

## 4. Model Checkpoint Verification Summary

All model files in the submission copy were verified bit-for-bit against their recorded SHA-256 hashes:

```
[PASS] Crack segmentation    : cffe2ec06b921dc6374a2714644dd5c907c3eeded6bdd81988cd5e7f0130ec32
[PASS] Industrial defect     : 3013631424d4535b63efed04ea4d8968a7130421db7d43be1801c871dfea7685
[PASS] CODEBRIM classifier   : 07d0dd2546b004ec9f47b5fac1569f3ea41ad3e91d0a5c8af1e90a1379df8e80
[PASS] YOLO-Worldv2          : 9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792
[PASS] NLP classifier        : 70c31204238c01f13f06c4efdc63ab322a857b8a6f64965f5df038443a11fc9b
[PASS] CLIP text encoder     : 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
```

---

## 5. Security & Secret Exposure Audit

A recursive automated scan was performed across all text files in the submission copy:
- **API Keys / Secrets Scan**: **0 hardcoded credentials or keys found.**
- **Private Environment Files**: Zero `.env` files with secret values committed.
- **Personal Paths**: Zero hardcoded local Windows system paths in code files.
- **Redistribution Rights**: All models deployed are open-source or candidate-trained weights for this assessment.
