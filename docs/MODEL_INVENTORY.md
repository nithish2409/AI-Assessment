# AI Model Inventory & Provenance Specification

This document provides a comprehensive technical specification of all AI, computer vision, deep learning, and natural language processing models deployed in this application.

---

## 1. Summary of Model Checkpoints

| # | Model Identifier | Architecture | Checkpoint File Path | File Size | SHA-256 Hash | Target Task | Primary Execution Device |
|---|---|---|---|---|---|---|---|
| **1** | Crack Segmentation | YOLO26s-seg | `backend/models/damage/crack_segmentation_best.pt` | 22.26 MB | `cffe2ec06b921dc6374a2714644dd5c907c3eeded6bdd81988cd5e7f0130ec32` | Task 1 & Task 2 | CUDA (`cuda:0`) |
| **2** | Industrial Defect | YOLO26s | `backend/models/damage/industrial_defect_yolo26s.pt` | 19.37 MB | `3013631424d4535b63efed04ea4d8968a7130421db7d43be1801c871dfea7685` | Task 1 | CUDA (`cuda:0`) |
| **3** | Multilabel Concrete Classifier | ResNet18 | `backend/models/damage/codebrim_multilabel_best.pt` | 42.72 MB | `07d0dd2546b004ec9f47b5fac1569f3ea41ad3e91d0a5c8af1e90a1379df8e80` | Task 1 | CUDA (`cuda:0`) |
| **4** | Open-Vocabulary Component Detector | YOLOv8s-worldv2 | `backend/models/yolov8s-worldv2.pt` | 24.72 MB | `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792` | Task 1 | CUDA (`cuda:0`) |
| **5** | Topic Classifier | TF-IDF + LogReg | `backend/models/news_classifier.joblib` | 1.35 MB | `70c31204238c01f13f06c4efdc63ab322a857b8a6f64965f5df038443a11fc9b` | Task 3 | CPU |
| **6** | Multimodal Text Encoder | CLIP ViT-B/32 | `weights/clip/ViT-B-32.pt` | 337.58 MB | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` | Task 1 (YOLO-World) | CUDA / CPU |
| **7** | Text Detection & Recognition | CRAFT + BiLSTM-CTC | Managed in `~/.EasyOCR/model/` | ~85 MB | Official PyTorch Weights | Task 1 (Industrial) | CUDA (`cuda:0`) |

---

## 2. Detailed Technical Profiles

### Model 1 — Concrete Crack Segmentation (YOLO26s-seg)
- **Architecture**: Ultralytics YOLO26s-seg instance segmentation head with dual detection/mask output branches.
- **Checkpoint**: `backend/models/damage/crack_segmentation_best.pt`
- **Supported Class**: `crack` (class index 0).
- **Purpose**: Detects concrete cracks and produces pixel-level polygon segmentation masks. Used in Task 1 for building surface crack coverage area calculation and in Task 2 for hybrid crack progression analysis.
- **Input Expectation**: RGB image, normalized to $640\times 640$ px with letterbox padding.
- **Output**: Bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$, confidence scores $[0.0, 1.0]$, and binary mask contours $[[x_1, y_1], [x_2, y_2], \dots]$.
- **Inference Latency**: $\approx 12.5\text{ ms}$ on NVIDIA RTX 4060 Laptop GPU.
- **Known Limitations**: Optimized for concrete surface fractures; may yield false positives on dark asphalt pavement seams, expansion joints, or deep shadow crevices.

### Model 2 — Industrial Equipment Defect Detector (YOLO26s)
- **Architecture**: Ultralytics YOLO26s object detector with anchor-free decoupled head.
- **Checkpoint**: `backend/models/damage/industrial_defect_yolo26s.pt`
- **Supported Classes**:
  1. `corrosion` (surface oxidation, rust pitting, flaking paint)
  2. `metal_crack` (structural steel fractures, fatigue lines)
  3. `mechanical_slippage` (bolt displacement, misalignment marks)
- **Purpose**: Localizes surface degradation and mechanical defects on industrial machinery, pipelines, structural steel, and pressure vessels.
- **Input Expectation**: RGB image, resized to $640\times 640$ px.
- **Output**: Multi-class bounding boxes with class IDs and confidence values.
- **Inference Latency**: $\approx 9.8\text{ ms}$ on NVIDIA RTX 4060 Laptop GPU.
- **Known Limitations**: 2D RGB bounding boxes cannot measure pitting depth or structural wall thinning; visual screening only.

### Model 3 — Concrete Multi-Label Defect Classifier (ResNet18)
- **Architecture**: Deep residual network (ResNet18) with a 5-unit Sigmoid multi-label output layer replacing the standard single-label Softmax head.
- **Checkpoint**: `backend/models/damage/codebrim_multilabel_best.pt`
- **Supported Classes & Calibrated Thresholds**:
  1. `Crack` ($\tau = 0.475$)
  2. `Spallation` ($\tau = 0.375$)
  3. `CorrosionStain` ($\tau = 0.375$)
  4. `ExposedBars` ($\tau = 0.325$)
  5. `Efflorescence` ($\tau = 0.375$)
- **Purpose**: Comprehensive image-level classification of co-occurring concrete damage mechanisms based on the CODEBRIM dataset taxonomy.
- **Input Expectation**: RGB image, resized and normalized to $224\times 224$ px using ImageNet mean/standard deviation.
- **Output**: Independent sigmoid probabilities $[p_1, \dots, p_5]$ and binary active class predictions.
- **Performance**: Mean Average Precision (mAP) of $0.7042$, Exact Subset Accuracy of $0.7407$, Macro F1 of $0.6698$ on held-out test split.
- **Known Limitations**: Image-level classification only (`bbox=None`). Does not provide spatial coordinates for where spallation or efflorescence resides on the surface.

### Model 4 — Open-Vocabulary Component Detector (YOLOv8s-worldv2)
- **Architecture**: Vision-Language Object Detection model combining a YOLO backbone with a frozen CLIP visual/textual embedding head.
- **Checkpoint**: `backend/models/yolov8s-worldv2.pt`
- **Supported Classes**: Dynamic zero-shot prompting:
  - *Building*: `["beam", "column", "wall", "slab", "ceiling", "floor", "window", "door"]`
  - *Industrial*: `["pipe", "flange", "valve", "tank", "motor", "pump", "gauge", "panel"]`
- **Purpose**: Locates structural building components and industrial equipment assets to enable spatial defect-to-component association (e.g. mapping a crack to a "beam" or corrosion to a "pipe").
- **Input Expectation**: RGB image ($640\times 640$ px) + text prompt strings.
- **Output**: Bounding boxes tagged with natural language prompt labels.
- **Inference Latency**: $\approx 14.4\text{ ms}$ on GPU; prompt embedding projection takes $<2.5\text{ ms}$.

### Model 5 — Multimodal Text Encoder (CLIP ViT-B/32)
- **Architecture**: OpenAI CLIP Vision Transformer (ViT-B/32) textual encoder.
- **Checkpoint**: `weights/clip/ViT-B-32.pt` (337.58 MB)
- **Role in Application**: Ultralytics YOLO-World utilizes OpenAI's CLIP text encoder to project candidate component class names into multimodal embedding space.
- **Offline Operation Necessity**: Ultralytics hardcodes the download destination for CLIP weights to `WEIGHTS_DIR / "clip"`. By preserving `weights/clip/ViT-B-32.pt` locally, the application can initialize and re-project open-vocabulary prompts completely offline without making outbound HTTP calls to OpenAI or Hugging Face.

### Model 6 — Text Detection & Recognition (EasyOCR)
- **Architecture**: CRAFT (Character Region Awareness for Text Detection) + Deep BiLSTM with CTC loss for sequence decoding.
- **Location**: Automatically cached in `~/.EasyOCR/model/`.
- **Purpose**: Extracts equipment identification numbers, pressure ratings, manufacturing dates, and caution tags from industrial nameplates.
- **Known Limitations**: Dependent on surface legibility, lighting angle, and contrast. Unreadable on heavily corroded or blurred plates.

### Model 7 — News Topic Classifier (TF-IDF + Logistic Regression)
- **Architecture**: Scikit-Learn Pipeline (`TfidfVectorizer(max_features=10000, ngram_range=(1,2))` + `LogisticRegression(C=1.0, max_iter=1000)`).
- **Checkpoint**: `backend/models/news_classifier.joblib` (1.35 MB)
- **Supported Classes**: `World` (0), `Sports` (1), `Business` (2), `Sci/Tech` (3).
- **Performance**:
  - Test Accuracy: **93.25%**
  - Macro Precision: **0.9327**
  - Macro Recall: **0.9325**
  - Macro F1: **0.9324**
- **Input Expectation**: Raw news article headline, summary, or full-text string.
- **Output**: Predicted category, full posterior probability distribution across all 4 classes, and cleaned token statistics.
- **Inference Latency**: $<1.2\text{ ms}$ on CPU.
