# Local model artifacts

Keep these paths unchanged. Model binaries are excluded from Git; JSON metadata stays with the source.

| Path relative to this folder | Purpose |
|---|---|
| news_classifier.joblib | Task 3 trained NLP pipeline |
| metrics.json | Task 3 saved evaluation metrics |
| damage/crack_segmentation_best.pt | Building crack segmentation and hybrid comparison |
| damage/industrial_defect_yolo26s.pt | Industrial defect detection |
| damage/codebrim_multilabel_best.pt | CODEBRIM multilabel damage classification |
| yolov8s-worldv2.pt | Open-vocabulary component localization |
| damage/metadata.json | Crack-model metadata |
| damage/industrial_yolo26s_metadata.json | Industrial-model metadata |

Restore binary files from your verified local originals before running the corresponding services. No public download URLs or redistribution rights are assumed. Keep compatible Python/library versions for serialized models, and load only trusted checkpoints. Additional text-encoder/OCR assets may live in local caches or weights/.

A second yolov8s-worldv2.pt exists directly under backend/. It was deliberately retained because external workflows may use that location; runtime inspection currently points to backend/models/yolov8s-worldv2.pt.
