# GitHub preparation

## Included source

Publish backend/app, backend/damage_inference, configs, scripts, tests, documentation, model metadata, frontend source/static assets, package manifests, and package-lock.json. The root .gitignore excludes local artifacts; it does not delete them.

## Preserved locally, excluded from source publication

- backend/venv, backend/venv_yolo, and frontend/node_modules: installed environments/dependencies.
- datasets/, task1_dataset/, backend/datasets/, backend/data/: raw/processed datasets, manifests, and local training data.
- backend/models model binaries, backend/yolov8s-worldv2.pt, weights/, and training checkpoints: restore separately using verified provenance.
- backend/runs/, reports/, and backups/: training outputs, evaluation evidence, exported reports, and historical backups.
- Environment files, common credential filenames, private-key files, caches, logs, and frontend build output.

Do not delete these local directories merely because they are ignored. Scripts and historical reports may refer to them. Runtime model locations remain unchanged.

## Cleanup performed

Loose root task/integration notes were moved to docs/reference. The old folder structure was preserved there as a historical snapshot; the root README shows the current source layout. The generic Vite README was replaced with project-specific frontend guidance. Disposable Python/pytest caches and generated frontend dist output were removed after verification. The generated PDF smoke output was moved under reports/local-smoke/.

Application source, tests, model metadata/checkpoints, original dependency pins, and training inputs were preserved. No Git commands, repository initialization, commits, or publication were performed.

## Publication review

A targeted scan of application code, scripts, tests, and configs found no matches for common private-key, AWS access-key, GitHub-token, or OpenAI-token formats. This is not a complete secret audit; inspect documentation and any files you choose to add before publishing.

No license was invented. Confirm rights and licensing for your code, pretrained models, and datasets. Model download/release links are not available in this project yet, so a source-only clone requires separately supplied artifacts. Store large permitted artifacts in a release or external storage with checksums; do not force-add ignored datasets or checkpoints.

Git ignore rules apply to untracked files. If files have already been committed in another checkout, adding .gitignore will not remove them from existing history. Review the exact staged file list before your later publication.

## Cleanup verification (17 November 2026)

- Frontend Node test suite: 30 passed.
- Frontend production build: passed; generated dist/ removed after verification.
- Representative backend regression suite documented in SETUP.md: 96 passed, with three existing HTTP 422 deprecation warnings.
- SHA-256 verification: 101 pre-existing application, test, model, and metadata files unchanged, including the generated smoke PDF relocated under reports/local-smoke/.
- Dataset, backup, report, and environment directories retained. GPU inference was not restarted or retrained during cleanup.
