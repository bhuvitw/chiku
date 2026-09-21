# AI-Assisted Musculoskeletal Imaging Platform

Research and educational prototype. **Not a medical device, not FDA/CE cleared,
and not for clinical use.** No output from this system is a diagnosis.

- [`prd.md`](prd.md) — what it does and explicitly does not do
- [`System Design — AI-Assisted Musculoskeletal Imaging Platform.md`](System%20Design%20—%20AI-Assisted%20Musculoskeletal%20Imaging%20Platform.md) — architecture
- [`implementation-plan.md`](implementation-plan.md) — phased build order and exit criteria

## Status

**Phase 2 complete — the X-ray MVP runs end to end.** A user signs in, uploads a
wrist radiograph, polls a job, and sees a prediction with a confidence and a
disclaimer; bad files and low-confidence images take their own labelled paths.
110 tests pass.

| Phase | State |
|---|---|
| 0 — scaffolding | Done. Services boot, migrations apply, CI workflow in place |
| 1 — dataset & baseline X-ray model | Done. ResNet-18 on GRAZPEDWRI-DX, evaluated on a patient-level test split |
| 2 — X-ray product / MVP | Done. Upload → analyze → poll → result, with the model behind it |
| 3 — localization (Grad-CAM) | **Next.** Not started |
| 4-6 — MRI, 3D, multimodal | Not started |

Still unverified from Phase 0: the container images have never been built on a
machine with a Docker daemon, and CI has not run on a remote. Everything below
was run natively against containerized Postgres and Redis.

### Phase 1 — baseline model results

ResNet-18, GRAZPEDWRI-DX pediatric wrist radiographs, patient-level split
(seed `20260920`), held-out test set of **3,021 images from 882 patients** with
zero patient overlap. Full artifact: [`experiments/outputs/xray-eval.json`](experiments/outputs/xray-eval.json).

| Metric | Test |
|---|---|
| Sensitivity | 0.959 |
| Specificity | 0.910 |
| Precision / F1 | 0.960 / 0.960 |
| AUROC / AUPRC | 0.985 / 0.994 |
| Brier / ECE | 0.043 / 0.025 |

The decision threshold (**0.4184**) and the abstention band
(**[0.038, 0.798)**) were chosen on validation against a 0.95 sensitivity
target, from the calibration curve — not guessed. Under abstention: coverage
0.866, selective sensitivity 0.985, selective specificity 0.940.

**Read these numbers with the caveats.** They describe pediatric wrist
radiographs from one Austrian hospital on one scanner vendor, not adult imaging
or other body parts. Casts are a known shortcut in this dataset (28.4% of
images) and metrics are reported stratified by it — specificity collapses to
0.10 on the cast subgroup, which is the expected sign of the confounder, not a
regression. `data/README.md` has the full provenance, label policy and
generalization limits; carry them into any claim made from this model.

### Phase 2 — what's wired

- `POST /studies`, `GET /studies`, `GET /studies/{id}`, `POST /studies/{id}/upload`,
  `POST /studies/{id}/analyze`, `GET /studies/{id}/status`, `GET /studies/{id}/results`
- `TorchPredictor` serving the Phase 1 checkpoint, taking its preprocessing from
  the checkpoint rather than from settings; the operating point is read from the
  evaluation artifact that chose it and refuses to start against a different one
- Celery/Redis async jobs (`ASYNC_JOBS=true`), with an inline path running the
  same `backend.services.analysis` code so a bug cannot exist in only one
- Error contract end to end: `UNSUPPORTED_MODALITY`, `LOW_IMAGE_QUALITY`,
  `LOW_CONFIDENCE`, `PROCESSING_FAILED`, each with fixed user-facing copy
- Frontend: sign-in, dropzone upload, polling processing view, results view with
  confidence and disclaimer, error view
- Minimal session (cookie or `DEV_AUTH_BYPASS_EMAIL` bypass) — deliberately not
  real authentication; see the note at the top of `frontend/lib/session.ts`

Not yet: DICOM (rejected with a clear error), localization overlays, per-user
data isolation.

## Running it

```bash
cp .env.example .env
dvc pull                    # fetches experiments/outputs/resnet18-best.pt
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Health | http://localhost:8000/api/v1/health |

The backend applies migrations on start, so the first boot creates the schema.
Without the checkpoint, set `INFERENCE_BACKEND=stub` — a deployment configured
for `torch` that cannot find its weights fails loudly rather than silently
serving fakes.

### Without Docker

```bash
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.txt
# Postgres and Redis must be reachable per .env
.venv/bin/alembic upgrade head
.venv/bin/uvicorn backend.main:app --reload
.venv/bin/celery -A backend.workers.celery_app worker -l info   # only if ASYNC_JOBS=true

cd frontend && npm install && npm run dev
```

## Reproducing the model

```bash
python -m scripts.download_grazpedwri --labels-only   # 1.8 MB, seconds
python -m scripts.download_grazpedwri --extract       # ~16 GB, hours
python -m scripts.build_xray_cache --resolution 384   # CPU, one pass

python -m ml.xray.train                               # logs to MLflow
python -m ml.evaluation.run_xray --checkpoint experiments/outputs/resnet18-best.pt
```

Training logs config, metrics, checkpoint and dataset version to MLflow from
the first run. Evaluation is a separate entry point that runs against any
checkpoint and the frozen test split, and writes the operating point the API
then serves.

## Tests

```bash
.venv/bin/pytest          # 110 tests; runs against Postgres, not SQLite — see backend/tests/conftest.py
.venv/bin/ruff check backend ml scripts
cd frontend && npm run lint && npm run build
```

The patient-leakage test in `ml/tests/test_splits.py` is the one check in this
repo that must never be allowed to regress silently (PRD §10).

## Layout

```
backend/    FastAPI app, SQLAlchemy models, Celery workers, Alembic migrations
            inference/  predictor interface, TorchPredictor, stub, operating point
frontend/   Next.js app (App Router, Tailwind)
ml/         data/ deid + patient-level splits + manifest
            preprocessing/  xray-v1 transforms
            xray/  dataset, augmentation, model, training loop
            evaluation/  metrics, calibration, threshold selection
data/       DVC-tracked datasets; nothing here is committed to Git
experiments/  checkpoints (DVC) and evaluation artifacts (JSON, in Git)
docker/     container definitions
scripts/    dataset download, preprocessing cache, CPU benchmark
```
