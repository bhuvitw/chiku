# Implementation Plan

## AI-Assisted Musculoskeletal Imaging Platform

This turns `prd.md` and `System Design.md` into an execution order. It follows System Design §31 strictly: **data → baseline model → evaluation → localization → API → web UI → MRI → 3D → multimodal.** Each phase has a gate — don't start the next phase until the current one's exit criteria pass. A phase that's "code complete but unvalidated" is not done.

Time estimates assume one engineer working solo at research-project pace; treat them as ordering hints, not commitments.

---

## Phase 0 — Environment & Repo Scaffolding (~2-3 days)

Goal: a repo that runs, with nothing ML-specific working yet.

- [x] `git init`; set up `.gitignore` (data/, checkpoints, `.env`, `__pycache__`, `node_modules`)
- [x] Create repo structure from System Design §29 (`frontend/`, `backend/`, `ml/`, `data/`, `experiments/`, `scripts/`, `tests/`, `docker/`)
- [x] `docker-compose.yml`: Postgres, Redis, backend, frontend, ml-worker skeleton containers
- [x] Backend: FastAPI app skeleton, health check endpoint, Alembic migrations wired up
- [x] Postgres schema migration for `users`, `studies`, `jobs` only (System Design §15) — the rest of the tables land when the phase that needs them starts, not all upfront
- [x] Frontend: Next.js + TypeScript + Tailwind skeleton, one placeholder page
- [x] CI: lint + test workflow (GitHub Actions or equivalent) running on every push, even with near-empty test suites — this is cheaper to set up now than to retrofit later
- [x] DVC initialized (`dvc init`), pointed at whatever remote storage you'll use (local disk is fine to start)

**Exit criteria:** `docker-compose up` brings up all services; CI runs green on an empty test suite; a stub API call round-trips frontend → backend → Postgres.

**Verified:** migration applies and `alembic check` reports no drift from the models;
6 backend tests pass against Postgres; `ruff check`/`ruff format --check` clean;
`next build` clean with `/` correctly dynamic; round-trip confirmed end to end —
form submit → Server Action → API → Postgres, read back on the page.

**Not yet verified:** the `backend`, `worker` and `frontend` container images have
never been built (no Docker daemon access in the dev environment — Postgres and
Redis were run as containers, everything else natively), and CI has not run on a
remote yet. Both need one clean `docker compose up --build` and one pushed commit
before Phase 0 is genuinely closed.

---

## Phase 1 — Dataset & Baseline X-ray Model (~2-4 weeks, the highest-risk phase)

Goal: prove a fracture classifier is feasible on real data before writing a single line of product code around it. This is the phase most likely to invalidate assumptions in the PRD — treat it as research, not implementation.

### 1.1 Dataset acquisition

- [ ] Pick a starting X-ray dataset with patient-level identifiers and fracture labels. Concrete candidates to evaluate: **FracAtlas** (fracture classification + localization, permissive license), **GRAZPEDWRI-DX** (pediatric wrist fractures, bounding boxes included), **MURA** (musculoskeletal abnormality, not fracture-specific — useful for pretraining/backbone selection, not as the primary label source)
- [ ] Confirm the dataset actually has a `patient_id` field or equivalent — if it doesn't, patient-level splitting (PRD §10) is impossible and you need a different dataset
- [ ] Document dataset provenance and license in `data/README.md`

### 1.2 Data pipeline

- [ ] De-identification pass (System Design §4) even if the dataset claims to already be de-identified — verify, don't trust
- [ ] Patient-level split implementation (System Design §23) as a pipeline step with a **unit test** asserting zero patient-ID overlap across train/val/test — this is the one rule in the whole project that must never silently regress
- [ ] DVC-track the processed dataset; tag with a version the model registry will reference later

### 1.3 Baseline model

- [ ] Start with a simple, fast-to-train backbone (ResNet-18/34 or EfficientNet-B0) — resist the urge to reach for a ViT before a CNN baseline exists to compare against
- [ ] Standard training loop: augmentation only here (System Design §22), not in any inference path
- [ ] Log every run to MLflow: config, metrics, checkpoint, dataset version (System Design §20-21) from the very first experiment, not retrofitted later

### 1.4 Evaluation

- [ ] Build the evaluation pipeline as a separate script/module from training (System Design §24) — it should be runnable against any checkpoint + the frozen test split
- [ ] Compute sensitivity, specificity, precision, recall, F1, AUROC, confusion matrix, calibration (PRD §11)
- [ ] Decide the confidence-floor threshold for the abstention state (PRD §3, FR-04) based on the calibration curve, not a guessed round number

**Exit criteria (PRD §14, items 1-3):** reproducible predictions on the held-out patient-level test set; metrics reported with model+dataset version attached; calibration checked, not assumed. If sensitivity/specificity are unusably low here, stop and reassess dataset/scope before proceeding — don't carry a broken model into Phase 2.

---

## Phase 2 — X-ray Product / MVP (~2-3 weeks)

Goal: the PRD §13 MVP, end to end, with the model from Phase 1 behind it.

- [ ] Backend: `POST /studies`, `POST /studies/{id}/upload`, `POST /studies/{id}/analyze`, `GET /studies/{id}/status`, `GET /studies/{id}/results` (System Design §3, §19)
- [ ] Ingestion pipeline: format detection, validation, de-identification, storage (System Design §4) for PNG/JPEG first; DICOM validation can be stubbed to "reject with clear error" if not yet supported — don't silently mishandle it
- [ ] `jobs` table wired to Celery/Redis async flow (System Design §17); synchronous inline execution is an acceptable placeholder for the very first version if async infra isn't ready, but the API contract (status polling) must be final from day one so the frontend doesn't need to change later
- [ ] Wrap the Phase 1 model as an inference service; return the FR-04 JSON shape including the `unable_to_assess` abstention path — test this path explicitly, not just the happy path
- [ ] Error contract end to end (System Design §3, §26): `UNSUPPORTED_MODALITY`, `LOW_IMAGE_QUALITY`, `LOW_CONFIDENCE`, `PROCESSING_FAILED` all produce the specified user-facing copy
- [ ] Frontend: upload flow, processing-status view (progress bar per System Design §18), results view showing prediction + confidence + disclaimer
- [ ] Basic auth (even if minimal — email/password or a dev-mode bypass flag) so `users`/authorization has something real to check against

**Exit criteria (PRD §14, item 9 partially):** a user can upload a supported X-ray and see a result end to end, including at least one exercised failure path (bad file, low confidence).

---

## Phase 3 — Localization (~1-2 weeks)

Goal: PRD G2 — show *why*, not just *what*.

- [ ] Implement Grad-CAM (or equivalent) against the Phase 1 model (System Design §7, V1)
- [ ] Extend the results API/schema with the `localization`/`region` + `method` fields (PRD FR-05)
- [ ] Frontend: render the localization overlay on the source image, labeled explicitly as **model-generated** (PRD §12) — this label is not optional styling
- [ ] Error analysis pass: manually review a sample of localizations against the Phase 1 evaluation set to sanity-check that saliency maps are plausible, not just present

**Exit criteria (PRD §14, item 4):** every non-abstained prediction ships with a localization overlay, visually distinguished from any ground-truth annotation if the dataset has one.

At this point the MVP (PRD §13) is complete. **Pause and reassess before Phase 4** — MRI is a substantially larger scope increase than any prior phase.

---

## Phase 4 — Knee MRI Segmentation (~4-6 weeks)

Goal: PRD G3/G4 — segment the initial structure set and classify abnormalities as a separate task.

### 4.1 Dataset

- [ ] Evaluate knee MRI datasets with segmentation masks: **MRNet** (Stanford — ACL/meniscus/abnormal labels, no segmentation masks), **OAI** (Osteoarthritis Initiative — large, some segmentation subsets), or a smaller manually-curated set if neither has masks for all of ACL/PCL/menisci/cartilage
- [ ] If no dataset has masks for the full target structure list (PRD G3), narrow the v1 structure set rather than mixing weakly-labeled and strongly-labeled data without tracking which is which
- [ ] Apply the same patient-level split + de-identification pipeline as Phase 1 (reuse the code, don't reimplement)

### 4.2 DICOM/series handling

- [ ] Series identification and sequence selection (System Design §8) — build the "unsupported sequence" short-circuit first, since real MRI studies will contain sequences you don't handle
- [ ] Spatial normalization/resampling across series with differing voxel spacing

### 4.3 Segmentation model

- [ ] Start with MONAI's U-Net implementation before Attention U-Net/UNETR/Swin UNETR (System Design §9) — same "simple baseline first" discipline as Phase 1
- [ ] Register models/checkpoints in MLflow the same way as the X-ray model

### 4.4 Abnormality models

- [ ] Build as models genuinely separate from segmentation (System Design §10), with their own evaluation, even if they share a backbone

### 4.5 Evaluation

- [ ] Dice, IoU, sensitivity, specificity, Hausdorff distance per structure (PRD §11)

**Exit criteria (PRD §14, item 5):** segmentation masks produced for the supported structure set with Dice/IoU reported per structure on the held-out test set.

---

## Phase 5 — 3D Reconstruction (~2-3 weeks)

Goal: PRD G5.

- [ ] Marching cubes surface extraction from segmentation masks → mesh smoothing → polygon reduction → GLTF/GLB export (System Design §11)
- [ ] **Benchmark the full pipeline against the 30-60s target (PRD §8) early in this phase**, on realistic study sizes — this was flagged as a likely-to-slip target and should be measured, not assumed
- [ ] Frontend 3D viewer: React Three Fiber scene, rotate/zoom/pan, per-structure toggle, structure selection, AI-region highlighting (PRD G5)
- [ ] Finding-to-anatomy mapping (System Design §12): wire mesh IDs to structure IDs to findings so selection state is driven by one source of truth, not duplicated across viewer components

**Exit criteria (PRD §14, item 6):** at least the initial structure set reconstructs and renders interactively within the performance budget, or the budget is explicitly revised with data to back it.

---

## Phase 6 — Multimodal (Report NLP + Correlation) (~3-4 weeks)

Goal: PRD G6, System Design §13-14 — the phase most likely to produce confidently-wrong output if rushed.

- [ ] Report NLP pipeline: text normalization → section extraction → entity extraction → **negation detection → uncertainty detection** → structure normalization (System Design §13)
- [ ] Build a test set of report sentences specifically targeting negation/uncertainty edge cases (e.g. "no evidence of X", "cannot exclude X", "possible X") and assert correct certainty tagging **before** wiring this into the product — this is a unit-testable NLP correctness property, treat it like one
- [ ] Finding Engine: merge vision-model and report-derived findings per structure without collapsing `source` (System Design §14) — the FR-09 visual-correlation UI depends on this staying separate
- [ ] Frontend: FR-09 synchronized selection across finding list ↔ MRI slice viewer ↔ 3D viewer, with source-of-truth labeling visible on every finding

**Exit criteria (PRD §14, item 7):** selecting a structure or finding anywhere in the UI correctly synchronizes all three views, and every displayed finding is traceable to `vision_model`, `report_nlp`, or `ground_truth`.

---

## Cross-Cutting Workstreams (ongoing, not phase-gated)

These start in Phase 0/1 and continue throughout — don't defer them to "later," since retrofitting them is where most research prototypes accumulate silent risk.

- **Security/Privacy** (System Design §25): de-identification, access control, audit logging — present from Phase 2 onward, since that's when real uploads start flowing through the system.
- **Observability** (System Design §27): structured logs and the abstention/failure-rate metrics should exist by Phase 2, so Phase 4-6's more complex pipelines are debuggable from day one instead of retrofitted after a confusing failure.
- **Model registry & dataset versioning** (System Design §20-21): every phase that trains a model uses MLflow + DVC from its first experiment, not after results look promising.
- **Testing**: the patient-leakage check (Phase 1.2) and the negation/uncertainty test set (Phase 6) are the two highest-value correctness tests in the whole project — treat them as blocking, not optional.

---

## Milestone Summary

| Phase | Deliverable | Gate before proceeding |
|---|---|---|
| 0 | Running skeleton repo | CI green, services boot |
| 1 | Validated X-ray baseline | Metrics + calibration acceptable on patient-level test set |
| 2 | X-ray MVP end to end | PRD MVP demoable, failure paths exercised |
| 3 | Localization | Every prediction has a labeled visual explanation |
| 4 | Knee MRI segmentation | Per-structure Dice/IoU reported |
| 5 | 3D reconstruction | Interactive viewer within (or against a revised) perf budget |
| 6 | Multimodal correlation | Negation/uncertainty test set passing; 3-way UI sync working |

If any gate fails, the next phase does not start on schedule — the schedule bends to the validation result, not the other way around (System Design §31).
