# System Design
## AI-Assisted Musculoskeletal Imaging Platform

---

## 0. Design Principles

These constraints override individual component decisions below when they conflict:

1. **Prove the ML before building around it.** Architecture built on top of an unvalidated model gets rebuilt anyway (see §31).
2. **Fail closed.** Every stage that can't proceed confidently returns an explicit "unsupported/uncertain" state — never a best-effort guess dressed up as a result.
3. **Never mix provenance.** Model output, report-derived text, and expert ground truth are tracked and rendered as distinct categories, end to end, from the database up to the UI.
4. **Images never live in PostgreSQL.** Object storage holds bytes; Postgres holds references and structured data.
5. **Patient-level splits, always.** Enforced in code, not just process (§23).

---

## 1. High-Level Architecture

```text
                              ┌──────────────┐
                              │     User     │
                              └──────┬───────┘
                                     │ HTTPS
                                     ▼
                          ┌────────────────────┐
                          │  Next.js Frontend   │
                          │  (TypeScript)       │
                          └──────────┬──────────┘
                                     │ HTTPS / JSON
                                     ▼
                          ┌────────────────────┐
                          │   FastAPI Backend   │
                          └──────────┬──────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
      ┌───────────────┐     ┌───────────────┐      ┌───────────────┐
      │ Object Storage │     │  PostgreSQL   │      │  Redis Queue  │
      │  (studies, S3) │     │ (metadata)    │      │  (jobs)       │
      └───────────────┘     └───────────────┘      └───────┬───────┘
                                                             │
                                                             ▼
                                                    ┌─────────────────┐
                                                    │   ML Workers     │
                                                    │   (Celery)       │
                                                    └────────┬─────────┘
                                                             │
                             ┌───────────────────────────────┼───────────────────────────────┐
                             ▼                               ▼                               ▼
                     ┌───────────────┐               ┌───────────────┐               ┌───────────────┐
                     │  X-ray Model  │               │  MRI Pipeline │               │  Report NLP   │
                     │  (classify +  │               │  (segment +   │               │  (extract +   │
                     │  localize)    │               │  3D + abnorm.)│               │  negation)    │
                     └───────┬───────┘               └───────┬───────┘               └───────┬───────┘
                             └───────────────────────────────┼───────────────────────────────┘
                                                             ▼
                                                    ┌─────────────────┐
                                                    │ Results Engine  │
                                                    │ (writes DB, syncs
                                                    │  finding ↔ mesh) │
                                                    └────────┬─────────┘
                                                             ▼
                                                    Frontend Visualization
                                                    (2D viewer + 3D viewer,
                                                     synced by finding ID)
```

The frontend never talks to workers or storage directly — everything routes through the API, which is the single point enforcing auth, provenance tagging, and the fail-closed states.

---

## 2. Component A — Web Frontend

**Stack:** Next.js, TypeScript, Tailwind, React Three Fiber, Three.js.

**Responsibilities:** authentication, upload, study selection, X-ray viewer, MRI slice viewer, 3D viewer, result visualization, report display, confidence display, processing-status display.

**Design rule:** the frontend renders whatever provenance tag the API sends (`model_output` / `report_derived` / `ground_truth`) with a visually distinct treatment for each — this is a hard requirement, not a styling nicety, per Design Principle 3.

---

## 3. Component B — API Gateway

**Stack:** FastAPI.

**Responsibilities:** authentication, upload orchestration, study management, inference requests, results/report retrieval, model metadata.

```http
POST   /api/v1/studies
POST   /api/v1/studies/{id}/upload
POST   /api/v1/studies/{id}/analyze
GET    /api/v1/studies/{id}/status
GET    /api/v1/studies/{id}/results
GET    /api/v1/studies/{id}/visualization
GET    /api/v1/models                      # registry metadata, for reproducibility
```

**Error contract:** every error response carries a machine-readable `code` plus a human-readable `message`, so the frontend can map failure states to the specific copy required by §26 rather than a generic "something went wrong":

```json
{ "error": { "code": "UNSUPPORTED_MODALITY", "message": "This imaging study is not currently supported." } }
```

---

## 4. Component C — Imaging Ingestion

**Supported input:** PNG, JPEG, DICOM, NIfTI.

```text
Upload → Format detection → Validation → Metadata extraction → De-identification → Storage
```

DICOM path:

```text
DICOM → pydicom → {metadata, pixel data}
```

De-identification strips patient-identifying DICOM tags (name, DOB, MRN, institution, etc.) before anything is persisted or used for training — this runs even for prototype/test uploads, not just training data, so a bad upload can't leak PII into logs or storage.

---

## 5. Component D — X-Ray Pipeline

```text
X-ray → Quality check → Orientation normalization → Resize → Intensity normalization
      → Model → Fracture probability → Confidence check
      → (abstain if below threshold) → Localization → Explainability → Result
```

Note: this pipeline is the **inference** path — augmentation belongs strictly to training (§22) and never appears here, since it would make the same image produce different predictions on repeated inference.

---

## 6. X-Ray Model Architecture

```text
X-ray → CNN / Vision Transformer backbone → Feature extraction → Classification head → fracture probability
```

Candidate stack: PyTorch + ResNet / EfficientNet / ViT. The exact architecture is an experimental decision, made against the evaluation metrics in §24 — not fixed in advance by this document.

---

## 7. Localization — staged approach

### V1 — cheap, ships with the classifier

```text
Classification + Grad-CAM → region of interest
```

No extra labels required; explanation quality is coarse and must be labeled as such in the UI.

### V2 — only if annotation quality supports it

Train an explicitly localized model on bounding boxes or segmentation masks. This is strictly better when good annotations exist, but is not a prerequisite for the MVP (PRD §13) — don't block the MVP waiting on annotation collection.

---

## 8. MRI Pipeline

```text
DICOM study → Identify series → Select supported sequence → Sort slices
   → Spatial normalization → Resampling → Intensity normalization
   → Segmentation model → Anatomical masks
   → Abnormality models → 3D reconstruction
```

An unrecognized sequence must short-circuit here with an explicit "unsupported sequence" status rather than being forced through the rest of the pipeline.

---

## 9. MRI Segmentation

```text
                MRI
                 │
                 ▼
          Segmentation Model
                 │
   ┌──────┬──────┼──────┬──────┬──────┬──────┐
   ▼      ▼      ▼      ▼      ▼      ▼      ▼
  ACL    PCL   Med.   Lat.  Cartilage Femur Tibia/
                Menisc Menisc                Patella
   └──────┴──────┴──────┴──────┴──────┴──────┘
                 ▼
          Anatomical Masks
```

Candidate architectures: U-Net, Attention U-Net, UNETR, Swin UNETR — MONAI covers implementations and the surrounding medical-imaging tooling (loaders, transforms, metrics) for all of these.

---

## 10. Abnormality Detection

Kept as a **separate model/task from segmentation** — a structure can be segmented correctly while its abnormality status is a completely separate (and harder, noisier-label) prediction. Conflating the two makes both harder to debug and evaluate independently.

```text
MRI
 │
 ├─────────────┐
 ▼             ▼
Segmentation   Abnormality Model
 │             │
 ▼             ▼
{ACL, Meniscus, Cartilage, ...} masks    {ACL, Meniscus, Cartilage, ...} abnormality scores
```

---

## 11. 3D Reconstruction

```text
Segmentation masks → 3D voxel volume → thresholding → surface extraction (Marching Cubes)
   → mesh smoothing → polygon reduction → GLTF/GLB → Three.js rendering
```

**Frontend:** React Three Fiber + Three.js for the interactive viewer. VTK.js is a fallback option if a use case needs medical-visualization-specific features (volume rendering, DICOM-aware camera presets) that Three.js doesn't cover well.

Polygon reduction target should be benchmarked against the 30–60s generation budget (PRD §8) early — this is one of the more likely places the performance target slips.

---

## 12. Finding-to-Anatomy Mapping

```text
Finding → Structure ID → Segmentation mask → 3D mesh
```

```json
{
  "finding_id": "F102",
  "structure": "ACL",
  "structure_id": "ACL_001",
  "finding": "possible_abnormality",
  "confidence": 0.87,
  "source": "vision_model",
  "mesh_id": "mesh_acl_001"
}
```

The `source` field (`vision_model` | `report_nlp` | `ground_truth`) is what lets the frontend enforce the provenance separation required by Design Principle 3 — selecting "ACL" highlights the mesh, jumps relevant slices, and shows findings from all sources, each labeled by origin.

---

## 13. Radiology Report NLP

```text
Raw report → Text normalization → Section extraction → Entity extraction
   → Negation detection → Uncertainty detection → Structure normalization → Structured finding
```

This is the component most likely to produce a *confidently wrong* structured finding if under-built, because naive entity extraction ignores sentence-level negation and uncertainty:

```text
"There is no evidence of ACL tear."   must NOT become   { "structure": "ACL", "finding": "tear" }
```

Every extracted finding carries an explicit certainty tag from a closed set — `positive | negative | possible | uncertain` — and negation detection runs as its own pipeline stage, not as a post-hoc filter. A finding with no confidently assigned certainty tag should not be surfaced as if it were positive by default.

---

## 14. Multimodal Correlation

```text
             MRI
              │
       ┌──────┴──────┐
       ▼             ▼
   Vision Model   Report NLP
       │             │
       └──────┬──────┘
              ▼
        Finding Engine
              │
              ▼
        Structure ID
              │
              ▼
          3D Model
```

The Finding Engine merges vision-model and report-derived findings for the same structure **without collapsing them into one value** — a UI showing "ACL: possible tear" must be able to say whether that came from the image, the report, or both, and never mixed silently with ground-truth annotations if present. This is Design Principle 3 enforced at the point where it's most tempting to violate it.

---

## 15. Database Design

PostgreSQL, with explicit foreign keys and timestamps added throughout — the original sketch omitted both, which makes auditing ("who saw what, when, from which model version") impossible.

```text
users
─────
id               pk
email
role             student | clinician | admin
created_at

studies
───────
id               pk
user_id          fk -> users.id
modality         xray | mri
body_part
status           UPLOADED | VALIDATING | PREPROCESSING | ANALYZING
                 | GENERATING_3D | COMPLETED | FAILED | UNSUPPORTED
created_at
updated_at

jobs
────
id               pk
study_id         fk -> studies.id
job_type         analyze | generate_3d
status           QUEUED | RUNNING | SUCCEEDED | FAILED
progress         int (0-100)
error_code       nullable
created_at
completed_at

series
──────
id               pk
study_id         fk -> studies.id
modality
sequence
storage_location

images
──────
id               pk
series_id        fk -> series.id
storage_location
slice_index

models
──────
id               pk
name             e.g. fracture_detector
version          e.g. v1.2
dataset_version  fk -> dvc dataset tag
metrics_summary  jsonb
created_at

predictions
───────────
id               pk
study_id         fk -> studies.id
model_id         fk -> models.id
finding
confidence
created_at

structures
──────────
id               pk
name             e.g. ACL
body_part

annotations
───────────
id               pk
study_id         fk -> studies.id
structure_id     fk -> structures.id
annotation_type  bounding_box | segmentation_mask | grad_cam
source           model_output | ground_truth
location         jsonb

meshes
──────
id               pk
study_id         fk -> studies.id
structure_id     fk -> structures.id
model_id         fk -> models.id
storage_location

reports
───────
id               pk
study_id         fk -> studies.id
raw_text
created_at

findings
────────
id               pk
study_id         fk -> studies.id
structure_id     fk -> structures.id
finding_type
certainty        positive | negative | possible | uncertain
source           vision_model | report_nlp | ground_truth
confidence       nullable (ground_truth has none)
created_at
```

Every table that feeds a user-visible result carries `source`/`model_id` provenance and a timestamp — this is what makes "never silently replace the model used to generate previous results" (§20) actually checkable later, not just a policy statement.

---

## 16. Object Storage

```text
/studies/{study_id}/
    /dicom/
    /processed/
    /masks/
    /meshes/
    /reports/
```

PostgreSQL stores references only. Object storage: S3 (or S3-compatible for local/dev).

---

## 17. Asynchronous Processing

```text
User → POST /analyze → API → writes `jobs` row (QUEUED) → Redis queue
   → Celery worker picks up job → runs ML pipeline → writes results to DB
   → updates `jobs`.status/progress → Frontend polls status endpoint (or WebSocket for live progress)
```

Stack: Redis + Celery. A simpler in-process queue is acceptable for the prototype as long as the `jobs` table and status contract stay the same — swapping the queue implementation later shouldn't require an API change.

---

## 18. Processing States

```text
UPLOADED → VALIDATING → PREPROCESSING → ANALYZING → GENERATING_3D → COMPLETED
                                                              ↘ FAILED
                                                              ↘ UNSUPPORTED
```

Frontend renders progress directly from `jobs.progress`:

```text
MRI analysis
██████████████░░░░ 72%
Generating anatomical model...
```

`FAILED` and `UNSUPPORTED` are distinct terminal states on purpose — `FAILED` implies retry might help, `UNSUPPORTED` tells the user not to bother retrying with the same input.

---

## 19. API Example

**Start analysis**

```http
POST /api/v1/studies/{study_id}/analyze
```

```json
{ "study_id": "S001", "job_id": "J001", "status": "QUEUED" }
```

**Get status**

```http
GET /api/v1/studies/S001/status
```

```json
{ "status": "ANALYZING", "progress": 72 }
```

**Get results**

```http
GET /api/v1/studies/S001/results
```

```json
{
  "model_version": "fracture_detector:v1.2",
  "findings": [
    { "structure": "ACL", "finding": "possible_abnormality", "confidence": 0.87, "source": "vision_model" }
  ]
}
```

`model_version` at the top level of every result payload — not just buried per-finding — makes it trivial to tell at a glance which model run produced a given result.

---

## 20. ML Model Registry

MLflow. Every registered model version carries: model name, version, dataset version, training configuration, metrics, checkpoint, preprocessing version.

```text
fracture_detector: v1.0 → v1.1 → v1.2
```

**Hard rule:** never silently replace the model used to generate a previously-shown result. Existing `predictions`/`meshes`/`annotations` rows keep the `model_id` they were created with; re-running analysis with a newer model creates new rows rather than mutating old ones.

---

## 21. Dataset Versioning

DVC, alongside Git:

```text
Git → code
DVC → dataset version

data/
├── xray/
├── mri/
└── reports/
```

Model registry entries reference the DVC dataset tag they were trained on (`models.dataset_version` in §15), so a model's training data is always reconstructable.

---

## 22. Training Pipeline

```text
Raw Dataset → Validation → De-identification → Patient-level split
   → Preprocessing → Augmentation → Training → Validation
   → Hyperparameter tuning → Held-out test → MLflow → Model Registry
```

Augmentation lives here, and only here — contrast with the inference path in §5, which has no augmentation step.

---

## 23. Critical Dataset Rule

```text
NEVER:  image-level random split      (when a patient has multiple images)

INSTEAD:
Patient IDs → {train patients, validation patients, test patients} → generate images/studies from each split
```

This is enforced by writing the split at the patient-ID level before any image-level dataset construction runs — not by hoping the random split happens to work out. Patient leakage silently inflates every downstream metric in §24, so this check belongs in the data pipeline's tests, not just documentation.

---

## 24. Evaluation Architecture

```text
Test Dataset → Model → Predictions → Metrics → Report
```

| Modality | Metrics |
|---|---|
| X-ray | Sensitivity, specificity, precision, recall, F1, AUROC, calibration |
| MRI segmentation | Dice, IoU, Hausdorff distance |

Evaluation runs as a pipeline separate from training, against the frozen test split, and its output (with model version + dataset version) is what gets attached to a model registry entry — never a number pulled from a training log.

---

## 25. Security Architecture

```text
Browser → HTTPS → API → Authentication → Authorization → {PostgreSQL, Object Storage}
```

Requirements: TLS everywhere, authentication, authorization scoped per study/user, encrypted storage where appropriate, secrets management (not hardcoded, not in Git), audit logging of study access, temp-file deletion after processing, de-identification before storage/training use.

**Threat model note:** the highest-value target here is patient-identifiable imaging data — de-identification (§4) and access control are the load-bearing controls, not encryption alone. Any endpoint returning study data must check study ownership/authorization, not just authentication.

---

## 26. Failure Handling

Explicit states, explicit user-facing copy, mapped to the API error codes in §3:

| Situation | Error code | User-facing copy |
|---|---|---|
| Unsupported image/modality | `UNSUPPORTED_MODALITY` | "This imaging study is not currently supported." |
| Poor image quality | `LOW_IMAGE_QUALITY` | "Image quality is insufficient for reliable analysis." |
| Low model confidence | `LOW_CONFIDENCE` | "The model could not make a confident prediction." |
| Processing failure | `PROCESSING_FAILED` | "Analysis failed. Please retry." |

**Rule that overrides all of the above:** the system must never convert an error or a low-confidence result into a confident-sounding medical finding. Silence or an explicit abstention is always preferable to a fabricated answer.

---

## 27. Observability

Not present in the original design — added because a system that fails closed (Principle 2) needs to be debuggable when it does.

- Structured logs per pipeline stage (study ID, job ID, model version, duration, outcome) — no PII in log payloads.
- Metrics: job queue depth, per-stage latency, abstention rate, failure rate by error code.
- Alerting on abstention-rate or failure-rate spikes, since those indicate either a model regression or an upstream data problem.

---

## 28. Deployment

**Development:**

```text
Docker Compose: Frontend, Backend, PostgreSQL, Redis, ML Worker
```

**Optional cloud deployment:**

```text
Frontend      → Vercel / equivalent
Backend       → AWS / GCP / Azure
Database      → Managed PostgreSQL
Object Store  → S3
ML Worker     → GPU instance
```

---

## 29. Repository Structure

```text
musculoskeletal-ai/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── viewers/
│   └── three/
│
├── backend/
│   ├── api/
│   ├── models/
│   ├── services/
│   ├── database/
│   └── workers/
│
├── ml/
│   ├── xray/
│   ├── mri/
│   ├── segmentation/
│   ├── classification/
│   ├── preprocessing/
│   └── evaluation/
│
├── data/
│   └── README.md
│
├── experiments/
├── scripts/
├── tests/
├── docker/
│
├── dvc.yaml
├── docker-compose.yml
└── README.md
```

---

## 30. Development Roadmap

| Sprint | Focus |
|---|---|
| 1 | Dataset acquisition → cleaning → patient-level split → baseline model → evaluation |
| 2 | X-ray product: upload → inference → confidence → visualization → results page |
| 3 | Localization: Grad-CAM → error analysis |
| 4 | MRI: DICOM → preprocessing → knee MRI → segmentation |
| 5 | 3D: masks → reconstruction → Three.js viewer |
| 6 | Multimodal: MRI + report → finding extraction → finding↔anatomy mapping |

Each sprint's exit criteria are the corresponding PRD success criteria (PRD §14) — a sprint isn't "done" because the code merged, it's done when its slice of the success criteria is demonstrable.

---

## 31. Most Important Engineering Principle

Do not build the entire architecture before proving the ML.

```text
DATA → BASELINE MODEL → EVALUATION → LOCALIZATION → API → WEB UI → MRI → 3D → MULTIMODAL
```

The model and dataset determine what the product can honestly claim — not the other way around. If the baseline model can't clear reasonable sensitivity/specificity on a patient-level test set, no amount of frontend or 3D polish fixes that; find this out at step 2, not step 8.
