# Product Requirements Document

## AI-Assisted Musculoskeletal Imaging Platform

**Version:** 2.0
**Product Type:** AI/ML + Medical Imaging + 3D Visualization
**Primary Users:** Students/researchers, clinicians evaluating research tooling, medical-imaging educators
**Status:** Prototype / Research Project — not a medical device, not for clinical use

---

## 0. Safety Statement (read first)

This is a research and educational prototype. It is **not** a diagnostic device, is **not** FDA/CE cleared, and must never be presented as a substitute for a radiologist, physician, or emergency care.

Every output the system produces must:

- carry a visible "AI-generated, research use only" label,
- separate **model output**, **report-derived text**, and **expert ground truth** so a user can never mistake one for another,
- prefer saying "unable to assess" over guessing when confidence is low or input is unsupported.

Any requirement below that conflicts with this statement loses.

---

## 1. Product Vision

Build a research prototype that takes musculoskeletal X-rays and MRI scans and turns them into an interpretable bundle: an AI finding, a visual localization of *why* the model said that, structured text pulled from the radiology report, and an interactive 3D view of the relevant anatomy.

> Convert complex medical imaging into an interpretable combination of AI findings, visual localization, structured information, and 3D anatomy — without ever presenting itself as a diagnosis.

---

## 2. Problem

Medical imaging carries far more information than a non-specialist can extract unaided. The typical path —

```text
X-ray/MRI → radiologist interpretation → written report → clinician explanation → patient understanding
```

— has multiple lossy handoffs. This project explores whether an AI layer can shorten that path for research and educational purposes by:

- detecting potential abnormalities,
- localizing findings on the image,
- identifying anatomical structures,
- extracting structured findings from radiology reports,
- linking findings to the corresponding anatomy, and
- presenting all of this visually, with the uncertainty intact.

Scope is deliberately narrow: **musculoskeletal imaging**, starting with X-ray fracture detection and knee MRI.

---

## 3. Goals

Each goal below states the target behavior and how we'll know it's met — vague goals produce untestable features.

### G1 — X-ray fracture detection

Given a musculoskeletal X-ray, classify whether a fracture is likely present, with a calibrated confidence score and an explicit low-confidence/abstain state.

**Output contract:**

```text
Fracture:         possible | not detected | unable to assess
Confidence:       0-100%
Affected region:  e.g. distal radius (if resolvable)
Visual evidence:  highlighted region on the source image
```

**Acceptance criteria:** reproducible predictions on a held-out, patient-level test set; sensitivity/specificity reported (see §11); abstains rather than guesses below a defined confidence floor.

### G2 — Fracture localization

Identify the image region driving the prediction, using bounding boxes, segmentation, or Grad-CAM-style saliency (see System Design §7 for the staged approach). The UI must label this as **model-generated**, never as a confirmed finding.

### G3 — MRI anatomical segmentation (knee, first pass)

Segment a deliberately small, well-understood set of structures: ACL, PCL, medial meniscus, lateral meniscus, cartilage, femur, tibia, patella. Do not attempt exhaustive anatomy in v1 — partial coverage done well beats broad coverage done badly.

### G4 — MRI abnormality analysis

Report imaging-level observations tied to what the model can actually see, not clinical conclusions it can't measure.

Say: *"Imaging features associated with possible muscle atrophy."*
Never: *"Your muscle is 40% weak."*

Covers: suspected meniscal abnormality, suspected ligament abnormality, cartilage abnormality, structural changes.

### G5 — 3D visualization

Convert segmented structures into an interactive 3D scene. Users can rotate, zoom, pan, toggle structure visibility, select individual structures, and see AI-identified regions highlighted in context.

### G6 — Radiology report analysis

Extract structured findings from free-text reports, preserving certainty and negation — this is the piece most likely to silently produce wrong answers if done carelessly (see System Design §13).

```text
Structure   Finding        Location         Certainty            Laterality
Meniscus    Possible tear  Posterior horn   reported | suspected  left | right
```

---

## 4. Non-Goals

The first version explicitly will **not**:

- replace radiologists or provide a definitive diagnosis,
- prescribe treatment or recommend surgery,
- triage emergency care,
- measure actual muscle strength, bone density, or other physical quantities not directly observable in the image,
- diagnose every musculoskeletal condition or support every body part,
- auto-interpret arbitrary/unsupported MRI sequences,
- claim clinical-grade performance without a validation study to back it up.

If a feature request falls into this list, it's a separate, later decision — not a scope-creep default.

---

## 5. Target Users

| User | Need | Priority |
|---|---|---|
| Student / research user | Upload imaging, inspect predictions, localization, segmentation, 3D anatomy, and confidence — a learning and experimentation tool | Primary |
| Clinician / radiology researcher | Evaluate whether AI-assisted imaging workflows are worth pursuing further | Secondary, future |

---

## 6. User Journeys

### X-ray flow

```text
Upload → Validate → Preprocess → Inference → Confidence check
   → (abstain if low confidence) → Localization → Visualization → Structured result
```

### MRI flow

```text
Upload study → DICOM validation → Series identification → Preprocessing
   → Segmentation → Structure identification → Abnormality analysis
   → 3D reconstruction → Interactive visualization → Structured report
```

Both flows fail closed: any stage that can't proceed confidently routes to an explicit "unable to analyze" state (§8, Reliability) rather than pushing best-effort output downstream.

---

## 7. Functional Requirements

### FR-01 — Upload

Accept PNG/JPEG (prototype X-ray testing), DICOM (real imaging workflow), and NIfTI (research datasets). Reject anything else with a clear, specific error — not a generic failure.

### FR-02 — Validation

Before any processing, verify: supported format, decodable image, expected dimensionality, valid DICOM metadata where applicable, and absence of corruption/missing data. Validation failures must name the specific problem (e.g. "unsupported DICOM transfer syntax") rather than a blanket rejection.

### FR-03 — X-ray preprocessing (inference path)

```text
Raw image → Orientation normalization → Resize → Intensity normalization → Model input
```

Augmentation (flips, rotations, noise) is a **training-time-only** technique for improving model robustness — it must never run on the inference path, since it would make a single image produce non-deterministic predictions.

### FR-04 — Fracture prediction

```json
{
  "prediction": "possible_fracture",
  "confidence": 0.92,
  "model_version": "fracture_detector:v1.2"
}
```

Below the confidence floor, return an explicit abstention rather than a low-confidence guess:

```json
{ "prediction": "unable_to_assess", "reason": "confidence_below_threshold" }
```

### FR-05 — Localization

```json
{ "region": { "x": 210, "y": 155, "width": 80, "height": 95 }, "method": "grad_cam" }
```

or a pixel-level mask where the model version supports it. The `method` field lets the UI render an appropriate "this is model-generated" disclaimer.

### FR-06 — MRI segmentation

Produce a mask per supported structure (ACL, PCL, meniscus, cartilage, femur, tibia, patella), each tagged with model version and a per-structure confidence/quality score where available.

### FR-07 — 3D reconstruction

```text
2D masks → 3D volume → surface extraction (marching cubes) → mesh smoothing
   → polygon reduction → GLTF/GLB → WebGL rendering
```

### FR-08 — Report extraction

```json
{
  "structure": "medial_meniscus",
  "finding": "possible_tear",
  "location": "posterior_horn",
  "certainty": "possible",
  "negated": false
}
```

A report stating "no evidence of ACL tear" must extract `negated: true`, not a positive finding — see System Design §13 for how this is enforced.

### FR-09 — Visual correlation

Selecting a structure or finding anywhere in the UI must synchronize all three views:

```text
Finding  ↔  MRI slice  ↔  3D structure
```

e.g., selecting "ACL" highlights it in 3D, jumps the slice viewer to relevant slices, and surfaces any related report/AI findings.

### FR-10 — Results

Every result payload includes: AI finding, confidence, localization, relevant structure, source image/slice references, model version, a disclaimer, and an explicit uncertainty flag where applicable. No result is returned without traceability back to the model version that produced it.

---

## 8. Non-Functional Requirements

### Performance (engineering targets, not clinical requirements)

| Stage | Target |
|---|---|
| X-ray preprocessing | < 2s |
| X-ray inference | < 5s |
| MRI preprocessing | depends on study size |
| 3D generation | 30–60s for prototype-scale datasets |

### Security & Privacy

- TLS in transit; encrypted storage where appropriate.
- Minimize retention of identifiable patient information; de-identify training datasets.
- Access control on every study; audit log of who accessed what, when.
- Temporary files deleted after processing completes or fails.

### Reliability — fail safe, not fail confident

If image quality is poor, modality is unsupported, an MRI sequence is unrecognized, model confidence is low, or preprocessing fails, the system reports that it cannot reliably analyze the input. **It must never fabricate a result to fill the gap.** This is the single most important reliability property of the whole system.

---

## 9. ML Requirements

### X-ray dataset needs

Fracture-positive and -negative images, body-part metadata, laterality, clinical labels, and localization annotations where obtainable.

### MRI dataset needs

Knee MRI studies across multiple sequences, anatomical segmentation masks, abnormality labels, and paired radiology reports where available.

---

## 10. Dataset Splitting

Split by **patient**, never by image — a patient with multiple images leaking across train/test silently inflates every metric.

```text
Train 70% / Validation 15% / Test 15%   (patient-level; proportions may flex with dataset size)
```

No patient ID may appear in more than one split. This is enforced at the pipeline level, not just as a guideline (see System Design §23).

---

## 11. ML Evaluation

| Modality | Metrics |
|---|---|
| X-ray classification | Sensitivity, specificity, precision, recall, F1, AUROC, confusion matrix, calibration, localization performance where labels permit |
| MRI segmentation | Dice, IoU, sensitivity, specificity, Hausdorff distance where appropriate |

Every reported metric must state the test set it was computed on and the model version — numbers without provenance aren't usable for comparison later.

---

## 12. Explainability

Every prediction ships with visual evidence (Grad-CAM, attention maps, segmentation masks, or bounding boxes). The UI must visually and textually distinguish:

- **Model-generated visualization** (this is the AI's evidence for its own claim)

from

- **Ground-truth expert annotation** (this is what a radiologist actually marked, if present)

These are never rendered identically, and never merged into one visual layer.

---

## 13. MVP and Phasing

Building the full architecture before the ML is validated is the main risk to this project (see System Design §31). Phase gates exist to prevent that.

### MVP — X-ray only

```text
Upload → Fracture detection → Confidence → Localization → Visualization → Result
```

No MRI, no 3D, no report NLP in the MVP.

### Phase 2 — Knee MRI segmentation

Segmentation + structure identification (ACL/PCL/meniscus) + abnormality classification.

### Phase 3 — 3D + correlation

3D reconstruction, interactive anatomy viewer, finding↔anatomy correlation.

### Phase 4 — Multimodal

MRI + radiology report combined into structured findings with 3D visual explanation.

Each phase should ship independently demoable — don't start Phase N+1 before Phase N clears its own success criteria.

---

## 14. Success Criteria

The project succeeds when:

1. A supported X-ray processes end-to-end.
2. Fracture predictions are reproducible.
3. Quantitative evaluation exists on a held-out, patient-level test set.
4. The model provides localization/explanation for its predictions.
5. Knee MRI structures segment correctly for the supported set.
6. At least the initially-targeted structures reconstruct in 3D.
7. Findings link correctly to their corresponding anatomy.
8. Unsupported/low-quality inputs are handled gracefully, not silently mishandled.
9. The full pipeline — upload → inference → visualization — is demonstrable end to end.

---

## 15. Open Risks

- **Dataset availability/quality** for knee MRI with segmentation masks may be the actual bottleneck, not model architecture — validate this before committing to Phase 2 timelines.
- **Calibration drift**: a model can have good AUROC and still be badly calibrated; confidence numbers shown to users must be checked for this, not assumed.
- **Report NLP negation/uncertainty handling** is a known hard NLP problem; under-scoping this risks the exact failure mode called out in §7.13 of the System Design (flipping a negative finding into a positive one).
- **3D pipeline cost**: mesh generation at interactive quality within the 30–60s target may not hold for larger studies — needs early benchmarking, not late discovery.
