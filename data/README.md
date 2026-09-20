# Data

Nothing in this directory is tracked by Git (see `.gitignore`). Datasets are
versioned with DVC; Git tracks only the `.dvc` pointer files and this README.

## Rules

1. **Patient-level splits only.** Never split by image when a patient can
   contribute more than one image — it leaks patients across train/test and
   silently inflates every metric downstream (PRD §10, System Design §23).
   The split is computed over patient IDs *before* any image-level dataset is
   built, and a test asserts zero overlap.
2. **De-identify before storing.** Run the de-identification pass even on
   datasets that claim to already be de-identified — verify, don't trust
   (System Design §4).
3. **Record provenance.** Every dataset added here gets an entry below: source,
   licence, date obtained, and whether it carries patient IDs and localization
   annotations.

## Datasets

| Name | Source | Licence | Patient IDs | Annotations | Added |
|---|---|---|---|---|---|
| GRAZPEDWRI-DX | [figshare 14825193](https://doi.org/10.6084/m9.figshare.14825193) ([paper](https://www.nature.com/articles/s41597-022-01328-z)) | CC BY 4.0 | **Yes** — explicit `patient_id` column | Image-level fracture labels + YOLO/VOC bounding boxes | 2026-09-20 |

### GRAZPEDWRI-DX — Phase 1 primary dataset

Pediatric wrist trauma radiographs, University Hospital Graz, 2008–2018.
16.3 GB across four zips plus a 1.8 MB `dataset.csv`; open download, no
registration. Attribution is required under CC BY 4.0 — carry the citation into
anything published from it.

Measured from `dataset.csv` (not quoted from the paper):

| | |
|---|---|
| Images | 20,327 |
| Patients | 6,091 |
| Studies | 10,699 |
| Images per patient | mean 3.34, max 30; **only 96 patients have just one** |
| `fracture_visible = 1` | 13,550 (66.7%) |
| Patients with ≥1 visible fracture | 3,587 (58.9%) |
| Flagged `diagnosis_uncertain` | 537 |
| Cast present | 5,776 (28.4%) |
| Metal implant present | 708 |
| Osteopenia | 2,473 |
| Age range | 0.2 – 19.0 years |
| Device | Siemens 20,215 / Agfa 112 |

**Why this and not FracAtlas.** FracAtlas was the plan's first candidate, but
its `dataset.csv` carries no patient identifier — patient data was stripped
during anonymization, leaving only a `multiscan` flag for repeat views of the
same organ. Rule 1 above is therefore unenforceable on it, which is exactly the
disqualifier implementation-plan §1.1 names. GRAZPEDWRI-DX has a real
`patient_id` column, and its bounding boxes also carry Phase 3 (localization)
for free. FracAtlas and MURA remain useful for pretraining, never as the
primary label source.

**Why the patient-level rule is load-bearing here.** 98.4% of patients
contribute more than one image, so an image-level random split would put the
same wrist in train and test. This is not a theoretical risk on this dataset.

### Known confounders — carry these into evaluation

- **Casts (28.4% of images).** A cast is visible, strongly correlated with
  fracture, and trivially learnable. A model that learns "cast ⇒ fracture"
  scores well overall and fails on exactly the pre-treatment images that
  matter. `ml.data.manifest` carries `cast` through to evaluation so metrics
  can be reported stratified by it; a large gap between the cast and no-cast
  subgroups means the headline number is not measuring fracture detection.
- **Metal implants (708 images)** — same shortcut, smaller scale.
- **Projection.** Studies contain multiple views (~50/50 AP and lateral). A
  fracture invisible in one view is still labelled at the study level; see the
  label policy below.

### Label policy

`label = 1` iff `fracture_visible` is set. 773 images carry an
`ao_classification` (a fracture was diagnosed for the study) with no
`fracture_visible` flag — these stay **negative**, because the per-image
question is whether this view shows a fracture, and training a model to predict
from a view containing no evidence teaches it to guess. The 537
`diagnosis_uncertain` rows are dropped by default rather than guessed at
(fail closed, System Design §0.2); `load_manifest(drop_uncertain=False)` keeps
them. After dropping: 19,790 images across 5,876 patients.

### Generalization limits — do not overstate Phase 1 results

Single institution, one manufacturer (99.4% Siemens), pediatric only (0.2–19
years), wrist only. Any metric produced in Phase 1 describes *pediatric wrist
radiographs from one Austrian hospital* — not adult imaging, not other body
parts, and not other scanners. The PRD's broader musculoskeletal scope is not
evidenced by this dataset and a second dataset is needed before it can be.

### Reproducing

```bash
python -m scripts.download_grazpedwri --labels-only   # 1.8 MB, seconds
python -m scripts.download_grazpedwri --extract       # ~16 GB, hours
python -m scripts.build_xray_cache --resolution 384   # CPU, one pass
```

Candidates under evaluation for Phase 4 (knee MRI) are listed in
`implementation-plan.md` §4.1.
