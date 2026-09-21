# ML

Structure mirrors System Design §29. Phase 1 packages exist; the Phase 4 ones
land when that phase starts.

```
ml/
├── xray/           # fracture classification (Phase 1) — dataset, augment, model, train
├── data/           # de-identification, patient-level splits, manifest
├── preprocessing/  # shared transforms (xray-v1)
├── evaluation/     # runs against the frozen test split, separate from training
├── mri/            # sequence handling, preprocessing (Phase 4)
├── segmentation/   # anatomical structures (Phase 4)
└── classification/ # abnormality models, separate from segmentation (Phase 4)
```

Baseline results and the served operating point are in the root `README.md`;
the raw artifact is `experiments/outputs/xray-eval.json`.

Two rules that apply from the first experiment:

- **Augmentation is training-only.** It must never appear on an inference path,
  or the same image will produce different predictions on repeat runs
  (System Design §5, §22).
- **Log every run to MLflow** with config, metrics, checkpoint and dataset
  version — from experiment one, not once results look promising
  (System Design §20).
