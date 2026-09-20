# ML

Empty until Phase 1. Structure mirrors System Design §29:

```
ml/
├── xray/           # fracture classification (Phase 1)
├── mri/            # sequence handling, preprocessing (Phase 4)
├── segmentation/   # anatomical structures (Phase 4)
├── classification/ # abnormality models, separate from segmentation (Phase 4)
├── preprocessing/  # shared transforms
└── evaluation/     # runs against the frozen test split, separate from training
```

Two rules that apply from the first experiment:

- **Augmentation is training-only.** It must never appear on an inference path,
  or the same image will produce different predictions on repeat runs
  (System Design §5, §22).
- **Log every run to MLflow** with config, metrics, checkpoint and dataset
  version — from experiment one, not once results look promising
  (System Design §20).
