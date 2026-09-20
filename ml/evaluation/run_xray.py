"""Evaluate a checkpoint against the frozen test split (System Design §24).

Runs as a pipeline separate from training: it takes a checkpoint path and a
cache, and nothing it reports comes from a training log. The operating point
and the abstention band are chosen on **validation** and then applied unchanged
to test, so the reported test numbers are not tuned on the split they describe.

    python -m ml.evaluation.run_xray --checkpoint experiments/outputs/resnet18-best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.evaluation.classification import (
    binary_metrics,
    calibration,
    choose_abstention_band,
    choose_threshold,
    evaluate,
)
from ml.xray.dataset import XrayCacheDataset
from ml.xray.model import build_model


@torch.inference_mode()
def _probabilities(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels = [], []
    for images, targets in loader:
        logits = model(images.to(device))
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/processed/xray-384"))
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    parser.add_argument("--min-coverage", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("experiments/outputs/xray-eval.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"]["backbone"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    cache_meta = json.loads((args.cache / "meta.json").read_text(encoding="utf-8"))
    if cache_meta["preprocess"]["version"] != checkpoint["preprocess"]["version"]:
        raise SystemExit(
            f"preprocessing mismatch: checkpoint was trained with "
            f"{checkpoint['preprocess']['version']}, cache is "
            f"{cache_meta['preprocess']['version']}. Evaluating across these "
            "would silently measure the wrong thing."
        )

    val_set = XrayCacheDataset(args.cache, "val")
    test_set = XrayCacheDataset(args.cache, "test")
    if set(val_set.patient_ids) & set(test_set.patient_ids):
        raise AssertionError("patients overlap between val and test")

    loader = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for name, dataset in (("val", val_set), ("test", test_set))
    }

    # Operating point and abstention band: validation only.
    val_labels, val_probs = _probabilities(model, loader["val"], device)
    threshold = choose_threshold(val_labels, val_probs, target_sensitivity=args.target_sensitivity)
    band = choose_abstention_band(
        val_labels, val_probs, threshold=threshold, min_coverage=args.min_coverage
    )

    test_labels, test_probs = _probabilities(model, loader["test"], device)
    report = evaluate(
        test_labels,
        test_probs,
        threshold=threshold,
        band=band,
        groups={
            "cast": test_set.group_mask("cast"),
            "metal": test_set.group_mask("metal"),
        },
    )
    report["selection"] = {
        "threshold": threshold,
        "target_sensitivity": args.target_sensitivity,
        "selected_on": "val",
        "abstention_band": [band.low, band.high],
    }
    report["provenance"] = {
        "checkpoint": str(args.checkpoint),
        "dataset": cache_meta["dataset"],
        "preprocess_version": cache_meta["preprocess"]["version"],
        "split_seed": cache_meta["split_seed"],
        "test_patients": len(set(test_set.patient_ids)),
        "test_images": len(test_set),
    }
    # Validation numbers at the same operating point, so a large val/test gap
    # is visible rather than something a reader has to go looking for.
    report["val_at_operating_point"] = binary_metrics(
        val_labels, val_probs, threshold=threshold
    ).as_dict()
    report["val_calibration_ece"] = calibration(val_labels, val_probs).ece

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    overall = report["overall"]
    print(json.dumps(report["overall"], indent=2))
    print(
        f"\nabstention band [{band.low:.2f}, {band.high:.2f}) coverage {band.coverage:.1%} on val"
    )
    subgroups = report["subgroups"]
    print(
        f"cast sens {subgroups['cast']['sensitivity']:.3f} vs "
        f"no-cast sens {subgroups['not_cast']['sensitivity']:.3f} "
        "— a large gap means the model learned the treatment, not the fracture"
    )
    print(f"calibration ECE {report['calibration']['ece']:.4f}")
    print(f"\nwrote {args.out}")

    mlflow.set_experiment("xray-fracture-baseline")
    with mlflow.start_run(run_name=f"eval-{args.checkpoint.stem}"):
        mlflow.log_params(report["provenance"] | report["selection"])
        mlflow.log_metrics(
            {
                "test_auroc": overall["auroc"] or float("nan"),
                "test_sensitivity": overall["sensitivity"],
                "test_specificity": overall["specificity"],
                "test_ece": report["calibration"]["ece"],
                "test_brier": report["calibration"]["brier"],
                "cast_sensitivity": subgroups["cast"]["sensitivity"],
                "no_cast_sensitivity": subgroups["not_cast"]["sensitivity"],
            }
        )
        mlflow.log_artifact(str(args.out))


if __name__ == "__main__":
    main()
