"""Baseline fracture-classifier training (implementation-plan §1.3).

Every run logs config, metrics, dataset version and checkpoint to MLflow from
experiment one, not once results look promising (System Design §20-21).

Runs unchanged on CPU and GPU, because the laptop this was written on trains
ResNet-18 at ~2.7 images/second: `--smoke` exists so the loop can be proven
correct locally in a couple of minutes before a real run is sent to a GPU.

    python -m ml.xray.train --smoke                      # minutes, CPU
    python -m ml.xray.train --epochs 15 --batch-size 32  # GPU
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.evaluation.classification import binary_metrics, calibration
from ml.xray.augment import AugmentConfig
from ml.xray.dataset import XrayCacheDataset
from ml.xray.model import build_model


@dataclass(frozen=True, slots=True)
class TrainConfig:
    cache: str = "data/processed/xray-384"
    backbone: str = "resnet18"
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 20260920
    pretrained: bool = True
    workers: int = 4
    limit: int | None = None


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Intel integrated graphics are not a torch backend: torch.xpu targets Arc
    # and Data Center GPUs, so a UHD-class iGPU falls through to CPU here.
    return torch.device("cpu")


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights.

    The training split is ~68% positive; unweighted, the cheapest way to a good
    loss is to lean positive, which costs exactly the specificity the product
    needs.
    """
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


@torch.inference_mode()
def collect_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def train(config: TrainConfig, *, device_name: str = "auto", run_name: str | None = None) -> Path:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = _device(device_name)

    cache = Path(config.cache)
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))

    train_set = XrayCacheDataset(
        cache, "train", augment_config=AugmentConfig(), seed=config.seed, limit=config.limit
    )
    val_set = XrayCacheDataset(cache, "val", limit=config.limit)
    # Patients must not span splits; the cache builder guarantees this, and
    # this re-checks it at the point of use rather than trusting the artefact.
    overlap = set(train_set.patient_ids) & set(val_set.patient_ids)
    if overlap:
        raise AssertionError(f"{len(overlap)} patients appear in both train and val: {overlap}")

    common = {"num_workers": config.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, **common)

    model = build_model(config.backbone, pretrained=config.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_set.labels).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    mlflow.set_experiment("xray-fracture-baseline")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                **asdict(config),
                "device": device.type,
                "train_images": len(train_set),
                "val_images": len(val_set),
                "dataset": meta["dataset"],
                "preprocess_version": meta["preprocess"]["version"],
                "resolution": meta["preprocess"]["resolution"],
                "split_seed": meta["split_seed"],
            }
        )
        best_auroc = -1.0
        checkpoint_path = Path("experiments/outputs") / f"{config.backbone}-best.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(config.epochs):
            model.train()
            started = time.perf_counter()
            running = 0.0
            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), targets)
                loss.backward()
                optimizer.step()
                running += loss.item() * images.size(0)
            scheduler.step()

            train_loss = running / len(train_set)
            labels, probabilities = collect_probabilities(model, val_loader, device)
            metrics = binary_metrics(labels, probabilities)
            reliability = calibration(labels, probabilities)
            elapsed = time.perf_counter() - started

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_auroc": metrics.auroc or float("nan"),
                    "val_sensitivity": metrics.sensitivity,
                    "val_specificity": metrics.specificity,
                    "val_ece": reliability.ece,
                    "val_brier": reliability.brier,
                    "epoch_seconds": elapsed,
                },
                step=epoch,
            )
            print(
                f"epoch {epoch + 1}/{config.epochs} loss={train_loss:.4f} "
                f"auroc={metrics.auroc:.4f} sens={metrics.sensitivity:.3f} "
                f"spec={metrics.specificity:.3f} ece={reliability.ece:.3f} "
                f"({elapsed / 60:.1f} min)"
            )

            if (metrics.auroc or 0.0) > best_auroc:
                best_auroc = metrics.auroc or 0.0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": asdict(config),
                        "preprocess": meta["preprocess"],
                        "dataset": meta["dataset"],
                        "split_seed": meta["split_seed"],
                        "epoch": epoch,
                        "val_auroc": best_auroc,
                    },
                    checkpoint_path,
                )

        # The checkpoint carries its preprocessing and dataset version, so a
        # registry entry can never point at a model whose training data is
        # unreconstructable (System Design §20-21).
        mlflow.log_artifact(str(checkpoint_path))
        print(f"\nbest val AUROC {best_auroc:.4f} → {checkpoint_path}")
        print("Evaluate on the frozen test split: python -m ml.evaluation.run_xray")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=TrainConfig.cache)
    parser.add_argument("--backbone", default=TrainConfig.backbone)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--workers", type=int, default=TrainConfig.workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="two epochs on a few hundred images — proves the loop, not the model",
    )
    args = parser.parse_args()

    config = TrainConfig(
        cache=args.cache,
        backbone=args.backbone,
        epochs=2 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        workers=0 if args.smoke else args.workers,
        seed=args.seed,
        limit=64 if args.smoke else None,
    )
    run_name = args.run_name or ("smoke" if args.smoke else None)
    train(config, device_name=args.device, run_name=run_name)


if __name__ == "__main__":
    main()
