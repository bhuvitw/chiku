"""Baseline fracture-classifier training (implementation-plan §1.3).

Every run logs config, metrics, dataset version and checkpoint to MLflow from
experiment one, not once results look promising (System Design §20-21).

Runs unchanged on CPU and GPU. Measured on the development laptop
(i7-8650U, no CUDA): ResNet-18 trains at 3.0 img/s at 224 and 1.1 img/s at
384, so a 15-epoch run is ~20 and ~55 hours respectively. Real runs belong on
a GPU; `--smoke` exists so the loop can be proven correct locally in a couple
of minutes before one is sent there.

Two things make a long run survivable:

- `--resume` continues from `experiments/outputs/<backbone>-last.pt`, which is
  rewritten every epoch. A preempted Colab or Kaggle session costs one epoch,
  not the whole run.
- Mixed precision is on by default and active only on CUDA, where it is worth
  roughly 2x. On CPU the autocast and the grad scaler are both no-ops, so this
  path is unchanged from before AMP existed.

    python -m ml.xray.train --smoke                      # minutes, CPU
    python -m ml.xray.train --epochs 15 --batch-size 32  # GPU
    python -m ml.xray.train --epochs 15 --resume         # after an interruption
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
    #: Requested, not effective: mixed precision only engages on CUDA.
    amp: bool = True


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Intel integrated graphics are not a torch backend: torch.xpu targets Arc
    # and Data Center GPUs, so a UHD-class iGPU falls through to CPU here.
    return torch.device("cpu")


def _resume_path(config: TrainConfig) -> Path:
    """Where the every-epoch training state lives.

    A smoke run must not leave a resume file that a real run would then refuse
    to start from, so the two never share a filename.
    """
    suffix = "-smoke" if config.limit is not None else ""
    return Path("experiments/outputs") / f"{config.backbone}{suffix}-last.pt"


def _checkpoint_path(config: TrainConfig) -> Path:
    """Best-AUROC checkpoint, under the same smoke/real split as the resume file.

    Without the suffix a two-minute `--smoke` run silently overwrites the
    checkpoint a multi-hour run produced, which is a bad way to find out.
    """
    suffix = "-smoke" if config.limit is not None else ""
    return Path("experiments/outputs") / f"{config.backbone}{suffix}-best.pt"


#: Config fields that must match for a resume to mean anything. `epochs` is
#: deliberately absent — extending a run's budget and continuing is legitimate
#: — but changing the data, architecture or seed would produce a checkpoint
#: whose logged config describes a run that never happened.
_RESUME_CRITICAL = ("backbone", "cache", "batch_size", "seed", "pretrained", "limit")


def _save_resume_state(
    path: Path,
    *,
    config: TrainConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_auroc: float,
    run_id: str,
    device: torch.device,
    meta: dict,
) -> None:
    """Write everything needed to continue, atomically.

    Optimizer, scheduler and scaler state are what the best-checkpoint does
    not carry, and resuming without them restarts Adam's moments and the LR
    schedule mid-run — which trains, but not the run you started.
    """
    payload = {
        "kind": "resume-state",
        "config": asdict(config),
        "epoch": epoch,
        "best_auroc": best_auroc,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "mlflow_run_id": run_id,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "preprocess": meta["preprocess"],
        "dataset": meta["dataset"],
        "split_seed": meta["split_seed"],
    }
    # Rename over the old file rather than writing in place: an interruption
    # during the save would otherwise destroy the state it is meant to protect.
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_resume_state(path: Path, config: TrainConfig) -> dict | None:
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    saved = state["config"]
    mismatched = {
        field: (saved.get(field), getattr(config, field))
        for field in _RESUME_CRITICAL
        if saved.get(field) != getattr(config, field)
    }
    if mismatched:
        raise SystemExit(
            f"cannot resume {path}: written under a different config {mismatched}. "
            "Delete it to start a fresh run."
        )
    return state


def _start_run(run_id: str | None, run_name: str | None):
    """Continue the interrupted run's MLflow record, or open a new one.

    A resumed run belongs in the same MLflow run as the epochs it continues,
    or its metric history is split across two records that neither compare nor
    plot. If that run is gone — a deleted `mlruns/`, a different tracking
    server — a new run beats crashing halfway through a resume.
    """
    if run_id is not None:
        try:
            return mlflow.start_run(run_id=run_id)
        except Exception as error:  # noqa: BLE001 - any tracking-store failure
            print(f"could not reopen MLflow run {run_id} ({error}); logging to a new run")
    return mlflow.start_run(run_name=run_name)


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
    *,
    amp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, targets in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images.to(device, non_blocking=True))
        # Back to fp32 before the softmax: these probabilities feed calibration
        # and the abstention band, and half precision is too coarse for either.
        probabilities.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def train(
    config: TrainConfig,
    *,
    device_name: str = "auto",
    run_name: str | None = None,
    resume: bool = False,
) -> Path:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = _device(device_name)
    # Requested vs effective: fp16 autocast is a CUDA win and a CPU pessimism,
    # so the flag asks and the device decides.
    amp_enabled = config.amp and device.type == "cuda"

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
    # Disabled on CPU, where it is a no-op passthrough, so this path stays
    # exactly what it was before AMP existed.
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    checkpoint_path = _checkpoint_path(config)
    resume_path = _resume_path(config)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_auroc = -1.0
    run_id: str | None = None
    if resume:
        state = _load_resume_state(resume_path, config)
        if state is None:
            print(f"no resume state at {resume_path}; starting from scratch")
        else:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            # Restoring the RNGs matters for more than tidiness: without it a
            # resumed run replays the same augmentation draws the interrupted
            # one already used, which is not the training set it reports.
            torch.set_rng_state(state["torch_rng"])
            np.random.set_state(state["numpy_rng"])
            if device.type == "cuda" and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            start_epoch = state["epoch"] + 1
            best_auroc = state["best_auroc"]
            run_id = state.get("mlflow_run_id")
            print(
                f"resuming {resume_path} at epoch {start_epoch + 1}/{config.epochs} "
                f"(best val AUROC so far {best_auroc:.4f})"
            )
            if start_epoch >= config.epochs:
                print("already at the epoch budget; raise --epochs to continue training")
                return checkpoint_path

    mlflow.set_experiment("xray-fracture-baseline")
    with _start_run(run_id, run_name):
        try:
            mlflow.log_params(
                {
                    **asdict(config),
                    "device": device.type,
                    "amp_effective": amp_enabled,
                    "train_images": len(train_set),
                    "val_images": len(val_set),
                    "dataset": meta["dataset"],
                    "preprocess_version": meta["preprocess"]["version"],
                    "resolution": meta["preprocess"]["resolution"],
                    "split_seed": meta["split_seed"],
                }
            )
        except Exception as error:  # noqa: BLE001 - tracking store rejects rewrites
            # Resuming on a different machine re-logs `device` with a new
            # value, which the store refuses. The metric history is the point
            # of reopening the run, so note it and carry on.
            print(f"params not re-logged on resume ({error})")

        for epoch in range(start_epoch, config.epochs):
            model.train()
            started = time.perf_counter()
            running = 0.0
            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    loss = criterion(model(images), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += loss.item() * images.size(0)
            scheduler.step()

            train_loss = running / len(train_set)
            labels, probabilities = collect_probabilities(
                model, val_loader, device, amp=amp_enabled
            )
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

            # Written every epoch, not only improving ones: this file exists to
            # survive an interruption, and the optimizer/scheduler/scaler state
            # that makes continuing possible is not in the best-checkpoint.
            _save_resume_state(
                resume_path,
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_auroc=best_auroc,
                run_id=mlflow.active_run().info.run_id,
                device=device,
                meta=meta,
            )

        # The checkpoint carries its preprocessing and dataset version, so a
        # registry entry can never point at a model whose training data is
        # unreconstructable (System Design §20-21).
        mlflow.log_artifact(str(checkpoint_path))
        print(f"\nbest val AUROC {best_auroc:.4f} → {checkpoint_path}")
        print("Evaluate on the frozen test split: python -m ml.evaluation.run_xray")
    return checkpoint_path


def _build_parser() -> argparse.ArgumentParser:
    # Defaults come from an *instance*, never from `TrainConfig.field`:
    # `slots=True` replaces the class attributes with member descriptors, so
    # the class-level form silently hands argparse a descriptor object that
    # only fails much later, inside the training run.
    defaults = TrainConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=defaults.cache)
    parser.add_argument("--backbone", default=defaults.backbone)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"default {defaults.epochs}, or 2 under --smoke",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="a couple of epochs on 64 images — proves the loop, not the model",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from experiments/outputs/<backbone>-last.pt",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="disable mixed precision on GPU runs (no effect on CPU)",
    )
    parser.set_defaults(amp=True)
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Build the run config, so the CLI's defaults are testable without training."""
    defaults = TrainConfig()
    # An explicit --epochs wins even under --smoke, which is what makes a
    # resume verifiable on a 64-image run rather than only on a real one.
    unset = 2 if args.smoke else defaults.epochs
    epochs = unset if args.epochs is None else args.epochs
    return TrainConfig(
        cache=args.cache,
        backbone=args.backbone,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        workers=0 if args.smoke else args.workers,
        seed=args.seed,
        limit=64 if args.smoke else None,
        amp=args.amp,
    )


def main() -> None:
    args = _build_parser().parse_args()
    config = config_from_args(args)
    run_name = args.run_name or ("smoke" if args.smoke else None)
    train(config, device_name=args.device, run_name=run_name, resume=args.resume)


if __name__ == "__main__":
    main()
