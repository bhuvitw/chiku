"""Resume-state handling (ml/xray/train.py).

The value of a resume file is entirely in it being *correct*: a file that
loads but carries the wrong config silently produces a checkpoint whose
logged provenance describes a run that never happened. These tests cover the
cheap, pure parts of that contract — the path separation and the config
guard — without paying for a real training loop.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from ml.xray.model import build_model
from ml.xray.train import (
    TrainConfig,
    _checkpoint_path,
    _load_resume_state,
    _resume_path,
    _save_resume_state,
)


def _state(tmp_path, config: TrainConfig) -> None:
    model = build_model(config.backbone, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    _save_resume_state(
        tmp_path / "last.pt",
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs),
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        epoch=3,
        best_auroc=0.72,
        run_id="abc123",
        device=torch.device("cpu"),
        meta={"preprocess": {"version": "xray-v1"}, "dataset": "TEST", "split_seed": 1},
    )


def test_smoke_and_real_runs_never_share_files() -> None:
    """A two-minute smoke run must not overwrite a multi-hour run's outputs."""
    real = TrainConfig()
    smoke = dataclasses.replace(real, limit=64)
    assert _resume_path(real) != _resume_path(smoke)
    assert _checkpoint_path(real) != _checkpoint_path(smoke)


def test_missing_state_is_not_an_error(tmp_path) -> None:
    """`--resume` on a run that never started should begin from scratch."""
    assert _load_resume_state(tmp_path / "absent.pt", TrainConfig()) is None


def test_round_trip_preserves_position(tmp_path) -> None:
    config = TrainConfig()
    _state(tmp_path, config)
    state = _load_resume_state(tmp_path / "last.pt", config)
    assert state is not None
    assert state["epoch"] == 3
    assert state["best_auroc"] == pytest.approx(0.72)
    assert state["mlflow_run_id"] == "abc123"
    # Optimizer and scheduler state are the reason this file exists; the
    # best-checkpoint carries neither.
    assert "optimizer" in state and "scheduler" in state and "scaler" in state


@pytest.mark.parametrize(
    "field,value",
    [
        ("backbone", "resnet34"),
        ("cache", "data/processed/xray-224"),
        ("seed", 1),
        ("batch_size", 8),
        ("limit", 64),
    ],
)
def test_resume_refuses_a_different_config(tmp_path, field: str, value: object) -> None:
    """Continuing under changed data or architecture is not a resume."""
    _state(tmp_path, TrainConfig())
    changed = dataclasses.replace(TrainConfig(), **{field: value})
    with pytest.raises(SystemExit, match="different config"):
        _load_resume_state(tmp_path / "last.pt", changed)


def test_epochs_may_be_extended(tmp_path) -> None:
    """Raising the epoch budget and continuing is legitimate, not a mismatch."""
    config = TrainConfig()
    _state(tmp_path, config)
    longer = dataclasses.replace(config, epochs=config.epochs + 10)
    assert _load_resume_state(tmp_path / "last.pt", longer) is not None


def test_save_is_atomic(tmp_path) -> None:
    """No `.tmp` left behind, so an interrupted save cannot be mistaken for state."""
    _state(tmp_path, TrainConfig())
    assert (tmp_path / "last.pt").exists()
    assert not list(tmp_path.glob("*.tmp"))
