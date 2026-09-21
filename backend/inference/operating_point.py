"""The served operating point, read from the evaluation artifact.

PRD §3 requires the abstention threshold to come from the calibration curve
rather than a guessed round number, and `ml/evaluation/run_xray.py` already
picks it on validation and writes it to `xray-eval.json`. This module reads
that file rather than keeping a second, hand-copied set of the same numbers in
settings — a copy has no way to notice when the model behind it changes.

The file is the model's operating point, so serving one model's checkpoint
under another model's threshold is treated as a configuration error, not a
warning: the provenance block names the checkpoint it was measured on, and a
mismatch refuses to start.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from backend.config import settings

#: Relative paths in settings are repo-relative, not CWD-relative: the API,
#: the Celery worker and pytest are all started from different places.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Only used when no calibrated artifact exists and the stub is serving. These
#: are not an operating point; they exist so the abstention path stays
#: exercisable in tests that never load a model.
_PROVISIONAL = (0.5, 0.40, 0.60)


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    threshold: float
    abstain_low: float
    abstain_high: float
    #: Where these numbers came from, recorded so a stored prediction can be
    #: traced to the artifact that chose its threshold.
    source: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.abstain_low <= self.abstain_high <= 1.0:
            raise ValueError(
                f"abstention band [{self.abstain_low}, {self.abstain_high}) is not "
                "an interval within [0, 1]"
            )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold {self.threshold} is outside [0, 1]")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _REPO_ROOT / path


def _from_settings() -> OperatingPoint | None:
    """An explicit env-var override, which wins over the artifact.

    All three must be set together: a threshold moved without its band, or the
    reverse, is a half-configured operating point and the most likely way to
    end up serving one by accident.
    """
    values = (settings.decision_threshold, settings.abstention_low, settings.abstention_high)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise RuntimeError(
            "DECISION_THRESHOLD, ABSTENTION_LOW and ABSTENTION_HIGH must be set together; "
            f"got {values}"
        )
    threshold, low, high = values
    return OperatingPoint(float(threshold), float(low), float(high), source="settings")


def _from_artifact(path: Path) -> OperatingPoint:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selection = payload["selection"]
    provenance = payload.get("provenance", {})

    measured_on = provenance.get("checkpoint")
    configured = resolve_path(settings.model_checkpoint)
    if measured_on and resolve_path(measured_on).name != configured.name:
        raise RuntimeError(
            f"{path.name} reports an operating point measured on "
            f"{Path(measured_on).name}, but the configured checkpoint is "
            f"{configured.name}. Serving one model under another's threshold is "
            "not a warning — fix the configuration or re-run the evaluation."
        )

    low, high = selection["abstention_band"]
    return OperatingPoint(
        threshold=float(selection["threshold"]),
        abstain_low=float(low),
        abstain_high=float(high),
        source=f"{path.name}:{provenance.get('checkpoint', 'unknown')}",
    )


@lru_cache(maxsize=1)
def get_operating_point() -> OperatingPoint:
    """Resolve once per process: settings override, then artifact, then refuse."""
    override = _from_settings()
    if override is not None:
        return override

    path = resolve_path(settings.operating_point_path)
    if path.exists():
        return _from_artifact(path)

    if settings.inference_backend == "stub":
        threshold, low, high = _PROVISIONAL
        return OperatingPoint(threshold, low, high, source="provisional:stub")

    raise RuntimeError(
        f"no calibrated operating point: {path} does not exist and "
        f"inference_backend is {settings.inference_backend!r}. Run "
        "`python -m ml.evaluation.run_xray` or `dvc pull`, or set "
        "DECISION_THRESHOLD/ABSTENTION_LOW/ABSTENTION_HIGH explicitly."
    )
