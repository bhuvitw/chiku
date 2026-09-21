"""Inference boundary for the X-ray classifier (System Design §19).

Phase 2 ships the API around a stub so the whole product path — upload,
job, status polling, results, abstention — is exercised and testable before
the Phase 1 checkpoint exists. When it does exist, only `TorchPredictor`
lands: this module's shape, the API contract and the frontend do not move.

The backend deliberately does not import torch. The real predictor is
selected by `settings.inference_backend` and is imported lazily, so the API
image stays free of a multi-gigabyte dependency it only needs inside the
worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# The label set lives with the other enums because the database stores it too;
# one definition, so an API response and a `predictions` row can never disagree.
from backend.models.enums import PredictionLabel


class ImageQualityError(Exception):
    """Preprocessing's quality gate refused the image.

    Kept here rather than raised as an `AppError` directly, so a predictor
    never imports the API's error contract: the analysis service owns the
    mapping onto `LOW_IMAGE_QUALITY`. `QualityReport.passed` is what produces
    this (ml/preprocessing/xray.py).
    """

    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason or "quality_check_failed"
        super().__init__(self.reason)


@dataclass(frozen=True, slots=True)
class PredictionResult:
    prediction: PredictionLabel
    model_version: str
    #: Probability of fracture. Retained even when abstaining, because the
    #: stored value is what a later threshold change has to be re-judged
    #: against; it is simply not shown to the user in that state.
    probability: float
    confidence: float | None = None
    reason: str | None = None

    def as_response(self) -> dict[str, object]:
        """The FR-04 JSON shape, which differs by state on purpose."""
        if self.prediction is PredictionLabel.UNABLE_TO_ASSESS:
            return {
                "prediction": self.prediction.value,
                "reason": self.reason or "confidence_below_threshold",
                "model_version": self.model_version,
            }
        return {
            "prediction": self.prediction.value,
            "confidence": self.confidence,
            "model_version": self.model_version,
        }


def decide(
    probability: float,
    *,
    threshold: float,
    abstain_low: float,
    abstain_high: float,
    model_version: str,
) -> PredictionResult:
    """Turn a fracture probability into an FR-04 outcome.

    The band is `[low, high)` exactly as `ml.evaluation.classification`
    defines it, so the operating point chosen on validation there transfers
    here without reinterpretation. Confidence is reported as the probability
    of the class actually predicted, not the raw positive-class probability —
    a 0.08 fracture probability is a *confident* no-fracture call.
    """
    if abstain_low <= probability < abstain_high:
        return PredictionResult(
            prediction=PredictionLabel.UNABLE_TO_ASSESS,
            model_version=model_version,
            probability=probability,
            reason="confidence_below_threshold",
        )
    if probability >= threshold:
        return PredictionResult(
            prediction=PredictionLabel.POSSIBLE_FRACTURE,
            model_version=model_version,
            probability=probability,
            confidence=probability,
        )
    return PredictionResult(
        prediction=PredictionLabel.NO_FRACTURE,
        model_version=model_version,
        probability=probability,
        confidence=1.0 - probability,
    )


class Predictor(Protocol):
    model_version: str

    def probability(self, image_path: Path) -> float:
        """Probability that this view shows a fracture."""
        ...


def get_predictor() -> Predictor:
    from backend.config import settings

    if settings.inference_backend == "stub":
        from backend.inference.stub import StubPredictor

        return StubPredictor(settings.stub_probability)
    if settings.inference_backend == "torch":
        # Imported here and nowhere else: this is the only line in the
        # backend that may pull torch in, and only the worker takes it.
        from backend.inference.torch_predictor import load_predictor

        # Cached by path: a worker loads 45 MB of weights once per process,
        # not once per job.
        return load_predictor(settings.model_checkpoint)
    raise ValueError(f"unknown inference backend: {settings.inference_backend!r}")
