"""The served operating point, and the FR-04 states it produces.

These are the numbers that decide what a user is told, so the tests care less
about "does it load" than about the ways it could load the *wrong* numbers:
a half-set override, an artifact measured on a different checkpoint, or a
missing artifact silently falling back to something plausible.

No torch and no database here — this is pure contract.
"""

from __future__ import annotations

import json

import pytest

from backend.config import settings
from backend.inference import PredictionLabel, decide
from backend.inference.operating_point import (
    OperatingPoint,
    get_operating_point,
)

# The real calibrated point, from experiments/outputs/xray-eval.json.
THRESHOLD = 0.4184403121471405
BAND = (0.03844031214714033, 0.7984403121471406)


@pytest.fixture(autouse=True)
def _clear_cache():
    """The operating point is resolved once per process; tests need it fresh."""
    get_operating_point.cache_clear()
    yield
    get_operating_point.cache_clear()


@pytest.fixture
def _no_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    for field in ("decision_threshold", "abstention_low", "abstention_high"):
        monkeypatch.setattr(settings, field, None)


def write_artifact(path, *, checkpoint="experiments/outputs/resnet18-best.pt"):
    path.write_text(
        json.dumps(
            {
                "selection": {"threshold": THRESHOLD, "abstention_band": list(BAND)},
                "provenance": {"checkpoint": checkpoint},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_reads_the_real_evaluation_artifact(_no_overrides) -> None:
    """The committed artifact is the source of truth, not a copy in settings."""
    point = get_operating_point()
    assert point.threshold == pytest.approx(THRESHOLD)
    assert (point.abstain_low, point.abstain_high) == pytest.approx(BAND)
    assert "xray-eval.json" in point.source


def test_explicit_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "decision_threshold", 0.6)
    monkeypatch.setattr(settings, "abstention_low", 0.2)
    monkeypatch.setattr(settings, "abstention_high", 0.8)
    point = get_operating_point()
    assert (point.threshold, point.abstain_low, point.abstain_high) == (0.6, 0.2, 0.8)
    assert point.source == "settings"


def test_half_set_override_is_refused(monkeypatch: pytest.MonkeyPatch, _no_overrides) -> None:
    """A threshold moved without its band is a half-configured operating point."""
    monkeypatch.setattr(settings, "decision_threshold", 0.6)
    with pytest.raises(RuntimeError, match="must be set together"):
        get_operating_point()


def test_artifact_from_another_checkpoint_is_refused(
    tmp_path, monkeypatch: pytest.MonkeyPatch, _no_overrides
) -> None:
    """Serving one model under another model's threshold is a config error."""
    artifact = write_artifact(tmp_path / "eval.json", checkpoint="outputs/some-other-model.pt")
    monkeypatch.setattr(settings, "operating_point_path", str(artifact))
    with pytest.raises(RuntimeError, match="measured on"):
        get_operating_point()


def test_missing_artifact_refuses_to_serve_a_real_model(
    tmp_path, monkeypatch: pytest.MonkeyPatch, _no_overrides
) -> None:
    monkeypatch.setattr(settings, "operating_point_path", str(tmp_path / "absent.json"))
    monkeypatch.setattr(settings, "inference_backend", "torch")
    with pytest.raises(RuntimeError, match="no calibrated operating point"):
        get_operating_point()


def test_missing_artifact_still_lets_the_stub_run(
    tmp_path, monkeypatch: pytest.MonkeyPatch, _no_overrides
) -> None:
    """Tests that never load a model must not need the evaluation artifact."""
    monkeypatch.setattr(settings, "operating_point_path", str(tmp_path / "absent.json"))
    monkeypatch.setattr(settings, "inference_backend", "stub")
    assert get_operating_point().source == "provisional:stub"


@pytest.mark.parametrize(
    "threshold,low,high",
    [(0.5, 0.8, 0.2), (0.5, -0.1, 0.9), (1.5, 0.1, 0.9)],
)
def test_nonsensical_bands_are_rejected(threshold, low, high) -> None:
    with pytest.raises(ValueError):
        OperatingPoint(threshold, low, high, source="test")


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.0, PredictionLabel.NO_FRACTURE),
        (BAND[0] - 1e-9, PredictionLabel.NO_FRACTURE),
        # The band is half-open [low, high): low abstains, high does not.
        (BAND[0], PredictionLabel.UNABLE_TO_ASSESS),
        (THRESHOLD, PredictionLabel.UNABLE_TO_ASSESS),
        (BAND[1] - 1e-9, PredictionLabel.UNABLE_TO_ASSESS),
        (BAND[1], PredictionLabel.POSSIBLE_FRACTURE),
        (1.0, PredictionLabel.POSSIBLE_FRACTURE),
    ],
)
def test_band_edges_under_the_real_operating_point(probability, expected) -> None:
    result = decide(
        probability,
        threshold=THRESHOLD,
        abstain_low=BAND[0],
        abstain_high=BAND[1],
        model_version="test",
    )
    assert result.prediction is expected


def test_confidence_is_the_predicted_class_not_the_positive_class() -> None:
    """A 0.01 fracture probability is a *confident* no-fracture call."""
    result = decide(
        0.01, threshold=THRESHOLD, abstain_low=BAND[0], abstain_high=BAND[1], model_version="t"
    )
    assert result.prediction is PredictionLabel.NO_FRACTURE
    assert result.confidence == pytest.approx(0.99)


def test_abstention_response_hides_confidence_but_keeps_probability() -> None:
    """FR-04: the user sees no number, but the stored row keeps one."""
    result = decide(
        0.5, threshold=THRESHOLD, abstain_low=BAND[0], abstain_high=BAND[1], model_version="t"
    )
    body = result.as_response()
    assert body["prediction"] == "unable_to_assess"
    assert "confidence" not in body
    assert result.probability == 0.5
