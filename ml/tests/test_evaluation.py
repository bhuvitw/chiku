"""Evaluation tests (System Design §24, PRD §11).

Every case has a hand-checkable answer — an evaluation module that is itself
wrong is worse than none, because its output is what gets attached to a model
registry entry and quoted as a result.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ml.evaluation.classification import (
    binary_metrics,
    calibration,
    choose_abstention_band,
    choose_threshold,
    evaluate,
    sensitivity_specificity_curve,
    subgroup_metrics,
)


def test_confusion_matrix_matches_a_hand_computed_example() -> None:
    labels = [1, 1, 1, 0, 0, 0]
    probs = [0.9, 0.8, 0.2, 0.7, 0.1, 0.1]
    metrics = binary_metrics(labels, probs, threshold=0.5)
    assert (metrics.tp, metrics.fn, metrics.fp, metrics.tn) == (2, 1, 1, 2)
    assert metrics.sensitivity == pytest.approx(2 / 3)
    assert metrics.specificity == pytest.approx(2 / 3)
    assert metrics.precision == pytest.approx(2 / 3)  # 2 tp of 3 predicted positive
    assert metrics.f1 == pytest.approx(2 / 3)


def test_a_perfect_separator_scores_one() -> None:
    labels = [0, 0, 1, 1]
    metrics = binary_metrics(labels, [0.01, 0.02, 0.98, 0.99])
    assert metrics.sensitivity == 1.0
    assert metrics.specificity == 1.0
    assert metrics.auroc == 1.0


def test_a_subgroup_without_positives_reports_nan_not_zero() -> None:
    # 0.0 would read as "the model fails here"; the truth is "no denominator".
    metrics = binary_metrics([0, 0, 0], [0.1, 0.2, 0.9])
    assert math.isnan(metrics.sensitivity)
    assert metrics.specificity == pytest.approx(2 / 3)
    assert metrics.auroc is None


def test_a_calibrated_model_has_near_zero_ece() -> None:
    rng = np.random.default_rng(3)
    probs = rng.uniform(0.02, 0.98, 40000)
    labels = (rng.uniform(size=probs.size) < probs).astype(int)
    result = calibration(labels, probs)
    assert result.ece < 0.02
    assert result.brier < 0.25


def test_an_overconfident_model_is_caught_by_ece() -> None:
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 2, 20000)
    # Always claims 0.99 / 0.01 but is right only 70% of the time.
    correct = rng.uniform(size=labels.size) < 0.7
    probs = np.where(correct == (labels == 1), 0.99, 0.01)
    probs = np.where(labels == 1, np.where(correct, 0.99, 0.01), np.where(correct, 0.01, 0.99))
    result = calibration(labels, probs)
    assert result.ece > 0.25
    assert result.mce > 0.25


def test_calibration_bins_cover_every_sample() -> None:
    rng = np.random.default_rng(5)
    probs = np.r_[rng.uniform(size=500), 0.0, 1.0]
    labels = rng.integers(0, 2, probs.size)
    assert sum(row["count"] for row in calibration(labels, probs).bins) == probs.size


def test_threshold_sweep_matches_a_brute_force_scan() -> None:
    rng = np.random.default_rng(6)
    labels = rng.integers(0, 2, 300)
    probs = np.round(rng.uniform(size=labels.size), 2)  # deliberate ties
    thresholds, sensitivity, specificity = sensitivity_specificity_curve(labels, probs)
    for threshold, sens, spec in zip(thresholds, sensitivity, specificity, strict=True):
        reference = binary_metrics(labels, probs, threshold=float(threshold))
        assert sens == pytest.approx(reference.sensitivity)
        assert spec == pytest.approx(reference.specificity)


def test_chosen_threshold_reaches_the_target_sensitivity() -> None:
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, 2000)
    probs = np.clip(rng.normal(0.5 + 0.25 * (labels * 2 - 1), 0.2), 0, 1)
    threshold = choose_threshold(labels, probs, target_sensitivity=0.95)
    assert binary_metrics(labels, probs, threshold=threshold).sensitivity >= 0.95


def test_an_unreachable_target_returns_the_most_sensitive_threshold() -> None:
    # A model that cannot reach the target must not return a threshold that
    # silently misses it.
    labels = [1, 1, 0, 0]
    probs = [0.4, 0.4, 0.4, 0.4]
    threshold = choose_threshold(labels, probs, target_sensitivity=1.0)
    assert binary_metrics(labels, probs, threshold=threshold).sensitivity == 1.0


def test_abstention_band_covers_the_uncertain_middle_and_respects_coverage() -> None:
    rng = np.random.default_rng(8)
    labels = rng.integers(0, 2, 5000)
    probs = np.clip(rng.normal(0.5 + 0.25 * (labels * 2 - 1), 0.2), 0, 1)
    band = choose_abstention_band(labels, probs, threshold=0.5, min_coverage=0.85)
    assert band.low < 0.5 < band.high
    assert band.coverage >= 0.85
    # The whole point: what the model abstains on is what it is worst at.
    answered = binary_metrics(
        labels[(probs < band.low) | (probs >= band.high)],
        probs[(probs < band.low) | (probs >= band.high)],
    )
    assert band.abstained_accuracy < answered.accuracy


def test_full_coverage_requirement_yields_no_abstention() -> None:
    rng = np.random.default_rng(9)
    labels = rng.integers(0, 2, 500)
    probs = rng.uniform(size=labels.size)
    band = choose_abstention_band(labels, probs, min_coverage=1.0)
    assert band.abstained == 0
    assert band.coverage == 1.0


def test_subgroup_metrics_split_a_confounded_model() -> None:
    # A model that only detects the cast: right whenever a cast is present.
    cast = np.r_[np.ones(100, dtype=bool), np.zeros(100, dtype=bool)]
    labels = np.r_[np.ones(100, dtype=int), np.random.default_rng(1).integers(0, 2, 100)]
    probs = np.where(cast, 0.95, 0.4)
    report = subgroup_metrics(labels, probs, {"cast": cast})
    assert report["cast"]["sensitivity"] == 1.0
    assert report["not_cast"]["sensitivity"] == 0.0


def test_subgroup_mask_shape_is_validated() -> None:
    with pytest.raises(ValueError, match="expected"):
        subgroup_metrics([0, 1], [0.1, 0.9], {"cast": [True]})


def test_evaluate_assembles_the_registry_report() -> None:
    rng = np.random.default_rng(10)
    labels = rng.integers(0, 2, 1000)
    probs = np.clip(rng.normal(0.5 + 0.2 * (labels * 2 - 1), 0.2), 0, 1)
    band = choose_abstention_band(labels, probs)
    report = evaluate(labels, probs, band=band, groups={"cast": rng.random(1000) < 0.3})
    assert {"overall", "calibration", "abstention", "subgroups"} <= report.keys()
    assert report["overall"]["confusion"]["tp"] >= 0
    assert "not_cast" in report["subgroups"]


@pytest.mark.parametrize(
    ("labels", "probs", "message"),
    [
        ([1, 0], [0.5], "shape mismatch"),
        ([], [], "empty"),
        ([2, 0], [0.5, 0.5], "must be 0 or 1"),
        ([1, 0], [1.5, 0.5], r"\[0, 1\]"),
    ],
)
def test_inputs_are_validated(labels, probs, message) -> None:
    with pytest.raises(ValueError, match=message):
        binary_metrics(labels, probs)
