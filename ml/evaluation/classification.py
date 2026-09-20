"""Evaluation for the binary X-ray classifier (System Design §24, PRD §11).

Deliberately a separate module from training: it takes arrays of labels and
predicted probabilities and knows nothing about how they were produced, so the
same code scores any checkpoint against the frozen test split. Nothing here
reads a training log — the numbers attached to a model registry entry are the
ones this module computed (System Design §20).

Three things beyond the usual metric list, all required by the plan:

- **Calibration**, because a confidence number the product displays has to mean
  something (PRD §11).
- **An abstention band** derived from the risk–coverage curve rather than a
  guessed round number (PRD §3 FR-04, implementation-plan §1.4).
- **Subgroup metrics**, because this dataset's cast confounder makes a single
  headline number misleading (see `data/README.md`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    threshold: float
    support: int
    positives: int
    tp: int
    fp: int
    tn: int
    fn: int
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    accuracy: float
    auroc: float | None
    auprc: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "threshold": self.threshold,
            "support": self.support,
            "positives": self.positives,
            "confusion": {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn},
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "auroc": self.auroc,
            "auprc": self.auprc,
        }


@dataclass(frozen=True, slots=True)
class Calibration:
    brier: float
    ece: float
    mce: float
    bins: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AbstentionBand:
    """Probabilities in [low, high) return `unable_to_assess` (PRD FR-04)."""

    low: float
    high: float
    coverage: float
    abstained: int
    selective_sensitivity: float
    selective_specificity: float
    abstained_accuracy: float


def _as_arrays(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=int).ravel()
    probabilities = np.asarray(y_prob, dtype=float).ravel()
    if labels.shape != probabilities.shape:
        raise ValueError(f"shape mismatch: {labels.shape} labels vs {probabilities.shape} probs")
    if labels.size == 0:
        raise ValueError("cannot evaluate an empty set")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return labels, probabilities


def binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> BinaryMetrics:
    labels, probabilities = _as_arrays(y_true, y_prob)
    predicted = probabilities >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    tn = int(np.sum(~predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))

    def ratio(numerator: int, denominator: int) -> float:
        # A subgroup with no positives has no sensitivity; 0.0 would read as a
        # failing model rather than an absent denominator, so report NaN.
        return numerator / denominator if denominator else float("nan")

    sensitivity = ratio(tp, tp + fn)
    precision = ratio(tp, tp + fp)
    both_classes_present = 0 < labels.sum() < labels.size
    return BinaryMetrics(
        threshold=threshold,
        support=int(labels.size),
        positives=int(labels.sum()),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        sensitivity=sensitivity,
        specificity=ratio(tn, tn + fp),
        precision=precision,
        f1=ratio(2 * tp, 2 * tp + fp + fn),
        accuracy=(tp + tn) / labels.size,
        auroc=float(roc_auc_score(labels, probabilities)) if both_classes_present else None,
        auprc=(
            float(average_precision_score(labels, probabilities)) if both_classes_present else None
        ),
    )


def calibration(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
) -> Calibration:
    """Reliability of the predicted probabilities.

    ECE is the support-weighted mean gap between confidence and accuracy across
    equal-width bins; MCE is the worst single bin. Both are reported because a
    model can have a low ECE while being badly wrong in the high-confidence bin
    that the product actually acts on.
    """
    labels, probabilities = _as_arrays(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Rightmost bin is closed so p == 1.0 does not fall outside every bin.
    indices = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, bins - 1)

    rows: list[dict[str, float]] = []
    ece = 0.0
    mce = 0.0
    for index in range(bins):
        mask = indices == index
        count = int(mask.sum())
        if not count:
            continue
        confidence = float(probabilities[mask].mean())
        observed = float(labels[mask].mean())
        gap = abs(confidence - observed)
        ece += gap * count / labels.size
        mce = max(mce, gap)
        rows.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_predicted": confidence,
                "observed_rate": observed,
                "gap": gap,
            }
        )
    return Calibration(
        brier=float(np.mean((probabilities - labels) ** 2)),
        ece=ece,
        mce=mce,
        bins=rows,
    )


def sensitivity_specificity_curve(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized sweep over every distinct threshold: (thresholds, sens, spec).

    One sort and two cumulative sums rather than a per-threshold recomputation
    — the naive loop is quadratic and takes minutes on a test split of this
    size, which is enough to discourage running the evaluation at all.
    """
    labels, probabilities = _as_arrays(y_true, y_prob)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("need both classes present to sweep thresholds")

    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    sorted_probs = probabilities[order]
    # Last index of each run of equal probabilities: `p >= t` must treat tied
    # scores as a single decision, not split them across two thresholds.
    cuts = np.r_[np.flatnonzero(np.diff(sorted_probs)), sorted_probs.size - 1]

    tp = np.cumsum(sorted_labels)[cuts]
    fp = np.cumsum(1 - sorted_labels)[cuts]
    return sorted_probs[cuts], tp / positives, 1.0 - fp / negatives


def choose_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    target_sensitivity: float = 0.95,
) -> float:
    """Lowest-cost threshold that still reaches a target sensitivity.

    Fractures are the costly miss, so the operating point is chosen by pinning
    sensitivity and taking the best specificity available at that constraint,
    rather than defaulting to 0.5. Selected on the **validation** split — using
    the test split here would make the reported test metrics self-fulfilling.
    """
    thresholds, sensitivity, specificity = sensitivity_specificity_curve(y_true, y_prob)
    feasible = sensitivity >= target_sensitivity
    if not feasible.any():
        # No threshold reaches the target: return the most sensitive one rather
        # than a threshold that quietly misses the requirement.
        return float(thresholds[np.argmax(sensitivity)])
    candidates = np.flatnonzero(feasible)
    return float(thresholds[candidates[np.argmax(specificity[candidates])]])


def choose_abstention_band(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
    min_coverage: float = 0.80,
    step: float = 0.01,
) -> AbstentionBand:
    """Widest band around the decision threshold that keeps coverage acceptable.

    The abstention state exists because a prediction near the decision boundary
    is close to a coin flip, and the product must say so rather than dress it up
    (PRD §3 FR-04, System Design §0.2). The band is grown symmetrically in
    probability until abstaining would drop coverage below `min_coverage`, so
    its width comes from the model's own error distribution.

    Selected on validation, applied unchanged at test time.
    """
    labels, probabilities = _as_arrays(y_true, y_prob)
    best = _band_stats(labels, probabilities, threshold, 0.0)
    width = step
    while True:
        low = max(0.0, threshold - width)
        high = min(1.0, threshold + width)
        candidate = _band_stats(labels, probabilities, threshold, width)
        if candidate.coverage < min_coverage:
            break
        best = candidate
        if low == 0.0 and high == 1.0:
            break
        width += step
    return best


def _band_stats(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    width: float,
) -> AbstentionBand:
    low = max(0.0, threshold - width)
    high = min(1.0, threshold + width)
    if width > 0:
        abstain = (probabilities >= low) & (probabilities < high)
    else:
        abstain = np.zeros_like(probabilities, dtype=bool)
    answered = ~abstain
    coverage = float(answered.mean())

    # Rates computed directly: binary_metrics would recompute AUROC on every
    # iteration of the band search, for a number this function never uses.
    kept_labels = labels[answered]
    predicted = probabilities[answered] >= threshold
    tp = int(np.sum(predicted & (kept_labels == 1)))
    fn = int(np.sum(~predicted & (kept_labels == 1)))
    tn = int(np.sum(~predicted & (kept_labels == 0)))
    fp = int(np.sum(predicted & (kept_labels == 0)))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")

    if abstain.any():
        abstained_predictions = probabilities[abstain] >= threshold
        abstained_accuracy = float((abstained_predictions == labels[abstain]).mean())
    else:
        abstained_accuracy = float("nan")

    return AbstentionBand(
        low=low,
        high=high,
        coverage=coverage,
        abstained=int(abstain.sum()),
        selective_sensitivity=sensitivity,
        selective_specificity=specificity,
        abstained_accuracy=abstained_accuracy,
    )


def subgroup_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    groups: Mapping[str, Sequence[bool] | np.ndarray],
    *,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Metrics split by each boolean subgroup mask.

    The cast subgroup is the one that matters on GRAZPEDWRI-DX: if performance
    collapses on images without a cast, the model learned the treatment, not
    the fracture.
    """
    labels, probabilities = _as_arrays(y_true, y_prob)
    report: dict[str, dict] = {}
    for name, raw_mask in groups.items():
        mask = np.asarray(raw_mask, dtype=bool).ravel()
        if mask.shape != labels.shape:
            raise ValueError(f"group {name!r} has shape {mask.shape}, expected {labels.shape}")
        for suffix, selected in ((name, mask), (f"not_{name}", ~mask)):
            if not selected.any():
                continue
            report[suffix] = binary_metrics(
                labels[selected], probabilities[selected], threshold=threshold
            ).as_dict()
    return report


def evaluate(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
    band: AbstentionBand | None = None,
    groups: Mapping[str, Sequence[bool] | np.ndarray] | None = None,
    bins: int = 10,
) -> dict:
    """The full report attached to a model registry entry."""
    report: dict = {
        "overall": binary_metrics(y_true, y_prob, threshold=threshold).as_dict(),
        "calibration": asdict(calibration(y_true, y_prob, bins=bins)),
    }
    if band is not None:
        report["abstention"] = asdict(band)
    if groups:
        report["subgroups"] = subgroup_metrics(y_true, y_prob, groups, threshold=threshold)
    return report
