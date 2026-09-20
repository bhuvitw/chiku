"""Patient-level train/val/test splitting (System Design §23, PRD §10).

The single hard rule of this project: a patient's images live in exactly one
split. GRAZPEDWRI-DX makes this non-negotiable — 98.4% of its 6,091 patients
contribute more than one image (mean 3.3, max 30), so an image-level random
split would put the same wrist in train and test and inflate every metric in
`ml.evaluation`.

The rule is enforced twice: the split is *constructed* over patient IDs, and
`assert_no_patient_leakage` re-checks the result at runtime — not only in the
test suite, so a caller that builds a split some other way still trips it.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ml.data.manifest import ManifestRow, patient_is_positive

SplitName = str
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "test")


class PatientLeakageError(AssertionError):
    """Raised when a patient appears in more than one split."""


@dataclass(frozen=True, slots=True)
class Split:
    """A frozen assignment of patient IDs to splits.

    Stores patient IDs rather than rows so the split is small, diffable and
    stable across manifest revisions: adding images for an existing patient
    must not move that patient to another fold.
    """

    train: frozenset[str]
    val: frozenset[str]
    test: frozenset[str]
    seed: int
    ratios: tuple[float, float, float]

    def __post_init__(self) -> None:
        assert_no_patient_leakage(self)

    def patients(self, name: SplitName) -> frozenset[str]:
        if name not in SPLIT_NAMES:
            raise KeyError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
        return getattr(self, name)

    def rows(self, name: SplitName, manifest: Iterable[ManifestRow]) -> list[ManifestRow]:
        """Materialize the image-level rows for one split."""
        members = self.patients(name)
        return [row for row in manifest if row.patient_id in members]

    def to_json(self, path: Path | str) -> None:
        payload = {
            "seed": self.seed,
            "ratios": list(self.ratios),
            **{name: sorted(self.patients(name)) for name in SPLIT_NAMES},
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path | str) -> Split:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        ratios = tuple(payload["ratios"])
        return cls(
            train=frozenset(payload["train"]),
            val=frozenset(payload["val"]),
            test=frozenset(payload["test"]),
            seed=int(payload["seed"]),
            ratios=(ratios[0], ratios[1], ratios[2]),
        )


def assert_no_patient_leakage(split: Split) -> None:
    """Fail loudly if any patient ID appears in more than one split."""
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = split.patients(left) & split.patients(right)
        if shared:
            sample = sorted(shared)[:5]
            raise PatientLeakageError(
                f"{len(shared)} patient(s) appear in both {left} and {right} "
                f"(e.g. {sample}). Every metric computed on this split is invalid."
            )


def patient_level_split(
    manifest: Sequence[ManifestRow],
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 20260920,
) -> Split:
    """Partition patients — never images — into train/val/test.

    Stratified on the patient-level label (does this patient have any image
    with a visible fracture) so all three folds carry a comparable prevalence.
    Deterministic for a given manifest and seed: patients are sorted before
    shuffling, so dict/CSV ordering cannot change the result.
    """
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must be three fractions summing to 1.0, got {ratios}")
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError(f"every split needs a positive share, got {ratios}")

    positive = patient_is_positive(manifest)
    buckets: dict[bool, list[str]] = {True: [], False: []}
    for patient, is_positive in sorted(positive.items()):
        buckets[is_positive].append(patient)

    assigned: dict[SplitName, set[str]] = {name: set() for name in SPLIT_NAMES}
    rng = random.Random(seed)
    for stratum in (True, False):
        patients = buckets[stratum]
        rng.shuffle(patients)
        n_train = round(len(patients) * ratios[0])
        n_val = round(len(patients) * ratios[1])
        assigned["train"].update(patients[:n_train])
        assigned["val"].update(patients[n_train : n_train + n_val])
        assigned["test"].update(patients[n_train + n_val :])

    return Split(
        train=frozenset(assigned["train"]),
        val=frozenset(assigned["val"]),
        test=frozenset(assigned["test"]),
        seed=seed,
        ratios=ratios,
    )


def split_summary(split: Split, manifest: Sequence[ManifestRow]) -> dict[str, dict[str, float]]:
    """Per-split counts and prevalence, for logging alongside every run."""
    summary: dict[str, dict[str, float]] = {}
    for name in SPLIT_NAMES:
        rows = split.rows(name, manifest)
        positives = sum(row.label for row in rows)
        summary[name] = {
            "patients": len(split.patients(name)),
            "images": len(rows),
            "positives": positives,
            "prevalence": positives / len(rows) if rows else 0.0,
            "cast_fraction": sum(row.cast for row in rows) / len(rows) if rows else 0.0,
        }
    return summary
