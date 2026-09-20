"""The patient-leakage guard (System Design §23, implementation-plan §1.2).

This is a blocking test: if it fails, every metric the project reports is
invalid, so it must stay fast, dataset-independent and impossible to skip.
"""

from __future__ import annotations

import random

import pytest

from ml.data.manifest import ManifestRow
from ml.data.splits import (
    SPLIT_NAMES,
    PatientLeakageError,
    Split,
    patient_level_split,
)


def make_manifest(
    n_patients: int = 300,
    *,
    seed: int = 7,
) -> list[ManifestRow]:
    """A manifest shaped like GRAZPEDWRI-DX: most patients have several images."""
    rng = random.Random(seed)
    rows: list[ManifestRow] = []
    for index in range(n_patients):
        patient_id = f"p{index:04d}"
        patient_positive = rng.random() < 0.6
        for image in range(rng.choice([1, 2, 2, 2, 4, 6, 8])):
            rows.append(
                ManifestRow(
                    filestem=f"{patient_id}_img{image}",
                    patient_id=patient_id,
                    study_id=f"{patient_id}-1",
                    label=int(patient_positive and rng.random() < 0.8),
                    projection=str(1 + image % 2),
                    cast=rng.random() < 0.3,
                    metal=False,
                    osteopenia=False,
                    diagnosis_uncertain=False,
                    ao_classification="23r-M/2.1" if patient_positive else "",
                    age=rng.uniform(0.2, 19.0),
                    gender=rng.choice("MF"),
                )
            )
    return rows


def test_no_patient_appears_in_more_than_one_split() -> None:
    split = patient_level_split(make_manifest())
    assert split.train & split.val == frozenset()
    assert split.train & split.test == frozenset()
    assert split.val & split.test == frozenset()


def test_every_image_lands_in_exactly_one_split() -> None:
    manifest = make_manifest()
    split = patient_level_split(manifest)
    assigned = [row.filestem for name in SPLIT_NAMES for row in split.rows(name, manifest)]
    assert sorted(assigned) == sorted(row.filestem for row in manifest)
    assert len(assigned) == len(set(assigned))


def test_an_image_level_split_would_leak_on_this_dataset_shape() -> None:
    """Proves the rule is load-bearing here, not defensive boilerplate.

    If a future manifest ever made this pass, patient-level splitting would be
    merely redundant — on GRAZPEDWRI-DX it is not.
    """
    manifest = make_manifest()
    rng = random.Random(0)
    shuffled = list(manifest)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * 0.7)
    naive_train = {row.patient_id for row in shuffled[:cut]}
    naive_test = {row.patient_id for row in shuffled[cut:]}
    assert naive_train & naive_test, "expected image-level splitting to leak patients"


def test_leaked_split_is_rejected_at_construction() -> None:
    with pytest.raises(PatientLeakageError, match="both train and test"):
        Split(
            train=frozenset({"p1", "p2"}),
            val=frozenset({"p3"}),
            test=frozenset({"p2"}),
            seed=1,
            ratios=(0.7, 0.15, 0.15),
        )


def test_split_is_deterministic_for_a_seed_and_varies_across_seeds() -> None:
    manifest = make_manifest()
    assert patient_level_split(manifest, seed=11).test == patient_level_split(manifest, seed=11).test
    assert patient_level_split(manifest, seed=11).test != patient_level_split(manifest, seed=12).test


def test_row_order_does_not_change_the_split() -> None:
    manifest = make_manifest()
    reversed_manifest = list(reversed(manifest))
    assert patient_level_split(manifest).test == patient_level_split(reversed_manifest).test


def test_adding_images_for_an_existing_patient_does_not_move_them() -> None:
    """Split stability: the manifest grows, patients do not change folds."""
    manifest = make_manifest()
    before = patient_level_split(manifest)
    extra = [
        ManifestRow(
            filestem=f"{row.patient_id}_extra",
            patient_id=row.patient_id,
            study_id=f"{row.patient_id}-2",
            label=row.label,
            projection="2",
            cast=row.cast,
            metal=False,
            osteopenia=False,
            diagnosis_uncertain=False,
            ao_classification=row.ao_classification,
            age=row.age,
            gender=row.gender,
        )
        for row in manifest[:50]
    ]
    after = patient_level_split(manifest + extra)
    assert after.train == before.train
    assert after.test == before.test


def test_splits_carry_comparable_prevalence() -> None:
    manifest = make_manifest(n_patients=1200)
    split = patient_level_split(manifest)
    rates = [
        sum(row.label for row in split.rows(name, manifest)) / len(split.rows(name, manifest))
        for name in SPLIT_NAMES
    ]
    assert max(rates) - min(rates) < 0.05


def test_ratios_are_validated() -> None:
    manifest = make_manifest(n_patients=20)
    with pytest.raises(ValueError, match="summing to 1.0"):
        patient_level_split(manifest, ratios=(0.8, 0.15, 0.15))
    with pytest.raises(ValueError, match="positive share"):
        patient_level_split(manifest, ratios=(1.0, 0.0, 0.0))


def test_split_round_trips_through_json(tmp_path) -> None:
    split = patient_level_split(make_manifest(n_patients=40))
    path = tmp_path / "split.json"
    split.to_json(path)
    assert Split.from_json(path) == split
