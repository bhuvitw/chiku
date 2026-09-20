"""Canonical image manifest for the GRAZPEDWRI-DX wrist X-ray dataset.

The manifest is the single source of truth every downstream step reads: the
split (`ml.data.splits`), preprocessing, training and evaluation all consume
`ManifestRow` objects and never re-parse the vendor CSV themselves. Label
policy therefore lives here and nowhere else.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Columns the loader requires. A vendor CSV missing any of these is a different
# dataset revision than the one this pipeline was written against, and we fail
# rather than silently produce a manifest with wrong labels.
REQUIRED_COLUMNS = frozenset(
    {
        "filestem",
        "patient_id",
        "study_number",
        "projection",
        "fracture_visible",
        "ao_classification",
        "diagnosis_uncertain",
        "cast",
        "metal",
        "osteopenia",
        "age",
        "gender",
    }
)


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """One radiograph. `patient_id` is what the split partitions on."""

    filestem: str
    patient_id: str
    study_id: str
    label: int
    projection: str
    # Confounders, carried through to evaluation so metrics can be reported
    # stratified by them (a cast is visible on 28% of images and correlates
    # heavily with fracture — a model that learns "cast ⇒ fracture" scores well
    # overall and is useless on the pre-treatment images that actually matter).
    cast: bool
    metal: bool
    osteopenia: bool
    diagnosis_uncertain: bool
    ao_classification: str
    age: float | None
    gender: str

    @property
    def filename(self) -> str:
        return f"{self.filestem}.png"


def _flag(value: str) -> bool:
    # The vendor CSV encodes booleans as "1" or empty.
    return value.strip() == "1"


def load_manifest(
    csv_path: Path | str,
    *,
    drop_uncertain: bool = True,
) -> list[ManifestRow]:
    """Read the vendor `dataset.csv` into the canonical manifest.

    Label policy (fixed here so every experiment shares it):

    - `label = 1` iff `fracture_visible` is set — the per-image question the
      classifier is actually being asked.
    - Images carrying an `ao_classification` but no `fracture_visible` flag
      (773 of 20,327) stay **negative**. The fracture was diagnosed for the
      study, but is not visible in this projection; calling it positive would
      train the model to predict from a view with no evidence in it.
    - `drop_uncertain` removes the 537 rows the radiologists flagged as
      uncertain. They are excluded by default rather than guessed at, per the
      fail-closed principle (System Design §0.2); pass False to keep them.
    """
    path = Path(csv_path)
    # utf-8-sig: the vendor CSV ships with a BOM, which would otherwise end up
    # inside the first column name.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"{path} is missing expected columns: {sorted(missing)}. "
                "This is not the dataset revision the pipeline was built for."
            )
        rows = [_parse_row(record) for record in reader]

    if drop_uncertain:
        rows = [row for row in rows if not row.diagnosis_uncertain]
    return rows


def _parse_row(record: dict[str, str]) -> ManifestRow:
    age = record["age"].strip()
    return ManifestRow(
        filestem=record["filestem"].strip(),
        patient_id=record["patient_id"].strip(),
        study_id=f"{record['patient_id'].strip()}-{record['study_number'].strip()}",
        label=int(_flag(record["fracture_visible"])),
        projection=record["projection"].strip(),
        cast=_flag(record["cast"]),
        metal=_flag(record["metal"]),
        osteopenia=_flag(record["osteopenia"]),
        diagnosis_uncertain=_flag(record["diagnosis_uncertain"]),
        ao_classification=record["ao_classification"].strip(),
        age=float(age) if age else None,
        gender=record["gender"].strip(),
    )


def patient_ids(rows: Iterable[ManifestRow]) -> set[str]:
    return {row.patient_id for row in rows}


def patient_is_positive(rows: Iterable[ManifestRow]) -> dict[str, bool]:
    """Patient-level label: does this patient have any image with a fracture?

    Used to stratify the patient-level split, since stratifying on image labels
    would require splitting a patient across folds — the one thing we never do.
    """
    positive: dict[str, bool] = {}
    for row in rows:
        positive[row.patient_id] = positive.get(row.patient_id, False) or bool(row.label)
    return positive
