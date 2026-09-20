"""Label-policy tests for the GRAZPEDWRI-DX manifest loader."""

from __future__ import annotations

import pytest

from ml.data.manifest import load_manifest, patient_is_positive

HEADER = (
    "filestem,patient_id,study_number,timehash,gender,age,laterality,projection,"
    "initial_exam,ao_classification,cast,diagnosis_uncertain,osteopenia,"
    "fracture_visible,metal,pixel_spacing,device_manufacturer\n"
)
ROWS = (
    # AO classification present, fracture not visible in this projection.
    "0001_a_01_WRI-L1_M014,1,1,1,M,14.1,L,1,1,23r-M/2.1,,,,,,0.144,Siemens\n"
    "0001_b_01_WRI-L2_M014,1,1,2,M,14.1,L,2,1,23r-M/2.1,,,,1,,0.144,Siemens\n"
    # Radiologist flagged the diagnosis as uncertain.
    "0002_a_01_WRI-R1_F012,2,1,3,F,12,R,1,1,,,1,,1,,0.144,Siemens\n"
    "0003_a_01_WRI-R1_F009,3,1,4,F,9,R,1,1,,1,,,,1,0.144,Agfa\n"
)


@pytest.fixture()
def csv_path(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text(HEADER + ROWS, encoding="utf-8")
    return path


def test_ao_classification_without_visible_fracture_stays_negative(csv_path) -> None:
    rows = {row.filestem: row for row in load_manifest(csv_path)}
    assert rows["0001_a_01_WRI-L1_M014"].label == 0
    assert rows["0001_a_01_WRI-L1_M014"].ao_classification == "23r-M/2.1"
    assert rows["0001_b_01_WRI-L2_M014"].label == 1


def test_uncertain_rows_are_dropped_by_default_and_keepable(csv_path) -> None:
    assert all(not row.diagnosis_uncertain for row in load_manifest(csv_path))
    kept = load_manifest(csv_path, drop_uncertain=False)
    assert sum(row.diagnosis_uncertain for row in kept) == 1


def test_confounder_flags_are_carried_through(csv_path) -> None:
    rows = {row.filestem: row for row in load_manifest(csv_path, drop_uncertain=False)}
    assert rows["0003_a_01_WRI-R1_F009"].cast is True
    assert rows["0003_a_01_WRI-R1_F009"].metal is True
    assert rows["0001_a_01_WRI-L1_M014"].cast is False


def test_patient_level_label_is_any_positive_image(csv_path) -> None:
    positive = patient_is_positive(load_manifest(csv_path))
    assert positive["1"] is True  # one of the two projections shows the fracture
    assert positive["3"] is False


def test_a_csv_missing_columns_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("filestem,patient_id\na,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing expected columns"):
        load_manifest(path)
