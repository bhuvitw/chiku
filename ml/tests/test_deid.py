"""De-identification verification tests (System Design §4)."""

from __future__ import annotations

from PIL import Image, PngImagePlugin

from ml.data.deid import canonical_id, scan_filename, scan_image, strip_metadata


def _write_png(path, *, text: dict[str, str] | None = None):
    image = Image.new("L", (8, 8), color=128)
    info = PngImagePlugin.PngInfo()
    for key, value in (text or {}).items():
        info.add_text(key, value)
    image.save(path, pnginfo=info)
    return path


def test_free_text_metadata_is_reported(tmp_path) -> None:
    path = _write_png(tmp_path / "a.png", text={"Comment": "Patient: Jane Doe, MRN 123456789"})
    kinds = {finding.kind for finding in scan_image(path)}
    assert "metadata_text" in kinds


def test_clean_image_produces_no_findings(tmp_path) -> None:
    assert scan_image(_write_png(tmp_path / "0001.png")) == []


def test_stripping_metadata_clears_the_findings(tmp_path) -> None:
    dirty = _write_png(tmp_path / "dirty.png", text={"Artist": "Dr Smith"})
    assert scan_image(dirty)
    clean = strip_metadata(dirty, tmp_path / "out" / "clean.png")
    assert scan_image(clean) == []


def test_identifying_filenames_are_reported(tmp_path) -> None:
    assert scan_filename(tmp_path / "patient_jane_2019-03-04.png")
    # Underscore-separated: a \b-anchored rule misses this one entirely.
    assert {f.kind for f in scan_filename(tmp_path / "acc_123456789012.png")} == {
        "filename_long_number"
    }


def test_vendor_stems_are_flagged_and_cleared_by_renaming(tmp_path) -> None:
    # The GRAZPEDWRI-DX stem carries a masked timestamp plus sex and age, so it
    # is expected to trip the scan — renaming to canonical_id is what clears it.
    stem = "0001_1297860395_01_WRI-L1_M014"
    assert scan_filename(tmp_path / f"{stem}.png")
    assert scan_filename(tmp_path / f"{canonical_id(stem)}.png") == []


def test_canonical_id_is_stable_and_opaque() -> None:
    stem = "0001_1297860395_01_WRI-L1_M014"
    assert canonical_id(stem) == canonical_id(stem)
    assert canonical_id(stem) != canonical_id("0002_1297860395_01_WRI-L1_M014")
    assert len(canonical_id(stem)) == 16
    assert "M014" not in canonical_id(stem)
