"""De-identification verification for imported datasets (System Design §4).

GRAZPEDWRI-DX ships pseudonymized, and the plan's instruction is explicit:
verify, don't trust. Two things survive "anonymization" in practice — metadata
chunks the exporter forgot to strip, and identifiers baked into filenames — so
this module checks both and strips what it can.

It deliberately does *not* claim to detect burned-in pixel annotations; that
needs a visual review pass, which `report_findings` flags as an outstanding
manual step rather than silently passing.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# Chunks that can carry free text (and therefore names, MRNs, institutions).
_TEXT_CHUNKS = ("tEXt", "iTXt", "zTXt", "Comment", "Description", "Software", "Artist")

# Patterns that should never appear in a filename or a metadata value.
# None of these use \b: an underscore is a word character, so "patient_jane" and
# "acc_123456789012" both slip past \b-anchored rules and the whole scan goes
# silently inert on exactly the filenames it exists to catch.
_IDENTIFIER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "date",
        re.compile(r"(?<!\d)(19|20)\d{2}[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])(?!\d)"),
    ),
    ("long_number", re.compile(r"(?<!\d)\d{9,}(?!\d)")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    (
        "name_like",
        re.compile(r"(?<![A-Za-z])(patient|name|mrn|dob|birth|accession)(?![A-Za-z])", re.I),
    ),
)


@dataclass(frozen=True, slots=True)
class Finding:
    path: Path
    kind: str
    detail: str


def scan_image(path: Path) -> list[Finding]:
    """Report identifying metadata found in one image file."""
    findings: list[Finding] = []
    with Image.open(path) as image:
        metadata = {key: str(value) for key, value in (image.info or {}).items()}
    for key, value in metadata.items():
        if key in _TEXT_CHUNKS or any(chunk.lower() in key.lower() for chunk in _TEXT_CHUNKS):
            findings.append(Finding(path, "metadata_text", f"{key}={value[:80]}"))
            continue
        for kind, pattern in _IDENTIFIER_PATTERNS:
            if pattern.search(value):
                findings.append(Finding(path, f"metadata_{kind}", f"{key}={value[:80]}"))
    findings.extend(scan_filename(path))
    return findings


def scan_filename(path: Path) -> list[Finding]:
    """Report identifying patterns in a filename.

    The GRAZPEDWRI-DX stem (`0001_1297860395_01_WRI-L1_M014`) encodes a
    sequential patient index, a masked timestamp, sex and age. Sex and age are
    quasi-identifiers that belong in the manifest, not in a filename that ends
    up in logs and object-storage keys. Vendor stems are therefore *expected*
    to be flagged here: the finding is cleared by renaming every processed
    artefact to `canonical_id`, not by whitelisting the pattern.
    """
    findings: list[Finding] = []
    for kind, pattern in _IDENTIFIER_PATTERNS:
        if pattern.search(path.stem):
            findings.append(Finding(path, f"filename_{kind}", path.stem))
    return findings


def strip_metadata(source: Path, destination: Path) -> Path:
    """Re-encode an image with every metadata chunk dropped."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        clean = Image.new(image.mode, image.size)
        clean.putdata(list(image.getdata()))
        clean.save(destination, format="PNG")
    return destination


def scan_directory(root: Path | str, *, pattern: str = "*.png") -> list[Finding]:
    return [finding for path in sorted(Path(root).rglob(pattern)) for finding in scan_image(path)]


def report_findings(findings: Iterable[Finding]) -> str:
    findings = list(findings)
    lines = [
        f"de-identification scan: {len(findings)} finding(s)",
        "",
        "NOT covered by this scan — burned-in pixel annotations (patient name or "
        "date rendered into the image itself) still require a manual review of a "
        "sample before the dataset is considered clean.",
    ]
    if findings:
        lines.append("")
        by_kind: dict[str, int] = {}
        for finding in findings:
            by_kind[finding.kind] = by_kind.get(finding.kind, 0) + 1
        lines.extend(f"  {kind}: {count}" for kind, count in sorted(by_kind.items()))
        lines.append("")
        lines.extend(f"  {f.path.name}: {f.kind} — {f.detail}" for f in findings[:10])
    return "\n".join(lines)


def canonical_id(filestem: str) -> str:
    """Opaque, stable ID for a processed artefact.

    Processed images, cache entries and object-storage keys are named with this
    rather than the vendor stem, so no quasi-identifier (sex, age, masked
    timestamp) travels into logs or storage. Deterministic, so re-running the
    pipeline does not churn every filename.
    """
    return hashlib.sha256(filestem.encode("utf-8")).hexdigest()[:16]
