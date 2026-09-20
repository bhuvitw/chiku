"""Preprocess the X-ray set once into a compact uint8 cache.

This is the job the laptop CPU is genuinely good at, and it is what makes
training elsewhere cheap: 20,327 full-resolution radiographs (~16 GB of PNG)
become a single ~3 GB uint8 memmap that loads with no per-epoch decode cost and
is small enough to upload to a rented GPU.

Because the cache *is* the model's input, it is built with the inference-path
preprocessing in `ml.preprocessing.xray` — no augmentation (System Design §22)
— and stamped with the preprocessing version, so a cache built under one config
can never be silently trained against another.

    python -m scripts.build_xray_cache --resolution 384
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ml.data.deid import canonical_id
from ml.data.manifest import ManifestRow, load_manifest
from ml.data.splits import SPLIT_NAMES, Split, patient_level_split, split_summary
from ml.preprocessing.xray import PreprocessConfig, preprocess

_CONFIG: PreprocessConfig | None = None
_IMAGE_ROOT: Path | None = None


def _init_worker(config: PreprocessConfig, image_root: Path) -> None:
    global _CONFIG, _IMAGE_ROOT
    _CONFIG = config
    _IMAGE_ROOT = image_root
    # Each worker is one image at a time; letting every one of them spawn a
    # full thread pool oversubscribes the 8 available threads and runs slower.
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _process(item: tuple[int, str]) -> tuple[int, np.ndarray | None, str | None]:
    index, filename = item
    assert _CONFIG is not None and _IMAGE_ROOT is not None
    path = _IMAGE_ROOT / filename
    if not path.exists():
        return index, None, "missing file"
    try:
        image, report = preprocess(path, _CONFIG)
    except Exception as error:  # noqa: BLE001 - one bad file must not kill the build
        return index, None, f"{type(error).__name__}: {error}"
    return index, image, None if report.passed else report.reason


def find_images(image_root: Path, rows: list[ManifestRow]) -> dict[str, str]:
    """Map filestem → path relative to `image_root`.

    The archives extract into a nested folder structure, so the file is located
    by stem rather than assuming a flat layout.
    """
    located = {path.stem: str(path.relative_to(image_root)) for path in image_root.rglob("*.png")}
    return {row.filestem: located[row.filestem] for row in rows if row.filestem in located}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/grazpedwri-dx/dataset.csv"))
    parser.add_argument("--images", type=Path, default=Path("data/raw/grazpedwri-dx/images"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--keep-uncertain", action="store_true")
    args = parser.parse_args()

    out = args.out or Path(f"data/processed/xray-{args.resolution}")
    out.mkdir(parents=True, exist_ok=True)
    config = PreprocessConfig(resolution=args.resolution)

    rows = load_manifest(args.manifest, drop_uncertain=not args.keep_uncertain)
    print(f"manifest: {len(rows)} images")

    located = find_images(args.images, rows)
    missing = [row for row in rows if row.filestem not in located]
    if missing:
        print(f"  {len(missing)} manifest rows have no image on disk (e.g. {missing[0].filestem})")
    rows = [row for row in rows if row.filestem in located]
    if not rows:
        raise SystemExit(f"no images found under {args.images} — run download_grazpedwri.py first")

    split = patient_level_split(rows, seed=args.seed)
    split.to_json(out / "split.json")
    assignment = {
        patient: name for name in SPLIT_NAMES for patient in split.patients(name)
    }
    print(json.dumps(split_summary(split, rows), indent=2))

    images = np.lib.format.open_memmap(
        out / "images.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(rows), args.resolution, args.resolution),
    )

    rejected: list[dict[str, str]] = []
    work = [(index, located[row.filestem]) for index, row in enumerate(rows)]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(config, args.images),
    ) as pool:
        for done, (index, image, reason) in enumerate(pool.map(_process, work, chunksize=16), 1):
            if image is None:
                rejected.append({"filestem": rows[index].filestem, "reason": reason or "unknown"})
            else:
                images[index] = image
            if done % 250 == 0 or done == len(work):
                print(f"\r  {done}/{len(work)} processed, {len(rejected)} rejected", end="")
    print()
    images.flush()

    rejected_stems = {entry["filestem"] for entry in rejected}
    index_records = [
        {
            "row": position,
            "id": canonical_id(row.filestem),
            "patient_id": row.patient_id,
            "study_id": row.study_id,
            "split": assignment[row.patient_id],
            "label": row.label,
            "projection": row.projection,
            "cast": row.cast,
            "metal": row.metal,
            "osteopenia": row.osteopenia,
            "usable": row.filestem not in rejected_stems,
        }
        for position, row in enumerate(rows)
    ]
    (out / "index.json").write_text(json.dumps(index_records), encoding="utf-8")
    (out / "rejected.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "dataset": "GRAZPEDWRI-DX",
                "source_manifest": str(args.manifest),
                "preprocess": asdict(config),
                "split_seed": args.seed,
                "images": len(rows),
                "rejected": len(rejected),
                "drop_uncertain": not args.keep_uncertain,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    size_gb = (out / "images.npy").stat().st_size / 1e9
    print(f"\nwrote {out}/images.npy ({size_gb:.2f} GB), {len(rejected)} rejected")
    print("Next: dvc add data/processed && git add data/processed.dvc")


if __name__ == "__main__":
    main()
