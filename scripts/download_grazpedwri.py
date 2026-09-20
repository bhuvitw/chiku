"""Fetch GRAZPEDWRI-DX from figshare (implementation-plan §1.1).

~16 GB across four zips, so downloads resume rather than restart and every
file is checked against the MD5 figshare publishes before it is extracted.

    python -m scripts.download_grazpedwri --labels-only   # 1.8 MB, instant
    python -m scripts.download_grazpedwri                 # full, hours
    python -m scripts.download_grazpedwri --extract
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ARTICLE_API = "https://api.figshare.com/v2/articles/14825193/files"
DEFAULT_ROOT = Path("data/raw/grazpedwri-dx")
LABEL_FILES = {"dataset.csv", "folder_structure.zip"}
CHUNK = 1 << 20


def list_files() -> list[dict]:
    with urllib.request.urlopen(ARTICLE_API, timeout=60) as response:
        return json.load(response)


def md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - figshare publishes MD5, not our choice
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry: dict, root: Path) -> Path:
    destination = root / entry["name"]
    expected = entry["size"]
    if destination.exists() and destination.stat().st_size == expected:
        print(f"  {entry['name']}: already complete")
        return destination

    # Resume a partial file rather than re-pulling gigabytes over a laptop link.
    start = destination.stat().st_size if destination.exists() else 0
    request = urllib.request.Request(entry["download_url"])
    if start:
        request.add_header("Range", f"bytes={start}-")
        print(f"  {entry['name']}: resuming at {start / 1e9:.2f} GB")

    with urllib.request.urlopen(request, timeout=120) as response:
        if start and response.status != 206:
            # Server ignored the range request; start over rather than append
            # to a partial file and produce a corrupt archive.
            print("  server does not support resume, restarting")
            start = 0
        mode = "ab" if start else "wb"
        done = start
        with destination.open(mode) as handle:
            while block := response.read(CHUNK):
                handle.write(block)
                done += len(block)
                print(f"\r  {entry['name']}: {done / 1e9:.2f}/{expected / 1e9:.2f} GB", end="")
    print()
    return destination


def verify(path: Path, entry: dict) -> bool:
    if path.stat().st_size != entry["size"]:
        print(f"  {path.name}: SIZE MISMATCH — delete and re-run")
        return False
    print(f"  {path.name}: checksumming…")
    if md5(path) != entry["supplied_md5"]:
        print(f"  {path.name}: CHECKSUM MISMATCH — delete and re-run")
        return False
    return True


def extract(path: Path, root: Path) -> None:
    target = root / "images"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(target)
    print(f"  {path.name}: extracted to {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--labels-only", action="store_true", help="dataset.csv only")
    parser.add_argument("--extract", action="store_true", help="unzip after verifying")
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    entries = list_files()
    if args.labels_only:
        entries = [entry for entry in entries if entry["name"] in LABEL_FILES]

    total = sum(entry["size"] for entry in entries)
    free = shutil.disk_usage(args.root).free
    needed = total * 2 if args.extract else total
    print(f"{len(entries)} file(s), {total / 1e9:.1f} GB; free disk {free / 1e9:.1f} GB")
    if free < needed:
        raise SystemExit(
            f"need ~{needed / 1e9:.1f} GB (archives plus extracted images) "
            f"but only {free / 1e9:.1f} GB is free"
        )

    for entry in entries:
        path = download(entry, args.root)
        if not verify(path, entry):
            raise SystemExit(1)
        if args.extract and path.suffix == ".zip":
            extract(path, args.root)
            if not args.keep_archives:
                path.unlink()
                print(f"  {path.name}: archive removed")

    print("\nDone. Next: python -m scripts.build_xray_cache")


if __name__ == "__main__":
    main()
