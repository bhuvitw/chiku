"""Torch dataset over the preprocessed X-ray cache.

Reads the memmap written by `scripts/build_xray_cache.py`, so no PNG decoding
happens per epoch. Augmentation is applied here and only for the training
split (System Design §22); val and test go through untouched, exactly as the
inference path will see them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.preprocessing.xray import to_model_tensor
from ml.xray.augment import AugmentConfig, augment


class XrayCacheDataset(Dataset):
    def __init__(
        self,
        root: Path | str,
        split: str,
        *,
        augment_config: AugmentConfig | None = None,
        seed: int = 0,
        limit: int | None = None,
    ) -> None:
        if augment_config is not None and split != "train":
            raise ValueError(
                f"augmentation requested for the {split!r} split; it belongs to "
                "training only (System Design §22)"
            )
        self.root = Path(root)
        self.split = split
        self.augment_config = augment_config
        self.seed = seed

        records = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        self.records = [r for r in records if r["split"] == split and r["usable"]]
        if limit is not None:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f"no usable rows for split {split!r} in {self.root}")
        self.images = np.load(self.root / "images.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = np.asarray(self.images[record["row"]])
        if self.augment_config is not None:
            # Seeded per (epoch-independent) sample index so a run is
            # reproducible while still varying across samples.
            rng = np.random.default_rng((self.seed, index, torch.initial_seed() % (1 << 31)))
            image = augment(image, self.augment_config, rng)
        tensor = torch.from_numpy(to_model_tensor(image))
        return tensor, torch.tensor(record["label"], dtype=torch.long)

    @property
    def labels(self) -> np.ndarray:
        return np.array([record["label"] for record in self.records], dtype=int)

    def group_mask(self, field: str) -> np.ndarray:
        return np.array([bool(record[field]) for record in self.records], dtype=bool)

    @property
    def patient_ids(self) -> list[str]:
        return [record["patient_id"] for record in self.records]
