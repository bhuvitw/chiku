"""The Phase 1 checkpoint, served on CPU.

This is the only module in the backend that imports torch, and it is imported
only when `settings.inference_backend == "torch"` — so the API image stays
free of a dependency that belongs to the worker.

Two properties matter more than anything else here:

- **Preprocessing comes from the checkpoint, not from settings.** A model
  trained at 224 and served at 384 still runs, still returns a confident
  number, and is quietly wrong. The checkpoint carries the exact
  `PreprocessConfig` it was trained under (`ml/xray/train.py` stores it for
  this reason) and that is what gets used.
- **No augmentation, ever.** `ml.preprocessing.xray` is the inference path by
  construction; `ml.xray.augment` is not imported here and must never be
  (System Design §22).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from backend.inference.base import ImageQualityError
from backend.inference.operating_point import resolve_path


def _checkpoint_fingerprint(path: Path) -> str:
    digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    return digest[:8]


class TorchPredictor:
    """Loads once, predicts many times.

    Construction is expensive — 45 MB read, hashed, and deserialized — so
    callers go through `load_predictor`, which caches per checkpoint path.
    """

    def __init__(self, checkpoint_path: Path | str) -> None:
        import torch

        from ml.preprocessing.xray import PreprocessConfig
        from ml.xray.model import build_model

        path = resolve_path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"model checkpoint {path} is missing. It is DVC-tracked: run "
                "`dvc pull experiments/outputs/resnet18-best.pt`."
            )

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config") or {}
        backbone = config.get("backbone", "resnet18")

        # `pretrained=False`: the weights about to be loaded are the whole
        # point, and the ImageNet download would be a 45 MB network fetch on
        # the first prediction of a cold worker.
        model = build_model(backbone, pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        self._torch = torch
        self._model = model
        self._preprocess = PreprocessConfig(**checkpoint["preprocess"])
        self.checkpoint_path = path
        self.dataset = checkpoint.get("dataset")
        self.split_seed = checkpoint.get("split_seed")
        # Identifies the exact file, not just the architecture: two resnet18
        # checkpoints are not interchangeable, and a stored prediction has to
        # say which one produced it (System Design §20-21).
        self.model_version = (
            f"{backbone}/{self._preprocess.version}@{_checkpoint_fingerprint(path)}"
        )

    @property
    def resolution(self) -> int:
        return self._preprocess.resolution

    def probability(self, image_path: Path) -> float:
        """P(fracture) for one image.

        Raises `ImageQualityError` when preprocessing's quality gate rejects
        the image, which the analysis service maps to LOW_IMAGE_QUALITY. The
        gate returns `None` for the image in that case, so there is no way to
        run the model on something it refused (PRD §8, fail closed).
        """
        from ml.preprocessing.xray import preprocess, to_model_tensor
        from ml.xray.model import predict_proba

        image, report = preprocess(image_path, self._preprocess)
        if image is None or not report.passed:
            raise ImageQualityError(report.reason)

        batch = self._torch.from_numpy(to_model_tensor(image)).unsqueeze(0)
        return float(predict_proba(self._model, batch)[0])


@lru_cache(maxsize=2)
def load_predictor(checkpoint_path: str) -> TorchPredictor:
    """Cached per path, so a worker pays the load cost once per process."""
    return TorchPredictor(checkpoint_path)
