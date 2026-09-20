"""X-ray inference preprocessing (System Design §5).

    X-ray → quality check → orientation normalization → resize
          → intensity normalization → model

This module is the **inference** path, so it contains no augmentation of any
kind (System Design §22). Augmentation lives in `ml.xray.augment` and is
applied only inside the training loop; mixing the two would make the same image
produce different predictions on repeated analysis of the same study.

Everything here is deterministic — `test_preprocessing.py` asserts byte-identical
output across repeat calls, which is what keeps that guarantee from rotting.

The module deliberately does not import from `backend`: it runs inside the ML
worker and must stay independent of the API process. `QualityReport.passed`
is what the Phase 2 adapter maps onto `ErrorCode.LOW_IMAGE_QUALITY`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

#: Training used ImageNet-pretrained backbones, so inference must match their
#: normalization exactly or the features shift.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Pinned alongside every model version in the registry (System Design §20).

    A checkpoint evaluated under one config and served under another is a
    silent accuracy regression, so this is part of the model's identity.
    """

    resolution: int = 384
    clip_percentiles: tuple[float, float] = (0.5, 99.5)
    min_source_pixels: int = 128
    min_dynamic_range: float = 0.05
    version: str = "xray-v1"


@dataclass(frozen=True, slots=True)
class QualityReport:
    passed: bool
    reason: str | None = None
    dynamic_range: float = 0.0


def load_grayscale(path: Path | str) -> np.ndarray:
    """Read an image as single-channel float32 in [0, 1].

    Radiographs are 8- or 16-bit grayscale; PIL's "I;16" mode does not convert
    to "L" without losing the high byte, so 16-bit input is rescaled explicitly.
    """
    with Image.open(path) as image:
        if image.mode in ("I;16", "I;16B", "I;16L", "I"):
            array = np.asarray(image, dtype=np.float32)
            peak = float(array.max())
            return array / peak if peak > 0 else array
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def assess_quality(image: np.ndarray, config: PreprocessConfig) -> QualityReport:
    """Fail closed on images the model should not be asked to judge (§0.2)."""
    height, width = image.shape[:2]
    if min(height, width) < config.min_source_pixels:
        return QualityReport(False, f"image is {width}x{height}, below the minimum")
    low, high = np.percentile(image, config.clip_percentiles)
    dynamic_range = float(high - low)
    if dynamic_range < config.min_dynamic_range:
        return QualityReport(False, "image is nearly uniform", dynamic_range)
    return QualityReport(True, None, dynamic_range)


def normalize_intensity(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Percentile-clip then rescale to [0, 1].

    Percentiles rather than min/max: a single saturated pixel (a marker, a
    collimation edge) would otherwise compress the whole useful range.
    """
    low, high = np.percentile(image, config.clip_percentiles)
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def resize_letterbox(image: np.ndarray, resolution: int) -> np.ndarray:
    """Resize to a square, preserving aspect ratio and padding the remainder.

    Stretching to a square would distort bone geometry, and fracture lines are
    a geometric cue — so the image is scaled and centred on a black canvas.
    """
    height, width = image.shape[:2]
    scale = resolution / max(height, width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = Image.fromarray(image).resize(new_size, Image.Resampling.BILINEAR)
    canvas = np.zeros((resolution, resolution), dtype=image.dtype)
    top = (resolution - new_size[1]) // 2
    left = (resolution - new_size[0]) // 2
    canvas[top : top + new_size[1], left : left + new_size[0]] = np.asarray(resized)
    return canvas


def preprocess(
    path: Path | str,
    config: PreprocessConfig | None = None,
) -> tuple[np.ndarray | None, QualityReport]:
    """Full inference-path preprocessing. Returns (uint8 square image, report).

    The image is `None` when the quality gate rejects it, so a caller cannot
    accidentally run the model on something this stage refused.
    """
    config = config or PreprocessConfig()
    image = load_grayscale(path)
    report = assess_quality(image, config)
    if not report.passed:
        return None, report
    normalized = normalize_intensity(image, config)
    letterboxed = resize_letterbox(normalized, config.resolution)
    return (letterboxed * 255).round().astype(np.uint8), report


def to_model_tensor(image: np.ndarray) -> np.ndarray:
    """uint8 HxW → float32 3xHxW, ImageNet-normalized.

    Grayscale is repeated across three channels because the baseline backbones
    are ImageNet-pretrained and expect RGB input.
    """
    scaled = image.astype(np.float32) / 255.0
    stacked = np.repeat(scaled[None, :, :], 3, axis=0)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return (stacked - mean) / std
