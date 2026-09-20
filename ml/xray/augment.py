"""Training-only augmentation (System Design §22).

Kept in its own module, away from `ml.preprocessing.xray`, so that the
separation the design document insists on is visible in the import graph: if
anything on an inference path ever imports this, that is the bug.

Augmentations are conservative on purpose. Radiographs have a canonical
orientation and fractures are small, low-contrast, geometric features, so
vertical flips and heavy distortion would teach anatomy that does not exist.
Horizontal flip is safe here because wrists appear in both lateralities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AugmentConfig:
    horizontal_flip: float = 0.5
    max_rotation_degrees: float = 10.0
    max_translate_fraction: float = 0.05
    max_scale_jitter: float = 0.10
    brightness_jitter: float = 0.10
    contrast_jitter: float = 0.10


def augment(image: np.ndarray, config: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    """Apply training augmentation to a uint8 square image.

    `rng` is passed in rather than drawn from global state so a training run
    is reproducible from its seed alone (System Design §20).
    """
    from PIL import Image

    result = Image.fromarray(image)
    if rng.random() < config.horizontal_flip:
        result = result.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    angle = float(rng.uniform(-config.max_rotation_degrees, config.max_rotation_degrees))
    scale = 1.0 + float(rng.uniform(-config.max_scale_jitter, config.max_scale_jitter))
    shift = config.max_translate_fraction * result.size[0]
    translate = (float(rng.uniform(-shift, shift)), float(rng.uniform(-shift, shift)))

    result = result.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        translate=(round(translate[0]), round(translate[1])),
        fillcolor=0,
    )
    if abs(scale - 1.0) > 1e-3:
        size = max(1, round(result.size[0] * scale))
        result = result.resize((size, size), Image.Resampling.BILINEAR)
        result = _center_crop_or_pad(result, image.shape[0])

    array = np.asarray(result, dtype=np.float32)
    brightness = 1.0 + float(rng.uniform(-config.brightness_jitter, config.brightness_jitter))
    contrast = 1.0 + float(rng.uniform(-config.contrast_jitter, config.contrast_jitter))
    array = (array - array.mean()) * contrast + array.mean() * brightness
    return np.clip(array, 0, 255).astype(np.uint8)


def _center_crop_or_pad(image, target: int):
    from PIL import Image

    size = image.size[0]
    if size == target:
        return image
    if size > target:
        offset = (size - target) // 2
        return image.crop((offset, offset, offset + target, offset + target))
    canvas = Image.new(image.mode, (target, target), color=0)
    offset = (target - size) // 2
    canvas.paste(image, (offset, offset))
    return canvas
