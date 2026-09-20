"""Augmentation tests — the training/inference separation (System Design §22)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from ml.xray.augment import AugmentConfig, augment


def _image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((64, 64)) * 255).astype(np.uint8)


def test_augmentation_is_reproducible_from_a_seed() -> None:
    config = AugmentConfig()
    first = augment(_image(), config, np.random.default_rng(42))
    second = augment(_image(), config, np.random.default_rng(42))
    assert np.array_equal(first, second)


def test_augmentation_actually_varies_across_draws() -> None:
    config = AugmentConfig()
    rng = np.random.default_rng(1)
    outputs = [augment(_image(), config, rng) for _ in range(5)]
    assert any(not np.array_equal(outputs[0], other) for other in outputs[1:])


def test_augmentation_preserves_shape_and_dtype() -> None:
    out = augment(_image(), AugmentConfig(), np.random.default_rng(2))
    assert out.shape == (64, 64)
    assert out.dtype == np.uint8


def test_a_disabled_config_is_close_to_the_identity() -> None:
    off = AugmentConfig(
        horizontal_flip=0.0,
        max_rotation_degrees=0.0,
        max_translate_fraction=0.0,
        max_scale_jitter=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )
    image = _image()
    assert np.abs(augment(image, off, np.random.default_rng(3)).astype(int) - image).max() <= 1


def test_the_inference_path_does_not_import_augmentation() -> None:
    # The structural guarantee behind System Design §5 and §22: if augmentation
    # ever reaches inference, the same study yields different results on retry.
    # Checked against the import graph, not the text — the module's docstring
    # legitimately discusses augmentation.
    tree = ast.parse(Path("ml/preprocessing/xray.py").read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported += [f"{node.module}.{alias.name}" for alias in node.names]
    assert not [name for name in imported if "augment" in name], imported
