"""Inference-path preprocessing tests (System Design §5, §22)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ml.preprocessing.xray import (
    PreprocessConfig,
    assess_quality,
    load_grayscale,
    normalize_intensity,
    preprocess,
    resize_letterbox,
    to_model_tensor,
)


@pytest.fixture()
def radiograph(tmp_path):
    rng = np.random.default_rng(0)
    array = (rng.random((600, 400)) * 255).astype(np.uint8)
    path = tmp_path / "scan.png"
    Image.fromarray(array).save(path)
    return path


def test_preprocessing_is_deterministic(radiograph) -> None:
    # The guarantee that separates the inference path from the training path:
    # analysing the same study twice must produce the same bytes (§5, §22).
    first, _ = preprocess(radiograph)
    second, _ = preprocess(radiograph)
    assert np.array_equal(first, second)


def test_output_is_a_square_uint8_image_at_the_configured_resolution(radiograph) -> None:
    image, report = preprocess(radiograph, PreprocessConfig(resolution=256))
    assert report.passed
    assert image.shape == (256, 256)
    assert image.dtype == np.uint8


def test_aspect_ratio_is_preserved_by_padding(tmp_path) -> None:
    array = np.linspace(0, 255, 128, dtype=np.uint8)[None, :].repeat(512, axis=0)
    path = tmp_path / "tall.png"
    Image.fromarray(array).save(path)
    image, _ = preprocess(path, PreprocessConfig(resolution=128))
    # A 4:1 source letterboxed into a square leaves empty columns either side.
    assert image[:, 0].max() == 0
    assert image[:, -1].max() == 0
    assert image[64].max() > 0


def test_a_tiny_image_is_rejected_rather_than_upscaled(tmp_path) -> None:
    path = tmp_path / "tiny.png"
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8) + 128).save(path)
    image, report = preprocess(path)
    assert image is None
    assert not report.passed
    assert "below the minimum" in report.reason


def test_a_blank_image_is_rejected(tmp_path) -> None:
    path = tmp_path / "blank.png"
    Image.fromarray(np.full((512, 512), 7, dtype=np.uint8)).save(path)
    image, report = preprocess(path)
    assert image is None
    assert report.reason == "image is nearly uniform"


def test_intensity_normalization_is_robust_to_a_saturated_pixel() -> None:
    base = np.linspace(0.2, 0.4, 100 * 100, dtype=np.float32).reshape(100, 100)
    with_marker = base.copy()
    with_marker[0, 0] = 1000.0
    config = PreprocessConfig()
    # A single outlier must not compress the useful range towards zero.
    assert normalize_intensity(with_marker, config).mean() == pytest.approx(
        normalize_intensity(base, config).mean(), abs=0.02
    )


def test_sixteen_bit_input_keeps_its_range(tmp_path) -> None:
    array = (np.linspace(0, 4095, 256 * 256).reshape(256, 256)).astype(np.uint16)
    path = tmp_path / "16bit.png"
    Image.fromarray(array, mode="I;16").save(path)
    loaded = load_grayscale(path)
    assert loaded.max() == pytest.approx(1.0, abs=1e-3)
    assert loaded.min() == pytest.approx(0.0, abs=1e-3)


def test_model_tensor_has_three_imagenet_normalized_channels() -> None:
    image = np.full((8, 8), 128, dtype=np.uint8)
    tensor = to_model_tensor(image)
    assert tensor.shape == (3, 8, 8)
    assert tensor.dtype == np.float32
    # Channels differ only by the per-channel ImageNet mean/std.
    assert not np.allclose(tensor[0], tensor[1])


def test_quality_report_carries_the_measured_dynamic_range() -> None:
    image = np.linspace(0, 1, 256 * 256, dtype=np.float32).reshape(256, 256)
    report = assess_quality(image, PreprocessConfig())
    assert report.passed
    assert 0.9 < report.dynamic_range <= 1.0


def test_letterbox_handles_a_wider_than_tall_image() -> None:
    image = np.full((50, 200), 0.5, dtype=np.float32)
    out = resize_letterbox(image, 64)
    assert out.shape == (64, 64)
    assert out[0].max() == 0  # padded top row
