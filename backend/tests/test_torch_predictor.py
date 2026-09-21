"""The real-checkpoint predictor.

Built against a synthetic checkpoint rather than the DVC-tracked 45 MB
artifact, so these run in CI without a `dvc pull`. What they protect is the
handful of ways a served model can be quietly wrong: preprocessing taken from
the wrong place, a version string that cannot identify which file produced a
prediction, and augmentation leaking onto the inference path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the backend CI job installs no torch")

from PIL import Image  # noqa: E402

from backend.inference.base import ImageQualityError  # noqa: E402
from backend.inference.torch_predictor import TorchPredictor, load_predictor  # noqa: E402
from ml.xray.model import build_model  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    """A real resnet18 with random weights, saved in the training loop's shape."""
    path = tmp_path_factory.mktemp("ckpt") / "resnet18-best.pt"
    model = build_model("resnet18", pretrained=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"backbone": "resnet18"},
            "preprocess": {
                "resolution": 224,
                "clip_percentiles": (0.5, 99.5),
                "min_source_pixels": 128,
                "min_dynamic_range": 0.05,
                "version": "xray-v1",
            },
            "dataset": "GRAZPEDWRI-DX",
            "split_seed": 20260920,
        },
        path,
    )
    return path


def xray(path: Path, *, size=(512, 512), uniform=False) -> Path:
    """A plausible radiograph, or a blank one the quality gate should refuse."""
    blank = Image.new("L", size, color=128)
    image = blank if uniform else Image.linear_gradient("L").resize(size)
    image.save(path)
    return path


def test_missing_checkpoint_says_how_to_get_it(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        TorchPredictor(tmp_path / "absent.pt")


def test_preprocessing_comes_from_the_checkpoint(checkpoint: Path) -> None:
    """Not from settings: a model trained at 224 and served at 384 is silently wrong."""
    assert TorchPredictor(checkpoint).resolution == 224


def test_model_version_identifies_the_file(checkpoint: Path, tmp_path) -> None:
    predictor = TorchPredictor(checkpoint)
    assert predictor.model_version.startswith("resnet18/xray-v1@")

    # A different checkpoint must not share a version string, or a stored
    # prediction cannot say which weights produced it.
    other = tmp_path / "other.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["state_dict"]["fc.bias"] = payload["state_dict"]["fc.bias"] + 1.0
    torch.save(payload, other)
    assert TorchPredictor(other).model_version != predictor.model_version


def test_probability_is_a_probability(checkpoint: Path, tmp_path) -> None:
    probability = TorchPredictor(checkpoint).probability(xray(tmp_path / "a.png"))
    assert 0.0 <= probability <= 1.0


def test_repeat_calls_are_identical(checkpoint: Path, tmp_path) -> None:
    """The inference path carries no augmentation (System Design §5, §22).

    If `ml.xray.augment` ever reached this path, the same upload would score
    differently on re-analysis — which is the bug this asserts against.
    """
    predictor = TorchPredictor(checkpoint)
    image = xray(tmp_path / "b.png")
    assert predictor.probability(image) == predictor.probability(image)


def test_uniform_image_is_refused_not_scored(checkpoint: Path, tmp_path) -> None:
    """Fail closed: the gate returns no image, so the model cannot run on it."""
    with pytest.raises(ImageQualityError):
        TorchPredictor(checkpoint).probability(xray(tmp_path / "c.png", uniform=True))


def test_tiny_image_is_refused(checkpoint: Path, tmp_path) -> None:
    with pytest.raises(ImageQualityError):
        TorchPredictor(checkpoint).probability(xray(tmp_path / "d.png", size=(64, 64)))


def test_loader_caches_per_path(checkpoint: Path) -> None:
    """A worker pays the 45 MB load once per process, not once per job."""
    load_predictor.cache_clear()
    assert load_predictor(str(checkpoint)) is load_predictor(str(checkpoint))
    load_predictor.cache_clear()
