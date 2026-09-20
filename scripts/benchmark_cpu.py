"""Measure this machine's real training/inference throughput before planning runs.

Phase 1 needs an honest answer to "can I train here, or do I need a rented
GPU?" — so measure the box instead of guessing from the CPU model number.
Reports images/second for forward+backward (training) and forward-only
(inference) at the resolutions the X-ray baseline is likely to use.

    python -m scripts.benchmark_cpu --epoch-images 13721
"""

from __future__ import annotations

import argparse
import platform
import time

import torch
from torch import nn
from torchvision.models import resnet18, resnet34


def _throughput(
    model: nn.Module,
    *,
    resolution: int,
    batch_size: int,
    steps: int,
    train: bool,
) -> float:
    images = torch.randn(batch_size, 3, resolution, resolution)
    targets = torch.randint(0, 2, (batch_size,))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train(train)

    def step() -> None:
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.inference_mode():
                model(images)

    step()  # warm-up: first call pays one-off allocation and kernel selection
    start = time.perf_counter()
    for _ in range(steps):
        step()
    elapsed = time.perf_counter() - start
    return steps * batch_size / elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-images", type=int, default=13721, help="train-split size")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    threads = torch.get_num_threads()
    print(f"{platform.processor() or platform.machine()} | torch {torch.__version__}")
    print(f"threads={threads} cuda={torch.cuda.is_available()}")
    print(f"epoch = {args.epoch_images} images, batch = {args.batch_size}\n")
    print(f"{'model':<10}{'res':>6}{'mode':>11}{'img/s':>9}{'epoch':>12}")
    print("-" * 48)

    for name, factory in (("resnet18", resnet18), ("resnet34", resnet34)):
        for resolution in (224, 384):
            model = factory(weights=None, num_classes=2)
            for train in (True, False):
                rate = _throughput(
                    model,
                    resolution=resolution,
                    batch_size=args.batch_size,
                    steps=args.steps,
                    train=train,
                )
                mode = "train" if train else "inference"
                epoch = args.epoch_images / rate
                epoch_text = f"{epoch / 60:.1f} min" if train else "-"
                print(f"{name:<10}{resolution:>6}{mode:>11}{rate:>9.1f}{epoch_text:>12}")


if __name__ == "__main__":
    main()
