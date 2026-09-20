"""Baseline X-ray classifier (System Design §6, implementation-plan §1.3).

A CNN baseline on purpose: the plan is explicit that a ViT does not get
evaluated until there is a CNN number to compare it against.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    efficientnet_b0,
    resnet18,
    resnet34,
)

BACKBONES = ("resnet18", "resnet34", "efficientnet_b0")


def build_model(backbone: str = "resnet18", *, pretrained: bool = True) -> nn.Module:
    """Two-class head on an ImageNet-pretrained backbone.

    Pretrained weights matter here: 13.7k training images is small, and the
    low-level edge filters ImageNet provides transfer to radiographs even
    though the domains do not otherwise resemble each other.
    """
    if backbone == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif backbone == "resnet34":
        model = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif backbone == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    else:
        raise ValueError(f"unknown backbone {backbone!r}; expected one of {BACKBONES}")
    return model


@torch.inference_mode()
def predict_proba(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """P(fracture) for a batch. `inference_mode` so no autograd state leaks in."""
    model.eval()
    return torch.softmax(model(batch), dim=1)[:, 1]
