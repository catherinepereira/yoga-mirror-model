"""The two model families the experiments train.

build_landmark_mlp is the small classifier over normalized landmark features.
build_image_model wraps a torchvision MobileNetV3-Large with a fresh head, either
frozen (transfer-learning, train the head only) or trainable (fine-tune).
"""

from __future__ import annotations

import torch.nn as nn

from common.landmarks import FEATURE_DIM


def build_landmark_mlp(num_classes: int, hidden: int = 128) -> nn.Module:
    """Two hidden layers with dropout over the 99-dim landmark feature.

    Small enough to export to ONNX and run per-frame in the browser with no
    visible cost, and to separate the poses without overfitting a few hundred
    samples
    """
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, num_classes),
    )


def build_image_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    """MobileNetV3-Large, ImageNet-pretrained, with a new num_classes head.

    freeze_backbone leaves the pretrained features fixed and trains only the new
    classifier, the transfer-learning setup. False trains everything end-to-end
    for the fine-tune. The head is always fresh, so it trains either way

    torchvision is imported here, not at module load, so the landmark experiment and
    the notebook do not pull in the image stack they never use
    """
    from torchvision import models

    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    )
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
