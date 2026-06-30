"""MobileNetV3-small pose classifier.

ImageNet-pretrained backbone with the final layer resized to the pose count.
Small and fast enough to run in the browser through ONNX.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    net = models.mobilenet_v3_small(weights=weights)
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, num_classes)
    return net
