"""Small MLP over normalized landmark features.

Two hidden layers with dropout, enough to separate the poses from a 99-dim
landmark vector without overfitting the few hundred samples. Tiny, so it exports
to ONNX and runs per-frame in the browser with no visible cost.
"""

from __future__ import annotations

import torch.nn as nn

from model.landmarks import FEATURE_DIM


def build_mlp(num_classes: int, hidden: int = 128) -> nn.Module:
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, num_classes),
    )
