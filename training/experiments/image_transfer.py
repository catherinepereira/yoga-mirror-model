"""Experiment 2: transfer learning on images, backbone frozen.

MobileNetV3-Large pretrained on ImageNet, features frozen, only a fresh head
trained on the pose photos. Fast to train and a strong baseline, since the frozen
features already separate bodies and poses well. Writes exports/image_transfer.pt.

Run from the repo root:
    python -m training.experiments.image_transfer
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.config import EXPORTS, TrainingConfig, stratified_split
from training.image_data import (
    PoseImageDataset,
    _TransformSubset,
    eval_transform,
    train_transform,
)
from training.models import build_image_model


@dataclass
class ImageTransferConfig(TrainingConfig):
    """Head-only training, so a higher LR and fewer epochs than the fine-tune."""

    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    image_size: int = 224
    freeze_backbone: bool = True
    checkpoint: Path = EXPORTS / "image_transfer.pt"


def _loaders(cfg: ImageTransferConfig):
    """Train and val loaders over the same file list, split by the shared seed.

    The train subset augments, the val subset only preprocesses, so validation
    scores the model on the pixel distribution a served model would see
    """
    base = PoseImageDataset(transform=eval_transform(cfg.image_size))
    train_idx, val_idx = stratified_split(base.labels, cfg.val_split, cfg.seed)
    train_set = _TransformSubset(base, train_idx, train_transform(cfg.image_size))
    val_set = _TransformSubset(base, val_idx, eval_transform(cfg.image_size))
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size)
    return train_loader, val_loader, base.classes


@torch.no_grad()
def _accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total if total else 0.0


def train(cfg: ImageTransferConfig) -> float:
    """Train the frozen-backbone head, save the best-val checkpoint, return its acc."""
    torch.manual_seed(cfg.seed)
    device = cfg.device
    train_loader, val_loader, classes = _loaders(cfg)
    print(f"device: {device}")
    print(f"{len(train_loader.dataset)} train, {len(val_loader.dataset)} val, "
          f"{len(classes)} classes")

    model = build_image_model(len(classes), freeze_backbone=cfg.freeze_backbone).to(device)
    # Only the parameters left trainable (the head) go to the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        val_acc = _accuracy(model, val_loader, device)
        if val_acc >= best_acc:
            best_acc = val_acc
            cfg.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "labels": classes,
                 "image_size": cfg.image_size},
                cfg.checkpoint,
            )
        print(f"epoch {epoch + 1}/{cfg.epochs}  val {val_acc:.3f}")

    print(f"best val accuracy: {best_acc:.3f}")
    print(f"wrote {cfg.checkpoint}")
    return best_acc


def main() -> None:
    train(ImageTransferConfig())


if __name__ == "__main__":
    main()
