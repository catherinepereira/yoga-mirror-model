"""Experiment 1: the MLP classifier over normalized landmark features.

Reads the cached landmark features, mirror-augments the training split, and trains
the small MLP. Writes exports/landmark_mlp.pt with the class labels. The classifier
keys on body geometry, not pixels, so it transfers from studio photos to a live
webcam where an image classifier would not.

Run from the repo root:
    python -m training.extract_landmarks   # once, builds the cache
    python -m training.experiments.landmark_mlp
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from common.landmarks import mirror_feature
from training.config import EXPORTS, TrainingConfig, stratified_split
from training.models import build_landmark_mlp

FEATURES = EXPORTS / "landmark_features.json"


@dataclass
class LandmarkConfig(TrainingConfig):
    """Full-batch AdamW on the landmark feature cache.

    The dataset is a few hundred vectors, so it trains full-batch in one step per
    epoch, batch_size only bounds the loaders the image experiments use
    """

    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 0
    checkpoint: Path = EXPORTS / "landmark_mlp.pt"


def load_features(path: Path):
    data = json.loads(path.read_text())
    labels = data["labels"]
    idx = {name: i for i, name in enumerate(labels)}
    X = [s["feature"] for s in data["samples"]]
    y = [idx[s["label"]] for s in data["samples"]]
    return X, y, labels


def train(cfg: LandmarkConfig) -> float:
    """Train the MLP, save the best-val checkpoint, and return its val accuracy."""
    if not FEATURES.exists():
        raise SystemExit(
            f"No features at {FEATURES}. Run the extract_landmarks step first."
        )
    torch.manual_seed(cfg.seed)

    X, y, labels = load_features(FEATURES)
    train_idx, val_idx = stratified_split(y, cfg.val_split, cfg.seed)
    X_tr = [X[i] for i in train_idx]
    y_tr = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    # Mirror-augment the training split only, so val stays a clean measure
    X_aug = X_tr + [mirror_feature(f) for f in X_tr]
    y_aug = y_tr + y_tr

    device = cfg.device
    Xtr = torch.tensor(X_aug, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_aug, dtype=torch.long, device=device)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.long, device=device)
    print(f"device: {device}")
    print(f"{len(X_tr)} train (+mirror = {len(X_aug)}), {len(X_val)} val, "
          f"{len(labels)} classes")

    model = build_landmark_mlp(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(Xva).argmax(1) == yva).float().mean().item()
        if val_acc >= best_acc:
            best_acc = val_acc
            cfg.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "labels": labels}, cfg.checkpoint)
        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch + 1}/{cfg.epochs}  loss {loss.item():.3f}  val {val_acc:.3f}")

    print(f"best val accuracy: {best_acc:.3f}")
    print(f"wrote {cfg.checkpoint}")
    return best_acc


def main() -> None:
    train(LandmarkConfig())


if __name__ == "__main__":
    main()
