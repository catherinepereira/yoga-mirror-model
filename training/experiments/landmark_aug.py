"""Experiment 4: landmark MLP trained on image-augmented landmark features.

Same model and training as experiment 1, but the feature cache comes from
extract_landmarks_aug: each photo plus augmented variants (stretch/squash, crop,
rotation, occlusion) run through BlazePose. Tests whether landmark geometry from
warped and partly-occluded images adds anything on top of the mirror augmentation
the landmark model already uses.

The split is on the source image (`group`), not the sample, so no augmented copy
of a training image can leak into validation. Validation scores clean originals
only, so the number compares directly against the other experiments.

Run from the repo root:
    python -m training.extract_landmarks_aug          # once, builds the aug cache
    python -m training.experiments.landmark_aug
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from common.landmarks import mirror_feature
from training.config import EXPORTS, TrainingConfig
from training.models import build_landmark_mlp

FEATURES = EXPORTS / "landmark_features_aug.json"


@dataclass
class LandmarkAugConfig(TrainingConfig):
    """Full-batch AdamW on the image-augmented landmark cache.

    Same hyperparameters as the plain landmark experiment, so the only difference
    under test is the augmented feature set, not the training recipe
    """

    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 0
    checkpoint: Path = EXPORTS / "landmark_aug.pt"


def load_samples(path: Path):
    data = json.loads(path.read_text())
    labels = data["labels"]
    idx = {name: i for i, name in enumerate(labels)}
    samples = [
        {"y": idx[s["label"]], "feature": s["feature"], "group": s["group"], "aug": s["aug"]}
        for s in data["samples"]
    ]
    return samples, labels


def _group_split(samples, val_split, seed):
    """Split source images into train/val, keeping every image's copies together.

    Returns (train_samples, val_samples). Each group (one source image) is a train
    or a val image as a whole. Train keeps clean and augmented copies, val keeps
    only clean ones so the score matches the other experiments
    """
    groups = sorted({s["group"] for s in samples})
    # stratify the group split by each group's label (constant within a group)
    group_label = {}
    for s in samples:
        group_label.setdefault(s["group"], s["y"])
    train_groups, val_groups = train_test_split(
        groups, test_size=val_split, random_state=seed,
        stratify=[group_label[g] for g in groups],
    )
    train_groups, val_groups = set(train_groups), set(val_groups)

    train = [s for s in samples if s["group"] in train_groups]
    val = [s for s in samples if s["group"] in val_groups and not s["aug"]]
    return train, val


def train(cfg: LandmarkAugConfig) -> float:
    """Train the MLP on the augmented cache, save best-val, return its accuracy."""
    if not FEATURES.exists():
        raise SystemExit(
            f"No features at {FEATURES}. Run the extract_landmarks_aug step first."
        )
    torch.manual_seed(cfg.seed)

    samples, labels = load_samples(FEATURES)
    train_s, val_s = _group_split(samples, cfg.val_split, cfg.seed)

    X_tr = [s["feature"] for s in train_s]
    y_tr = [s["y"] for s in train_s]
    # Mirror-augment the training split, same as the plain landmark experiment
    X_aug = X_tr + [mirror_feature(f) for f in X_tr]
    y_aug = y_tr + y_tr
    X_val = [s["feature"] for s in val_s]
    y_val = [s["y"] for s in val_s]

    device = cfg.device
    Xtr = torch.tensor(X_aug, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_aug, dtype=torch.long, device=device)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.long, device=device)
    print(f"device: {device}")
    n_clean_tr = sum(1 for s in train_s if not s["aug"])
    print(f"{n_clean_tr} clean + {len(train_s) - n_clean_tr} aug train "
          f"(+mirror = {len(X_aug)}), {len(val_s)} clean val, {len(labels)} classes")

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
    train(LandmarkAugConfig())


if __name__ == "__main__":
    main()
