"""Train the landmark pose classifier.

Reads the cached landmark features, augments each with its left-right mirror, and
trains a small MLP. Writes exports/pose_classifier.pt with the class labels. The
classifier keys on body geometry, not pixels, so it transfers from studio photos
to a live webcam where the image classifier did not.

Run from the repo root:
    python scripts/extract_landmarks.py   # once, to build the feature cache
    python scripts/train_landmarks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.landmark_mlp import build_mlp
from model.landmarks import mirror_feature

FEATURES = ROOT / "exports" / "landmark_features.json"
OUT = ROOT / "exports" / "pose_classifier.pt"

EPOCHS = 120
LR = 1e-3
WEIGHT_DECAY = 1e-4


def load_features():
    data = json.loads(FEATURES.read_text())
    labels = data["labels"]
    idx = {name: i for i, name in enumerate(labels)}
    X = [s["feature"] for s in data["samples"]]
    y = [idx[s["label"]] for s in data["samples"]]
    return X, y, labels


def main() -> None:
    if not FEATURES.exists():
        raise SystemExit(
            f"No features at {FEATURES}. Run scripts/extract_landmarks.py first."
        )
    torch.manual_seed(0)

    X, y, labels = load_features()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    # Mirror-augment the training split only, so val stays a clean measure
    X_aug = list(X_tr) + [mirror_feature(f) for f in X_tr]
    y_aug = list(y_tr) + list(y_tr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.tensor(X_aug, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_aug, dtype=torch.long, device=device)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.long, device=device)
    print(f"device: {device}")
    print(f"{len(X_tr)} train (+mirror = {len(X_aug)}), {len(X_val)} val, "
          f"{len(labels)} classes")

    model = build_mlp(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(EPOCHS):
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
            OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "labels": labels}, OUT)
        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch + 1}/{EPOCHS}  loss {loss.item():.3f}  val {val_acc:.3f}")

    print(f"best val accuracy: {best_acc:.3f}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
