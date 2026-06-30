"""Fine-tune the pose image classifier.

Trains MobileNetV3-small on data/raw/<pose-id>/ and writes a checkpoint to
exports/pose_classifier.pt with the class labels. Use --smoke for a 2-epoch
end-to-end check on a small slice before a real run.

Run from the repo root:
    python scripts/train.py
    python scripts/train.py --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.classifier import build_model
from model.dataset import load_split

DATA_DIR = ROOT / "data" / "raw"
OUT = ROOT / "exports" / "pose_classifier.pt"


def run_epoch(model, loader, device, optimizer=None) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, count = 0.0, 0, 0

    for images, targets in tqdm(loader, leave=False):
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = loss_fn(logits, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        count += images.size(0)

    return total_loss / count, correct / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    # Seed the data shuffle and weight init so runs are comparable. cudnn is left
    # nondeterministic, full determinism would slow training and is not needed here
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_ds, val_ds, labels = load_split(DATA_DIR)
    if args.smoke:
        train_ds = torch.utils.data.Subset(train_ds, range(min(64, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(32, len(val_ds))))
        args.epochs = 2
    print(f"{len(train_ds)} train, {len(val_ds)} val, {len(labels)} classes")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = build_model(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device)
        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"train {tr_acc:.3f}  val {val_acc:.3f}"
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "labels": labels}, OUT)

    print(f"best val accuracy: {best_acc:.3f}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    (OUT.parent / "labels.json").write_text(json.dumps(labels))


if __name__ == "__main__":
    main()
