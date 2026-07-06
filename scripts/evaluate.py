"""Held-out accuracy and the worst confusion pairs for the trained checkpoint.

Run from the repo root:
    python scripts/evaluate.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.classifier import build_model
from model.dataset import load_split

DATA_DIR = ROOT / "data" / "yoga-poses"
CKPT = ROOT / "exports" / "pose_classifier.pt"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    labels = ckpt["labels"]

    model = build_model(len(labels), pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, val_ds, _ = load_split(DATA_DIR)
    loader = DataLoader(val_ds, batch_size=32)

    correct, count = 0, 0
    confusions: Counter[tuple[str, str]] = Counter()
    with torch.no_grad():
        for images, targets in loader:
            preds = model(images.to(device)).argmax(1).cpu()
            for t, p in zip(targets, preds):
                count += 1
                if t == p:
                    correct += 1
                else:
                    confusions[(labels[t], labels[p])] += 1

    print(f"held-out accuracy: {correct / count:.3f} ({count} images)")
    print("\ntop confusions (true -> predicted):")
    for (true, pred), n in confusions.most_common(10):
        print(f"  {true} -> {pred}: {n}")


if __name__ == "__main__":
    main()
