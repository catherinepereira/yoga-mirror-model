"""Parity check: the exported ONNX must match the .pt model.

Runs the held-out set through both and reports the max logit drift. Drift over
1e-4 means an operator fused wrong during export.

Run from the repo root:
    python scripts/evaluate_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.classifier import build_model
from model.dataset import load_split

DATA_DIR = ROOT / "data" / "raw"
CKPT = ROOT / "exports" / "pose_classifier.pt"
ONNX = ROOT / "exports" / "pose_classifier.onnx"


def main() -> None:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = build_model(len(ckpt["labels"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    session = ort.InferenceSession(
        str(ONNX), providers=["CPUExecutionProvider"]
    )

    _, val_ds, _ = load_split(DATA_DIR)
    loader = DataLoader(val_ds, batch_size=16)

    max_drift = 0.0
    agree, count = 0, 0
    with torch.no_grad():
        for images, _ in loader:
            pt_logits = model(images).numpy()
            onnx_logits = session.run(None, {"input": images.numpy()})[0]
            max_drift = max(max_drift, float(np.abs(pt_logits - onnx_logits).max()))
            agree += int((pt_logits.argmax(1) == onnx_logits.argmax(1)).sum())
            count += images.shape[0]

    print(f"max logit drift: {max_drift:.2e}")
    print(f"argmax agreement: {agree}/{count}")
    if max_drift > 1e-4:
        print("WARNING: drift over 1e-4, an operator likely fused wrong")


if __name__ == "__main__":
    main()
