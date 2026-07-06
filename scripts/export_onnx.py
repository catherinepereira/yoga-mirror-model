"""Export the trained checkpoint to ONNX plus model_meta.json.

The two files are the entire contract with the app: it loads the ONNX with
onnxruntime-web and reproduces the preprocessing from the meta.

Run from the repo root:
    python scripts/export_onnx.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.classifier import build_model
from model.dataset import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

CKPT = ROOT / "exports" / "pose_classifier.pt"
ONNX_OUT = ROOT / "exports" / "pose_classifier.onnx"
META_OUT = ROOT / "exports" / "model_meta.json"


def main() -> None:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    labels = ckpt["labels"]

    model = build_model(len(labels), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    # dynamo=False uses the stable TorchScript exporter. The newer dynamo path
    # prints a unicode glyph that crashes on the Windows cp1252 console
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_OUT),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    meta = {
        "labels": labels,
        "input_shape": [1, 3, IMG_SIZE, IMG_SIZE],
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        # letterbox: pad to square with black, then resize, no crop. The browser
        # must reproduce this exactly
        "preprocess": "letterbox",
        "pad_color": [0, 0, 0],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    META_OUT.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {ONNX_OUT.relative_to(ROOT)} and {META_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
