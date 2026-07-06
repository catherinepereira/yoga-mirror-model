"""Export the trained landmark classifier to ONNX plus model_meta.json.

The two files are the whole contract with the app: it feeds live BlazePose world
landmarks through the same normalization (see model/landmarks) and runs this MLP.

Run from the repo root:
    python scripts/export_landmarks_onnx.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.landmark_mlp import build_mlp
from model.landmarks import FEATURE_DIM

CKPT = ROOT / "exports" / "pose_classifier.pt"
ONNX_OUT = ROOT / "exports" / "pose_classifier.onnx"
META_OUT = ROOT / "exports" / "model_meta.json"


def main() -> None:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    labels = ckpt["labels"]

    model = build_mlp(len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.zeros(1, FEATURE_DIM)
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
        "feature": "landmarks",
        "feature_dim": FEATURE_DIM,
        "num_landmarks": FEATURE_DIM // 3,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    META_OUT.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {ONNX_OUT.relative_to(ROOT)} and {META_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
