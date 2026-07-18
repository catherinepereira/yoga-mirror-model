"""Export a trained experiment checkpoint to ONNX plus a model_meta.json.

The two files are the whole handoff to a consumer: the ONNX graph and the meta
that says how to build its input. The landmark model takes a normalized feature
vector, the image models take a preprocessed frame, and the meta records which,
plus the labels and the input spec, so the consumer preprocesses identically.

Run from the repo root:
    python -m training.export landmark
    python -m training.export image_transfer
    python -m training.export image_finetune
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.landmarks import FEATURE_DIM
from training.config import EXPORTS
from training.image_data import IMAGENET_MEAN, IMAGENET_STD
from training.models import build_image_model, build_landmark_mlp

ROOT = Path(__file__).resolve().parents[1]

# experiment name -> (checkpoint, onnx out, meta out, feature kind)
EXPERIMENTS = {
    "landmark": ("landmark_mlp.pt", "pose_classifier.onnx", "model_meta.json", "landmarks"),
    "landmark_aug": ("landmark_aug.pt", "landmark_aug.onnx", "landmark_aug_meta.json", "landmarks"),
    "image_transfer": ("image_transfer.pt", "image_transfer.onnx", "image_transfer_meta.json", "image"),
    "image_finetune": ("image_finetune.pt", "image_finetune.onnx", "image_finetune_meta.json", "image"),
}


def _export_onnx(model, dummy, onnx_out: Path) -> None:
    """Export a model to ONNX with a dynamic batch axis.

    dynamo=False uses the stable TorchScript exporter. The newer dynamo path
    prints a unicode glyph that crashes on the Windows cp1252 console
    """
    model.eval()
    torch.onnx.export(
        model,
        dummy,
        str(onnx_out),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def export(name: str) -> None:
    ckpt_name, onnx_name, meta_name, feature = EXPERIMENTS[name]
    ckpt_path = EXPORTS / ckpt_name
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path.relative_to(ROOT)}, train {name} first")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    labels = ckpt["labels"]
    onnx_out = EXPORTS / onnx_name
    meta_out = EXPORTS / meta_name

    if feature == "landmarks":
        model = build_landmark_mlp(len(labels))
        model.load_state_dict(ckpt["state_dict"])
        _export_onnx(model, torch.zeros(1, FEATURE_DIM), onnx_out)
        meta = {
            "labels": labels,
            "feature": "landmarks",
            "feature_dim": FEATURE_DIM,
            "num_landmarks": FEATURE_DIM // 3,
        }
    else:
        size = ckpt["image_size"]
        model = build_image_model(len(labels), freeze_backbone=False)
        model.load_state_dict(ckpt["state_dict"])
        _export_onnx(model, torch.zeros(1, 3, size, size), onnx_out)
        meta = {
            "labels": labels,
            "feature": "image",
            "image_size": size,
            "normalize_mean": IMAGENET_MEAN,
            "normalize_std": IMAGENET_STD,
        }

    meta_out.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {onnx_out.relative_to(ROOT)} and {meta_out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained experiment to ONNX")
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS), help="which checkpoint to export")
    args = parser.parse_args()
    export(args.experiment)


if __name__ == "__main__":
    main()
