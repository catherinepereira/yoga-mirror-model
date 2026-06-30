# Yoga Mirror model

Training repo for the Yoga Mirror pose classifier. Fine-tunes an image CNN on
yoga photos and exports an ONNX artifact the app runs in the browser to name the
pose you are in. The app lives in the sibling [yoga-mirror](../yoga-mirror) repo
and depends only on the exported files, not on this code.

The app's live alignment scoring is separate and uses BlazePose joint angles.
This model only answers "which pose is this" from the image.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

A CUDA GPU is used automatically if present. Training is a short fine-tune.

## Data

Photos live in `data/raw/<pose-id>/`, one folder per pose (gitignored). Fetch or
top up with the keyless image fetch (needs the local prompt2dataset checkout):

```bash
python scripts/fetch_data.py
```

It reuses prompt2dataset's image sources (Bing, DuckDuckGo, Wikimedia, Openverse)
with no API key. Re-running tops up toward 50 images per pose and is idempotent.

## Train, export, evaluate

```bash
python scripts/train.py --smoke     # 2-epoch pipeline check on a small slice
python scripts/train.py             # real run, writes exports/pose_classifier.pt
python scripts/export_onnx.py       # -> pose_classifier.onnx + model_meta.json
python scripts/evaluate.py          # held-out accuracy and confusion pairs
python scripts/evaluate_onnx.py     # parity: ONNX matches the .pt model
```

## Reference pose targets

The app also scores live alignment against per-pose target angles. Those are
built here from the same photos, so the dataset stays in one place:

```bash
python scripts/build_reference.py   # -> exports/referencePoses.json
```

BlazePose runs through the mediapipe Python package (same lite model the app
uses) over each photo, and the per-pose median joint angle becomes the target.
The angle math is a Python port of the app's, kept in step by a parity test
(`tests/test_angle_parity.py`) against a fixture generated from the app's
TypeScript.

## Artifacts the app consumes

This repo exports three files. The app commits them and contains no training code
or dataset access:

```bash
cp exports/pose_classifier.onnx ../yoga-mirror/public/models/
cp exports/model_meta.json      ../yoga-mirror/public/models/
cp exports/referencePoses.json  ../yoga-mirror/src/data/
```

`model_meta.json` carries the labels, input shape, and normalization constants so
the browser preprocessing matches training. `referencePoses.json` is the target
angle library for scoring.

## Architecture

| Field | Value |
| --- | --- |
| Backbone | MobileNetV3-small, ImageNet-pretrained |
| Head | final linear resized to the pose count (32) |
| Input | 224x224 RGB, ImageNet normalization |
| Augmentation | random resized crop, horizontal flip, rotation, color jitter |
| Optimizer | AdamW, lr 3e-4 |
| Output | per-class logits, softmax in the app |

## Things that will bite you

- **Domain gap.** Training photos are posed studio shots. The app runs the model
  on a webcam frame in a home with a cluttered background and partial framing.
  Held-out accuracy on studio photos overstates real webcam accuracy. The
  horizontal flip and crop augmentation help a little. If it is too weak in
  practice, the next step is cropping to the person with the BlazePose bounding
  box before classifying.
- **Preprocessing parity.** The browser reproduces the torchvision resize and
  normalization by hand. A mismatch silently tanks accuracy. The app has a
  fixture test that feeds a known image through both paths and compares logits.
  Regenerate the fixture if the preprocessing changes here.
- **ONNX export drift.** `evaluate_onnx.py` compares the exported model to the
  checkpoint. Drift over 1e-4 means an operator fused wrong, do not ship it.

## Layout

```
model/
  classifier.py    MobileNetV3-small with a resized head
  dataset.py       ImageFolder loader, split, transforms
scripts/
  fetch_data.py    keyless image fetch into data/raw/
  train.py         fine-tune, --smoke for a quick check
  export_onnx.py   checkpoint -> onnx + model_meta.json
  evaluate.py      held-out accuracy and confusions
  evaluate_onnx.py onnx-vs-checkpoint parity
data/raw/          photos per pose (gitignored)
exports/           .pt + .onnx + meta (gitignored)
```
