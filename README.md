# Yoga Mirror model

Training repo for the Yoga Mirror pose classifier. Runs BlazePose over yoga
photos, trains a small MLP on the normalized landmarks, and exports an ONNX
artifact the app runs in the browser to name the pose you are in. The app lives
in the sibling [yoga-mirror](../yoga-mirror) repo and depends only on the exported
files, not on this code.

The classifier reads body geometry, not pixels, so a pose the model learned from
studio photos still reads correctly on a home webcam with a different background
and lighting.

The app's live alignment scoring uses BlazePose joint angles built here too, so
the photo dataset stays in one place.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

A CUDA GPU is used automatically if present. Training is a short fine-tune.

## Data

Photos live in `data/yoga-poses/<pose-id>/`, one folder per pose (gitignored),
curated with the sorty tool. Its `.sorty/` cache is ignored by the loaders.

## Train, export, evaluate

The classifier trains on landmark features, so the pipeline is: detect poses
once, cache the features, then train the MLP off the cache.

```bash
python scripts/build_reference.py       # downloads the BlazePose .task (first run)
python scripts/extract_landmarks.py     # -> exports/landmark_features.json
python scripts/train_landmarks.py       # -> exports/pose_classifier.pt
python scripts/export_landmarks_onnx.py # -> pose_classifier.onnx + model_meta.json
```

Spot-check that BlazePose landed on the bodies before trusting the features:

```bash
python scripts/viz_poses.py             # skeleton overlays -> exports/pose-viz/
```

Open the per-pose contact sheets in `exports/pose-viz/`. Any image with no
detection gets a red NO POSE banner.

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

`model_meta.json` carries the labels and the feature spec so the browser builds
the same input the model trained on. `referencePoses.json` is the target angle
library for scoring.

## Architecture

| Field | Value |
| --- | --- |
| Feature | BlazePose 33 world landmarks, hip-centered and torso-scaled, 99-dim |
| Model | 2-hidden-layer MLP (128 units), dropout 0.3 |
| Augmentation | left-right mirror (negate x, swap left/right landmarks) |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| Output | per-class logits, softmax in the app |

The feature is invariant to where the person stands and how far from the camera,
so a pose reads the same near or far, left- or right-facing.

## Things that will bite you

- **Feature parity.** The browser normalizes live landmarks with
  `src/lib/poseLandmarks.ts`, a port of `model/landmarks.py`. A mismatch feeds the
  model a distribution it never trained on and tanks accuracy. A fixture test in
  the app guards it, regenerate the fixture if the normalization changes here.
- **Landmark quality.** The classifier is only as good as BlazePose on the
  photos. Run `viz_poses.py` and skim the sheets, a class where the skeleton
  misses the body will train the model on noise.
- **ONNX export.** The MLP exports cleanly (no fusion ambiguity), but confirm the
  browser end-to-end test still predicts the dataset poses after any change.

## Layout

```
model/
  landmarks.py       landmark normalization and mirror, ported to the app
  landmark_mlp.py    the classifier MLP
  angles.py          joint-angle math for the reference targets
scripts/
  extract_landmarks.py    BlazePose over the dataset -> feature cache
  train_landmarks.py      train the MLP on the cache
  export_landmarks_onnx.py checkpoint -> onnx + model_meta.json
  build_reference.py      per-pose target angles -> referencePoses.json
  viz_poses.py            skeleton overlays for spot-checking
data/yoga-poses/     photos per pose (gitignored)
exports/             .pt + .onnx + meta + feature cache (gitignored)
```
