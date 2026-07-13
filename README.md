# Yoga Mirror model

Training repo for the Yoga Mirror pose classifier. Runs BlazePose over yoga
photos, trains a small MLP on the normalized landmarks, and exports an ONNX
artifact the app runs in the browser to name the pose you are in. The app lives
in the sibling [yoga-mirror](../yoga-mirror) repo and depends only on the exported
files, not on this code.

The classifier reads body geometry, not pixels, so a pose the model learned from
studio photos still reads correctly on a home webcam with a different background
and lighting.

The app scores a pose by whether the classifier's top class is the target pose,
so it inherits the classifier's rotation invariance. There is no separate
angle-based scorer.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Needs Python 3.11 (dinscribe, used by the STT scripts). A CUDA GPU is used
automatically if present. Training is a short fine-tune.

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

For the classifier's held-out accuracy, confusion matrix, and per-class scores,
run `notebooks/evaluate_classifier.ipynb`. It reproduces the training split with
the same seed, so its numbers match the validation set the checkpoint was picked
on.

## Reference pose targets (not used by the app)

`build_reference.py` writes per-pose median joint angles to
`exports/referencePoses.json`. The app once scored live alignment against these,
but now scores on the classifier's top class, so the app no longer reads this
file. The script and its angle-parity test are kept for analysis of the dataset.

```bash
python scripts/build_reference.py   # -> exports/referencePoses.json
```

## Follow-along tracks

The app's follow-along mode plays an instructor video and matches you against it.
Each video is processed here into a track the app plays against, since the
browser cannot read a cross-origin YouTube embed's pixels:

```bash
python scripts/process_video.py "https://youtu.be/VIDEO_ID"
```

This downloads the video, samples frames, runs BlazePose, and writes
`../yoga-mirror/public/tracks/<id>.json` plus a manifest entry. Each frame gets
joint angles (for scoring), 2D landmarks (for the skeleton overlay), and a
classified pose label. The track title comes from the video. It also caches the
world landmarks to `exports/skeletons/<id>.json` and prints the detected poses
and low-confidence guesses. Needs `ffmpeg` on PATH.

For a batch, list the YouTube URLs in `videos.txt` (one per line) and process
them in one run. Each video is transcribed as well, so the report shows the
detected poses, low-confidence visual guesses, and spoken poses the dataset does
not cover. The skeleton and transcript are cached under `exports/`:

```bash
python scripts/process_batch.py videos.txt
```

Re-run the classifier over a cached skeleton without re-downloading, after
retraining the model or changing the angle math:

```bash
python scripts/reclassify.py VIDEO_ID   # or --all for every cached skeleton
```

Optionally cross-check the narration against the visual poses:

```bash
python scripts/stt_crosscheck.py VIDEO_ID
```

This transcribes the audio with dinscribe, scans it for pose names against
`scripts/pose_vocab.txt`, and folds `spokenPoses` and `crossCheck` into the
track. A cue agrees when the spoken pose shows up on the visual track within a
forward-leaning window, since the instructor names a pose before settling into
it. The app shows a spoken cue when the narration names a pose the classifier is
not showing.

## Finding poses to add

The classifier only knows the poses with a folder in `data/yoga-poses/`. To see
what to collect next, transcribe a batch of videos and report the poses their
narration names but the dataset does not:

```bash
python scripts/mine_poses.py urls.txt   # one YouTube URL or id per line
```

Two passes per transcript. The phrase pass pulls every "X pose" / "X posture"
cue and drops the ones that name a dataset pose, ranked by how often they come up
across the batch. The rare-word pass counts every spoken word and lists the
least-frequent remainder, which catches poses cued without the word "pose"
(savasana, plank) that the phrase pass misses. Writes
`exports/candidate_poses.json` and prints a summary. Needs the STT deps.

## Artifacts the app consumes

This repo exports two files. The app commits them and contains no training code
or dataset access:

```bash
cp exports/pose_classifier.onnx ../yoga-mirror/public/models/
cp exports/model_meta.json      ../yoga-mirror/public/models/
```

`model_meta.json` carries the labels and the feature spec so the browser builds
the same input the model trained on. The label list is also what the app's pose
library keys on, so keep `src/data/poses.ts` in step with it.

## Architecture

| Field        | Value                                                               |
| ------------ | ------------------------------------------------------------------- |
| Feature      | BlazePose 33 world landmarks, hip-centered and torso-scaled, 99-dim |
| Model        | 2-hidden-layer MLP (128 units), dropout 0.3                         |
| Augmentation | left-right mirror (negate x, swap left/right landmarks)             |
| Optimizer    | AdamW, lr 1e-3, weight decay 1e-4                                   |
| Output       | per-class logits, softmax in the app                                |

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
  blazepose.py            shared pose detector (model path, options, decode)
  extract_landmarks.py    BlazePose over the dataset -> feature cache
  train_landmarks.py      train the MLP on the cache
  export_landmarks_onnx.py checkpoint -> onnx + model_meta.json
  build_reference.py      per-pose target angles -> referencePoses.json
  viz_poses.py            skeleton overlays for spot-checking
  process_video.py        a video -> follow-along track + cached skeleton
  process_batch.py        a videos.txt list -> a track each, with a pose report
  reclassify.py           re-label a track from its cached skeleton, no download
  stt_crosscheck.py       transcribe narration, cross-check the visual poses
  mine_poses.py           transcribe a batch, report poses not in the dataset
  stt_common.py           shared audio download + dinscribe transcription
  track_common.py         shared paths, id resolution, track helpers
data/yoga-poses/     photos per pose (gitignored)
exports/             .pt + .onnx + meta + feature cache + skeletons (gitignored)
```
