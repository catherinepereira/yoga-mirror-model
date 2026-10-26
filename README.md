# Yoga Mirror Model

Training repo for the Yoga Mirror pose classifier. Runs BlazePose over yoga
photos, trains classifiers on the poses, and exports an ONNX artifact the app runs
in the browser to name the pose you are in. The [Yoga Mirror App](https://reflect.cv/yoga-mirror/) depends only on the exported files from here.

I trained four models to compare:

- `landmark` (0.938): a small MLP over normalized BlazePose landmarks
- `landmark_aug` (0.908): the same MLP, trained on landmarks extracted from augmented
  photos (stretch/squash, crop, rotation, occlusion applied before BlazePose runs).
  The warped and partly-occluded frames give the detector inputs that produce
  landmark geometry the mirror augmentation alone cannot.
- `image_transfer` (0.769): MobileNetV3-Large pretrained on ImageNet, backbone frozen,
  only a new head trained on the photos.
- `image_finetune` (0.862): the same backbone unfrozen and fine-tuned end-to-end.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
```

Needs Python 3.11 (dinscribe, used by the STT scripts used to process YouTube yoga videos for the follow-along section).


## Data

Photos live in `data/yoga-poses/<pose-id>/`, one folder per pose (gitignored),
curated with the [Sorty](https://github.com/catherinepereira/sorty) tool.

## Train, export, benchmark

The landmark experiment trains off a feature cache, so detect poses once, cache the
features, then train the MLP off the cache. The image experiments read the photos
directly.

```bash
# once, downloads the BlazePose .task on first run and builds the feature cache
python -m training.extract_landmarks
# once, the augmented cache for experiment 4 (re-runs BlazePose over aug variants)
python -m training.extract_landmarks_aug

# experiment 1: landmark MLP -> exports/landmark_mlp.pt
python -m training.experiments.landmark_mlp
# experiment 2: landmark MLP on augmented features -> exports/landmark_aug.pt
python -m training.experiments.landmark_aug
# experiment 3: frozen-backbone transfer -> exports/image_transfer.pt
python -m training.experiments.image_transfer
# experiment 4: end-to-end fine-tune -> exports/image_finetune.pt
python -m training.experiments.image_finetune
```

Each experiment builds a `TrainingConfig` (in `training/config.py`) rather than
scattering hyperparameters through its loop, and they split the dataset with the
same seed so the accuracy numbers compare on matched folds. `landmark_aug` splits
on the source image so no augmented copy of a training photo leaks into validation,
and it validates on clean originals only. The image transforms in
`training/image_data.py` augment the training split and only preprocess (resize,
center-crop, normalize) for validation, export, and prediction, so a served model
sees the same pixel distribution it was validated on.

Export a checkpoint to ONNX plus a `model_meta.json` that records how to build the
input (a landmark feature vector, or a preprocessed frame with the normalize
constants):

```bash
python -m training.export landmark          # -> pose_classifier.onnx + model_meta.json
python -m training.export landmark_aug
python -m training.export image_transfer
python -m training.export image_finetune
```

Compare the exported models on held-out accuracy and per-inference latency:

```bash
python -m training.benchmark
```

It prints one table (classes, val accuracy, mean/p50/p95 ms) over whatever has been
exported, skipping models not yet trained. `tests/test_benchmark.py` asserts a
floor on accuracy and a ceiling on latency per model, and skips a model with no
export, so the suite is runnable before every experiment has been trained.

Spot-check that BlazePose landed on the bodies before using the landmark
features:

```bash
python -m training.viz_poses    # skeleton overlays -> exports/pose-viz/
```

For the landmark classifier's confusion matrix and per-class scores, run
`notebooks/evaluate_classifier.ipynb`. It reproduces the training split with the
same seed, so its numbers match the validation set used by the checkpoint.

## Follow-along tracks

The app's follow-along mode plays an instructor video and matches you against it.
Each video is processed here into a track the app plays against, since the browser
cannot read a cross-origin YouTube embed's pixels:

```bash
python -m video.process_video "https://youtu.be/VIDEO_ID"
```

This downloads the video, samples frames, runs BlazePose, and writes
`../yoga-mirror/public/tracks/<id>.json` plus a manifest entry. Each frame gets
2D landmarks (for the skeleton overlay) and a classified pose label, which is what
the app matches against. The track title comes from the video. It also caches the
world landmarks to `exports/skeletons/<id>.json` and prints the detected poses and
low-confidence guesses. Needs `ffmpeg` on PATH.

For a batch, list the YouTube URLs in `videos.txt` (one per line) and process them
in one run. Each video is transcribed as well, so the report shows the detected
poses, low-confidence visual guesses, and spoken poses that the dataset does not cover.
The skeleton and transcript are cached under `exports/`:

```bash
python -m video.process_batch videos.txt
```

Re-run the classifier over a cached skeleton without re-downloading, after
retraining the model:

```bash
python -m video.reclassify VIDEO_ID   # or --all for every cached skeleton
```

Optionally cross-check the narration against the visual poses:

```bash
python -m video.stt_crosscheck VIDEO_ID
```

This transcribes the audio with dinscribe, scans it for pose names against
`video/pose_vocab.txt`, and folds `spokenPoses` and `crossCheck`
into the track. A cue agrees when the spoken pose shows up on the visual track
within a forward-leaning window, since the instructor names a pose before settling
into it. The app shows a spoken cue when the narration names a pose the classifier
is not showing.

## Finding poses to add

The classifier only knows the poses with a folder in `data/yoga-poses/`. To see
what to collect next, transcribe a batch of videos and report the poses their
narration names but the dataset does not:

```bash
python -m video.mine_poses urls.txt   # one YouTube URL or id per line
```

Two passes per transcript. The phrase pass pulls every "X pose" / "X posture" cue
and drops the ones that name a dataset pose, ranked by how often they come up across
the batch. The rare-word pass counts every spoken word and lists the least-frequent
remainder, which catches poses cued without the word "pose" (savasana, plank) that
the phrase pass misses. Writes `exports/candidate_poses.json` and prints a summary.
Needs the STT deps.

## Artifacts the app consumes

The landmark export is the pair the app ships. The app commits them and contains no
training code or dataset access:

```bash
cp exports/pose_classifier.onnx ../yoga-mirror/public/models/
cp exports/model_meta.json      ../yoga-mirror/public/models/
```

`model_meta.json` carries the labels and the feature spec so the browser builds the
same input the model trained on. The label list is also what the app's pose library
keys on, so keep `src/data/poses.ts` in step with it.

## Architecture

| Field        | Value                                                               |
| ------------ | ------------------------------------------------------------------- |
| Feature      | BlazePose 33 world landmarks, hip-centered and torso-scaled, 99-dim |
| Model        | 2-hidden-layer MLP (128 units), dropout 0.3                         |
| Augmentation | left-right mirror (negate x, swap left/right landmarks)             |
| Optimizer    | AdamW, lr 1e-3, weight decay 1e-4                                   |
| Output       | per-class logits, softmax in the app                                |

That table describes the shipped landmark model. The image experiments swap in a
MobileNetV3-Large backbone and a horizontal-flip augmentation. The landmark feature
is invariant to where the person stands and how far from the camera, so a pose reads
the same near or far, left- or right-facing.

## Other notes

- **Feature parity.** The browser normalizes live landmarks with
  `src/lib/poseLandmarks.ts`, a port of `common/landmarks.py`. A mismatch feeds the
  model a distribution it never trained on and tanks accuracy. A fixture test in the
  app guards it, regenerate the fixture if the normalization changes here.
- **Landmark quality.** The landmark classifier is only as good as BlazePose on the
  photos. Run `viz_poses` and skim the sheets, a class where the skeleton misses the
  body will train the model on noise.
- **Prediction transforms.** The image models must preprocess a served frame with
  `eval_transform` (the same resize/crop/normalize used for validation), not the
  training augmentation. `training/export.py` records the normalize constants and
  size in the meta so a consumer can match them.
- **ONNX export.** The MLP exports cleanly. Confirm the browser end-to-end test still
  predicts the dataset poses after any change to the shipped model.
