"""Extract normalized landmark features from the dataset for the classifier.

Runs BlazePose over every image, normalizes the world landmarks (see
model/landmarks), and caches one feature vector per detected pose to
exports/landmark_features.json. Training reads that cache instead of re-running
detection, so a train run is fast and repeatable.

Run from the repo root:
    python scripts/extract_landmarks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.landmarks import normalize_landmarks

DATA_DIR = ROOT / "data" / "yoga-poses"
OUT = ROOT / "exports" / "landmark_features.json"
MODEL_PATH = ROOT / "exports" / "pose_landmarker_lite.task"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MIN_VISIBILITY = 0.3


def world_landmarks(result):
    """First detected pose as (x, y, z, visibility) world landmarks, or None."""
    if not result.pose_world_landmarks:
        return None
    return [
        (lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_world_landmarks[0]
    ]


def mean_visibility(landmarks) -> float:
    return sum(lm[3] for lm in landmarks) / len(landmarks)


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"No landmarker at {MODEL_PATH}. Run scripts/build_reference.py first."
        )

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )

    folders = sorted(
        p.name
        for p in DATA_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    samples: list[dict] = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for label in folders:
            files = [
                f
                for f in (DATA_DIR / label).iterdir()
                if f.suffix.lower() in IMAGE_EXT
            ]
            kept = 0
            for f in files:
                try:
                    rgb = np.asarray(Image.open(f).convert("RGB"))
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    lms = world_landmarks(landmarker.detect(image))
                except Exception as e:
                    print(f"warning: {label}/{f.name}: {e}", file=sys.stderr)
                    continue
                # Drop a barely-visible detection, its geometry is unreliable and
                # would teach the classifier noise
                if lms is None or mean_visibility(lms) < MIN_VISIBILITY:
                    continue
                samples.append({"label": label, "feature": normalize_landmarks(lms)})
                kept += 1
            print(f"{label}: {kept}/{len(files)} usable")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"labels": folders, "samples": samples}))
    print(f"\nwrote {len(samples)} features to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
