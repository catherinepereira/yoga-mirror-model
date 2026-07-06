"""Build the reference-pose target library from the photo dataset.

Runs BlazePose over every photo in data/raw/<pose-id>/, takes the per-pose median
joint angle as the target with a tolerance from the spread, and writes
exports/referencePoses.json. That file is an artifact the app consumes, the same
contract as the ONNX model. The app never reads this repo's data.

BlazePose runs through the mediapipe Python package with the same lite model the
app uses, so the angles match what the app computes at runtime.

Run from the repo root:
    python scripts/build_reference.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from statistics import median

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.angles import JOINT_NAMES, compute_angles

DATA_DIR = ROOT / "data" / "yoga-poses"
OUT = ROOT / "exports" / "referencePoses.json"
MODEL_PATH = ROOT / "exports" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

MIN_VISIBILITY = 0.5
MIN_TOLERANCE_DEG = 6
MAX_TOLERANCE_DEG = 25

SIDE_ON = "Stand side-on so the whole body is visible"
FACE_ON = "Face the camera, full body in frame"
FLOOR_SIDE = "Lie or kneel side-on to the camera"

CAMERA_HINTS = {
    "boat-pose": FLOOR_SIDE,
    "camel-pose": SIDE_ON,
    "cat-pose": FLOOR_SIDE,
    "child-pose": FLOOR_SIDE,
    "cobra-pose": FLOOR_SIDE,
    "cow-pose": FLOOR_SIDE,
    "downward-dog": SIDE_ON,
    "eagle-pose": FACE_ON,
    "pigeon-pose": FLOOR_SIDE,
    "tree-pose": FACE_ON,
    "triangle-pose": SIDE_ON,
    "warrior-i": SIDE_ON,
    "warrior-ii": SIDE_ON,
}

NAME_OVERRIDES = {
    "warrior-i": "Warrior I",
    "warrior-ii": "Warrior II",
    "cat-pose": "Cat Pose",
    "cow-pose": "Cow Pose",
    "child-pose": "Child's Pose",
    "downward-dog": "Downward Dog",
}

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def mad_tolerance(values: list[float], center: float) -> float:
    """Median absolute deviation scaled to about one sigma, clamped to a range."""
    mad = median([abs(v - center) for v in values]) * 1.4826
    return min(MAX_TOLERANCE_DEG, max(MIN_TOLERANCE_DEG, mad))


def to_name(folder: str) -> str:
    return NAME_OVERRIDES.get(folder) or folder.replace("-", " ").title()


def landmarks_of(result) -> list[tuple] | None:
    """First detected pose as (x, y, z, visibility) world landmarks, or None."""
    if not result.pose_world_landmarks:
        return None
    return [
        (lm.x, lm.y, lm.z, lm.visibility)
        for lm in result.pose_world_landmarks[0]
    ]


def build_targets(samples: list[list[tuple]]) -> dict:
    """Per-joint median target and tolerance from a pose's image samples."""
    joints = {}
    for name in JOINT_NAMES:
        angles = []
        for lms in samples:
            a = compute_angles(lms)[name]
            if a["angle"] is not None and a["visibility"] >= MIN_VISIBILITY:
                angles.append(a["angle"])
        if len(angles) < 3:
            continue
        target = median(angles)
        joints[name] = {"target": target, "tolerance": mad_tolerance(angles, target)}
    return joints


def ensure_model() -> None:
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("downloading pose_landmarker_lite.task...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"No dataset at {DATA_DIR}. Run scripts/fetch_data.py first.")

    ensure_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )

    poses = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for folder in sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir()):
            files = [
                f
                for f in (DATA_DIR / folder).iterdir()
                if f.suffix.lower() in IMAGE_EXT
            ]
            samples = []
            for f in files:
                try:
                    # Force RGB so a grayscale or palette image never yields a
                    # single-channel tensor, which crashes the inference graph
                    rgb = np.asarray(Image.open(f).convert("RGB"))
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    lms = landmarks_of(landmarker.detect(image))
                    if lms:
                        samples.append(lms)
                except Exception as e:
                    # a single bad image should not abort the pose, but report it
                    # so a systematically broken folder is visible
                    print(f"warning: {folder}/{f.name}: {e}", file=sys.stderr)

            if len(samples) < 3:
                print(f"skip {folder}: only {len(samples)} usable images")
                continue

            joints = build_targets(samples)
            if not joints:
                print(f"skip {folder}: no joints could be measured")
                continue

            poses.append(
                {
                    "id": folder,
                    "name": to_name(folder),
                    "cameraHint": CAMERA_HINTS.get(folder, FACE_ON),
                    "joints": joints,
                }
            )
            print(f"{folder}: {len(samples)}/{len(files)} images, {len(joints)} joints")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(poses, indent=2) + "\n")
    print(f"\nwrote {len(poses)} poses to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
