"""Build the reference-pose target library from the photo dataset.

Runs BlazePose over every photo in data/yoga-poses/<pose-id>/, takes the per-pose
median joint angle as the target with a tolerance from the spread, and writes
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
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.angles import JOINT_NAMES, compute_angles
from blazepose import PoseDetector
from dataset import DATA_DIR, pose_folders

OUT = ROOT / "exports" / "referencePoses.json"

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

def mad_tolerance(values: list[float], center: float) -> float:
    """Median absolute deviation scaled to about one sigma, clamped to a range."""
    mad = median([abs(v - center) for v in values]) * 1.4826
    return min(MAX_TOLERANCE_DEG, max(MIN_TOLERANCE_DEG, mad))


def to_name(folder: str) -> str:
    return NAME_OVERRIDES.get(folder) or folder.replace("-", " ").title()


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


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"No dataset at {DATA_DIR}.")

    poses = []
    with PoseDetector() as detect:
        for folder, files in pose_folders():
            samples = []
            for f in files:
                try:
                    lms = detect.landmarks(f)
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
