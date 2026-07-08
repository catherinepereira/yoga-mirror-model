"""Shared BlazePose detection for the scripts that run it over images.

The reference builder, the landmark extractor, and the video processor all decode
an image, run the same lite pose model, and read the world landmarks. This holds
that in one place so the model path, the detector options, and the RGB decode
stay identical across them.

Callers keep their own downstream gating (visibility thresholds, per-joint vs
per-frame), which genuinely differs by use.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
LANDMARKER_PATH = _ROOT / "exports" / "pose_landmarker_lite.task"
_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Confidence floor for a detection to count. Shared so every script feeds the
# model the same threshold
MIN_POSE_DETECTION_CONFIDENCE = 0.5


def ensure_landmarker() -> None:
    """Download the pose landmarker .task on first use if it is not present."""
    if LANDMARKER_PATH.exists():
        return
    LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("downloading pose_landmarker_lite.task...")
    urllib.request.urlretrieve(_LANDMARKER_URL, LANDMARKER_PATH)


def world_landmarks(result) -> list[tuple] | None:
    """First detected pose as (x, y, z, visibility) world landmarks, or None."""
    if not result.pose_world_landmarks:
        return None
    return [
        (lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_world_landmarks[0]
    ]


def make_landmarker():
    """A single-pose BlazePose landmarker with the shared options.

    For a caller that manages its own image decode (viz draws on the pixels).
    Most callers want PoseDetector instead
    """
    ensure_landmarker()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
    )
    return vision.PoseLandmarker.create_from_options(options)


class PoseDetector:
    """Context manager over a single-pose BlazePose landmarker.

    Use as `with PoseDetector() as detect:` then call `detect(image_path)` for the
    raw mediapipe result, or `detect.landmarks(image_path)` for just the world
    landmarks. Decodes to RGB so a grayscale or palette image never yields a
    single-channel tensor the graph rejects
    """

    def __enter__(self) -> "PoseDetector":
        self._landmarker = make_landmarker()
        return self

    def __exit__(self, *exc) -> None:
        self._landmarker.close()

    def __call__(self, image_path: Path):
        """Detect on one image, returning the mediapipe result."""
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect(image)

    def landmarks(self, image_path: Path) -> list[tuple] | None:
        """World landmarks for one image, or None if no pose was found."""
        return world_landmarks(self(image_path))
