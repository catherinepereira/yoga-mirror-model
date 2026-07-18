"""Render BlazePose skeletons over the dataset for a visual spot-check.

Runs the same PoseLandmarker the landmark extractor uses over every image, draws
the detected skeleton on it, and tiles a contact sheet per pose into
exports/pose-viz/. Open those PNGs to confirm the landmarks land on the body
before trusting the poses that feed the classifier.

Images where no pose is detected get a red NO POSE banner, so a class full of
misses is obvious at a glance.

Run from the repo root:
    python -m training.viz_poses
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

from common.blazepose import make_landmarker
from common.dataset import pose_folders

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "exports" / "pose-viz"

CELL = 260
COLS = 6

# BlazePose skeleton connections, the limbs and torso worth eyeballing
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 31), (28, 32),
]


def draw_skeleton(img: np.ndarray, landmarks) -> None:
    """Draw bones and joints in place on a BGR image using normalized coords."""
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 220, 0), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(img, (x, y), 3, (0, 120, 255), -1, cv2.LINE_AA)


def fit_cell(img: np.ndarray) -> np.ndarray:
    """Letterbox a BGR image into a CELL x CELL tile on a dark background."""
    h, w = img.shape[:2]
    scale = (CELL - 8) / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    tile = np.full((CELL, CELL, 3), 30, np.uint8)
    y0 = (CELL - resized.shape[0]) // 2
    x0 = (CELL - resized.shape[1]) // 2
    tile[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return tile


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with make_landmarker() as landmarker:
        for label, files in pose_folders():
            tiles, detected = [], 0
            for f in files:
                try:
                    rgb = np.asarray(Image.open(f).convert("RGB"))
                except Exception as e:
                    print(f"warning: {label}/{f.name}: {e}", file=sys.stderr)
                    continue
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(image)
                if result.pose_landmarks:
                    draw_skeleton(bgr, result.pose_landmarks[0])
                    detected += 1
                else:
                    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 34), (0, 0, 180), -1)
                    cv2.putText(
                        bgr, "NO POSE", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    )
                tiles.append(fit_cell(bgr))

            if not tiles:
                continue
            rows = (len(tiles) + COLS - 1) // COLS
            sheet = np.full((rows * CELL, COLS * CELL, 3), 30, np.uint8)
            for i, tile in enumerate(tiles):
                r, c = divmod(i, COLS)
                sheet[r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL] = tile
            out = OUT_DIR / f"{label}.png"
            cv2.imwrite(str(out), sheet)
            print(f"{label}: {detected}/{len(tiles)} detected -> {out.name}")

    print(f"\nwrote sheets to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
