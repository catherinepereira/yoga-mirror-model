"""Extract normalized landmark features from the dataset for the classifier.

Runs BlazePose over every image, normalizes the world landmarks (see
common.landmarks), and caches one feature vector per detected pose to
exports/landmark_features.json. The landmark experiment reads that cache instead
of re-running detection, so a train run is fast and repeatable.

Run from the repo root:
    python -m training.extract_landmarks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from common.landmarks import normalize_landmarks
from common.blazepose import PoseDetector
from common.dataset import pose_folders

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports" / "landmark_features.json"

MIN_VISIBILITY = 0.3


def mean_visibility(landmarks) -> float:
    return sum(lm[3] for lm in landmarks) / len(landmarks)


def main() -> None:
    folders = pose_folders()
    samples: list[dict] = []
    with PoseDetector() as detect:
        for label, files in folders:
            kept = 0
            for f in files:
                try:
                    lms = detect.landmarks(f)
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

    labels = [label for label, _ in folders]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"labels": labels, "samples": samples}))
    print(f"\nwrote {len(samples)} features to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
