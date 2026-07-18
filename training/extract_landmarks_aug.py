"""Extract landmark features with image-space augmentation before detection.

The plain extractor runs BlazePose on each photo once. This one also runs it on
augmented variants of the photo (stretch/squash, crop, small rotation, cutout
mask), adding landmark geometry the mirror-only augmentation cannot produce.
Augmenting upstream of BlazePose is the point: the detector sees a warped or
partly-occluded body and returns landmarks that differ from the clean frame's.

The hip-center + torso-scale normalization cancels uniform translate and zoom, so
the augmentations here are the ones that survive it: non-uniform aspect distortion,
crops that clip the body, rotation, and occlusion that changes which landmarks are
found.

Each sample records the source image index as `group`, so the experiment can split
train/val on originals and never leak an augmented copy across the boundary.

Run from the repo root:
    python -m training.extract_landmarks_aug
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from common.landmarks import normalize_landmarks
from common.blazepose import PoseDetector, world_landmarks
from common.dataset import pose_folders

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports" / "landmark_features_aug.json"

MIN_VISIBILITY = 0.3
# augmented copies per original, on top of the clean one
AUG_PER_IMAGE = 3
# seed for the augmentation RNG, so the cache is reproducible
SEED = 0
# crop output size fed to BlazePose. Absolute pixels do not reach the landmark
# feature (it works in normalized coords), the crop's aspect is what matters
CROP_SIZE = 512


def mean_visibility(landmarks) -> float:
    return sum(lm[3] for lm in landmarks) / len(landmarks)


def _augment() -> v2.Transform:
    """Light image-space augmentation applied before BlazePose runs.

    RandomResizedCrop with a wide, non-square aspect range gives both the crop and
    the stretch/squash. The rotation stays small because BlazePose is trained on
    upright bodies. RandomErasing is the occlusion mask, dropping a patch so the
    detector loses some landmarks the way a limb out of frame would
    """
    return v2.Compose([
        v2.RandomResizedCrop(
            size=CROP_SIZE,
            scale=(0.7, 1.0),
            ratio=(0.6, 1.7),  # non-uniform aspect: the stretch/squash
            antialias=True,
        ),
        v2.RandomRotation(degrees=15),
        v2.RandomErasing(p=0.5, scale=(0.02, 0.15), value=0),
    ])


def _detect_feature(detect, rgb: np.ndarray):
    """Normalized feature for an RGB frame, or None if no usable pose was found."""
    result = detect.detect_rgb(rgb)
    lms = world_landmarks(result)
    if lms is None or mean_visibility(lms) < MIN_VISIBILITY:
        return None
    return normalize_landmarks(lms)


def main() -> None:
    torch.manual_seed(SEED)
    augment = _augment()
    folders = pose_folders()
    samples: list[dict] = []
    group = 0  # source-image id, so augmented copies stay with their original

    with PoseDetector() as detect:
        for label, files in folders:
            kept = clean = 0
            for f in files:
                rgb = np.asarray(Image.open(f).convert("RGB"))

                # the clean original, marked so a val split can prefer originals
                feat = _detect_feature(detect, rgb)
                if feat is not None:
                    samples.append(
                        {"label": label, "feature": feat, "group": group, "aug": False}
                    )
                    kept += 1
                    clean += 1

                # augmented variants: warp the frame, then detect
                tensor = v2.functional.to_image(rgb)
                for _ in range(AUG_PER_IMAGE):
                    aug = v2.functional.to_pil_image(augment(tensor))
                    feat = _detect_feature(detect, np.asarray(aug))
                    if feat is not None:
                        samples.append(
                            {"label": label, "feature": feat, "group": group, "aug": True}
                        )
                        kept += 1
                group += 1
            print(f"{label}: {clean}/{len(files)} clean, {kept} total with aug")

    labels = [label for label, _ in folders]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"labels": labels, "samples": samples}))
    n_clean = sum(1 for s in samples if not s["aug"])
    print(f"\nwrote {len(samples)} features ({n_clean} clean, "
          f"{len(samples) - n_clean} augmented) to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
