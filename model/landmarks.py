"""Landmark feature extraction for the pose classifier.

Turns BlazePose 33 world landmarks into a translation- and scale-invariant
vector the classifier trains on. Backgrounds, lighting, and clothing never enter
the feature, only body geometry, so it transfers from studio photos to a live
webcam far better than a raw-image classifier.

A port of the app's src/lib/poseLandmarks.ts. The two must stay in step, since
the app feeds live BlazePose landmarks through the same normalization at runtime.
A shared fixture test (tests/test_landmark_parity) guards that.
"""

from __future__ import annotations

import math

# BlazePose landmark indices used to center and scale the body
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

NUM_LANDMARKS = 33
FEATURE_DIM = NUM_LANDMARKS * 3


def _mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


def normalize_landmarks(landmarks) -> list[float]:
    """Recenter on the hips and scale by torso length, then flatten to xyz.

    Each landmark is (x, y, z) or (x, y, z, visibility). Returns a FEATURE_DIM
    vector. Scale is the hip-center to shoulder-center distance, so a person near
    or far from the camera yields the same feature.
    """
    hip = _mid(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
    shoulder = _mid(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
    torso = math.sqrt(
        sum((shoulder[i] - hip[i]) ** 2 for i in range(3))
    )
    # Guard a degenerate torso (landmarks collapsed) so we never divide by zero
    scale = torso if torso > 1e-6 else 1.0

    feat: list[float] = []
    for lm in landmarks:
        feat.append((lm[0] - hip[0]) / scale)
        feat.append((lm[1] - hip[1]) / scale)
        feat.append((lm[2] - hip[2]) / scale)
    return feat


def mirror_landmarks(landmarks) -> list[tuple]:
    """Left-right flip: negate x and swap left/right landmark pairs.

    Makes the classifier symmetric so a pose facing either way reads the same,
    matching how a person faces a mirrored webcam.
    """
    flipped = [(-lm[0],) + tuple(lm[1:]) for lm in landmarks]
    for left, right in _LR_PAIRS:
        flipped[left], flipped[right] = flipped[right], flipped[left]
    return flipped


def mirror_feature(feature: list[float]) -> list[float]:
    """Mirror a normalized feature vector, matching mirror_landmarks in xyz space.

    Hip-centering and torso scale are mirror-symmetric, so flipping the feature is
    the same as normalizing a mirrored pose: negate every x, swap left/right xyz
    triples. Used for training-time augmentation off the cached features.
    """
    pts = [feature[i : i + 3] for i in range(0, len(feature), 3)]
    pts = [[-x, y, z] for x, y, z in pts]
    for left, right in _LR_PAIRS:
        pts[left], pts[right] = pts[right], pts[left]
    return [v for p in pts for v in p]


# BlazePose left/right landmark index pairs, swapped on a mirror
_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
    (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]
