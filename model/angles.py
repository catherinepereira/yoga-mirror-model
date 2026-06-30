"""Joint-angle extraction from BlazePose world landmarks.

A Python port of the app's src/lib/angles.ts and joints.ts. The reference-pose
builder uses it to turn pose photos into target angles. It must stay in step with
the TypeScript version, since the app scores against angles computed the same way.
A shared fixture test (tests/test_angle_parity) guards that.
"""

from __future__ import annotations

import math

# BlazePose 33-landmark indices for the joints scored
LM = {
    "leftShoulder": 11,
    "rightShoulder": 12,
    "leftElbow": 13,
    "rightElbow": 14,
    "leftWrist": 15,
    "rightWrist": 16,
    "leftHip": 23,
    "rightHip": 24,
    "leftKnee": 25,
    "rightKnee": 26,
    "leftAnkle": 27,
    "rightAnkle": 28,
}

# Joint name in JOINT_NAMES order -> (a, b, c) landmark indices, angle at b
JOINTS: dict[str, tuple[int, int, int]] = {
    "leftElbow": (LM["leftShoulder"], LM["leftElbow"], LM["leftWrist"]),
    "rightElbow": (LM["rightShoulder"], LM["rightElbow"], LM["rightWrist"]),
    "leftShoulder": (LM["leftElbow"], LM["leftShoulder"], LM["leftHip"]),
    "rightShoulder": (LM["rightElbow"], LM["rightShoulder"], LM["rightHip"]),
    "leftHip": (LM["leftShoulder"], LM["leftHip"], LM["leftKnee"]),
    "rightHip": (LM["rightShoulder"], LM["rightHip"], LM["rightKnee"]),
    "leftKnee": (LM["leftHip"], LM["leftKnee"], LM["leftAnkle"]),
    "rightKnee": (LM["rightHip"], LM["rightKnee"], LM["rightAnkle"]),
}

JOINT_NAMES = list(JOINTS.keys())

RAD_TO_DEG = 180 / math.pi


def angle_at(a, b, c) -> float | None:
    """Angle in degrees at vertex b formed by a-b-c, or None if degenerate."""
    v1 = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    v2 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    if mag1 < 1e-9 or mag2 < 1e-9:
        return None
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    cos = min(1.0, max(-1.0, dot / (mag1 * mag2)))
    return math.acos(cos) * RAD_TO_DEG


def visibility_of(lm) -> float:
    return lm[3] if len(lm) > 3 and lm[3] is not None else 1.0


def compute_angles(landmarks) -> dict[str, dict]:
    """Per-joint angle and min landmark visibility, keyed by JOINT_NAMES.

    Each landmark is (x, y, z) or (x, y, z, visibility). A joint whose landmarks
    are absent gets a None angle.
    """
    out: dict[str, dict] = {}
    for name in JOINT_NAMES:
        ai, bi, ci = JOINTS[name]
        if ai >= len(landmarks) or bi >= len(landmarks) or ci >= len(landmarks):
            out[name] = {"angle": None, "visibility": 0.0}
            continue
        a, b, c = landmarks[ai], landmarks[bi], landmarks[ci]
        out[name] = {
            "angle": angle_at(a, b, c),
            "visibility": min(visibility_of(a), visibility_of(b), visibility_of(c)),
        }
    return out
