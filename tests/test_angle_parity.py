"""The Python angle math must match the app's TypeScript computeAngles.

The fixture is generated from the TS implementation (the app's source of truth)
by lotus-pose/scripts/gen-angle-fixture.mts. If this drifts, the reference
targets this repo exports would not match how the app scores at runtime.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.angles import compute_angles

FIXTURE = ROOT / "tests" / "fixtures" / "angle_parity.json"


def test_python_angles_match_typescript():
    data = json.loads(FIXTURE.read_text())
    landmarks = [tuple(lm) for lm in data["landmarks"]]
    expected = data["angles"]

    measured = compute_angles(landmarks)
    for name, exp in expected.items():
        got = measured[name]["angle"]
        assert got is not None, f"{name} should be measurable"
        assert abs(got - exp) < 1e-6, f"{name}: python {got} vs ts {exp}"
