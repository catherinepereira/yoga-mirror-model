"""Accuracy and latency thresholds for each exported model.

Runs the benchmark and asserts a floor on accuracy and a ceiling on latency per
model. A model that has not been exported yet is skipped, so the suite is runnable
before every experiment has been trained.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training import benchmark

# A model well above chance on ~19 classes, and a latency a webcam app can afford
# The image models run a heavier graph, so they get a looser latency ceiling
MIN_ACCURACY = 0.5
LATENCY_CEILING_MS = {
    "landmark": 5.0,
    "landmark_aug": 5.0,
    "image_transfer": 100.0,
    "image_finetune": 100.0,
}
MODELS = ["landmark", "landmark_aug", "image_transfer", "image_finetune"]


@pytest.fixture(scope="module")
def results():
    found = benchmark.run()
    if not found:
        pytest.skip("no exported models to benchmark")
    return {r.name: r for r in found}


@pytest.mark.parametrize("name", MODELS)
def test_accuracy_above_floor(results, name):
    if name not in results:
        pytest.skip(f"{name} not exported")
    assert results[name].val_accuracy >= MIN_ACCURACY


@pytest.mark.parametrize("name", MODELS)
def test_latency_below_ceiling(results, name):
    if name not in results:
        pytest.skip(f"{name} not exported")
    assert results[name].latency_p95_ms <= LATENCY_CEILING_MS[name]
