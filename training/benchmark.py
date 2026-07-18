"""Accuracy and latency comparison across the exported experiments.

For each exported ONNX model this reproduces its validation split (same seed),
scores top-1 accuracy on the held-out items, and times per-inference latency.
Landmark and image models are scored on the same fold, so the accuracy numbers
compare directly. A model with no export is skipped.

Run from the repo root:
    python -m training.benchmark
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from training.config import EXPORTS, stratified_split
from training.experiments.landmark_mlp import load_features
from training.experiments.landmark_aug import FEATURES as AUG_FEATURES, load_samples, _group_split
from training.image_data import PoseImageDataset, eval_transform

# How many timed inferences to run past the warm-up, for the latency percentiles
WARMUP = 5
TIMED_RUNS = 50


@dataclass
class Evaluation:
    """One model's held-out predictions and latency, enough for every metric.

    y_true, y_pred, and probs are aligned per validation item, so a caller can
    build a confusion matrix or a per-class report without re-running inference
    """

    name: str
    labels: list[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    probs: np.ndarray
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    @property
    def val_accuracy(self) -> float:
        return float((self.y_pred == self.y_true).mean())


# BenchResult stays the summary view the CLI table prints
@dataclass
class BenchResult:
    name: str
    num_classes: int
    val_accuracy: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float


def _to_result(ev: Evaluation) -> BenchResult:
    return BenchResult(
        ev.name, ev.num_classes, ev.val_accuracy,
        ev.latency_mean_ms, ev.latency_p50_ms, ev.latency_p95_ms,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    exps = np.exp(logits - logits.max())
    return exps / exps.sum()


def _time_inference(session, feed_one) -> tuple[float, float, float]:
    """Mean, p50, p95 latency in ms over TIMED_RUNS single-item inferences.

    feed_one is a zero-arg callable returning one input feed dict, so each model
    times its own input shape. Warm-up runs first, since the first call pays a
    one-time graph setup that would skew the percentiles
    """
    for _ in range(WARMUP):
        session.run(None, feed_one())
    times = []
    for _ in range(TIMED_RUNS):
        feed = feed_one()
        start = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - start) * 1000)
    arr = np.array(times)
    return float(arr.mean()), float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


def _run_feature_val(session, feats):
    """Predict over a list of landmark feature vectors, timing one inference.

    Returns (y_pred, probs, latency). Shared by the two landmark models, which
    differ only in how their val set is built (plain stratified split vs the group
    split of the augmented cache)
    """
    input_name = session.get_inputs()[0].name
    y_pred, probs = [], []
    for feat in feats:
        arr = np.array([feat], dtype=np.float32)
        logits = session.run(None, {input_name: arr})[0][0]
        probs.append(_softmax(logits))
        y_pred.append(int(logits.argmax()))
    sample = np.array([feats[0]], dtype=np.float32)
    mean, p50, p95 = _time_inference(session, lambda: {input_name: sample})
    return np.array(y_pred), np.array(probs), (mean, p50, p95)


def evaluate_landmark(seed: int, val_split: float) -> Evaluation | None:
    onnx_path = EXPORTS / "pose_classifier.onnx"
    features_path = EXPORTS / "landmark_features.json"
    if not onnx_path.exists() or not features_path.exists():
        return None

    X, y, labels = load_features(features_path)
    _, val_idx = stratified_split(y, val_split, seed)
    feats = [X[i] for i in val_idx]
    y_true = np.array([y[i] for i in val_idx])
    session = ort.InferenceSession(str(onnx_path))
    y_pred, probs, (mean, p50, p95) = _run_feature_val(session, feats)
    return Evaluation("landmark", labels, y_true, y_pred, probs, mean, p50, p95)


def evaluate_landmark_aug(seed: int, val_split: float) -> Evaluation | None:
    onnx_path = EXPORTS / "landmark_aug.onnx"
    if not onnx_path.exists() or not AUG_FEATURES.exists():
        return None

    samples, labels = load_samples(AUG_FEATURES)
    # same group split the experiment used, so val is the clean held-out originals
    _, val_s = _group_split(samples, val_split, seed)
    feats = [s["feature"] for s in val_s]
    y_true = np.array([s["y"] for s in val_s])
    session = ort.InferenceSession(str(onnx_path))
    y_pred, probs, (mean, p50, p95) = _run_feature_val(session, feats)
    return Evaluation("landmark_aug", labels, y_true, y_pred, probs, mean, p50, p95)


def evaluate_image(name: str, onnx_name: str, seed: int, val_split: float) -> Evaluation | None:
    onnx_path = EXPORTS / onnx_name
    meta_path = EXPORTS / f"{name}_meta.json"
    if not onnx_path.exists() or not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    size = meta["image_size"]
    base = PoseImageDataset(transform=eval_transform(size))
    _, val_idx = stratified_split(base.labels, val_split, seed)
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    y_true, y_pred, probs = [], [], []
    first_input = None
    for i in val_idx:
        x, label = base[i]
        arr = x.unsqueeze(0).numpy()
        if first_input is None:
            first_input = arr
        logits = session.run(None, {input_name: arr})[0][0]
        probs.append(_softmax(logits))
        y_pred.append(int(logits.argmax()))
        y_true.append(label)

    mean, p50, p95 = _time_inference(session, lambda: {input_name: first_input})
    return Evaluation(
        name, base.classes, np.array(y_true), np.array(y_pred), np.array(probs), mean, p50, p95
    )


def evaluate_all(seed: int = 0, val_split: float = 0.2) -> list[Evaluation]:
    """Every exported model's held-out predictions and latency, skipping the missing."""
    evals = [
        evaluate_landmark(seed, val_split),
        evaluate_landmark_aug(seed, val_split),
        evaluate_image("image_transfer", "image_transfer.onnx", seed, val_split),
        evaluate_image("image_finetune", "image_finetune.onnx", seed, val_split),
    ]
    return [e for e in evals if e is not None]


def run(seed: int = 0, val_split: float = 0.2) -> list[BenchResult]:
    """Benchmark every exported model, skipping the ones not yet exported."""
    return [_to_result(e) for e in evaluate_all(seed, val_split)]


def print_table(results: list[BenchResult]) -> None:
    if not results:
        print("no exported models found, train and export an experiment first")
        return
    header = f"{'model':16} {'classes':>7} {'val acc':>8} {'mean ms':>8} {'p50 ms':>8} {'p95 ms':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.name:16} {r.num_classes:>7} {r.val_accuracy:>8.3f} "
              f"{r.latency_mean_ms:>8.2f} {r.latency_p50_ms:>8.2f} {r.latency_p95_ms:>8.2f}")


def main() -> None:
    print_table(run())


if __name__ == "__main__":
    main()
