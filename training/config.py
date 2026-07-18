"""Training configuration shared by the three experiments.

Each experiment builds a config instance instead of scattering constants through
its training loop, so the hyperparameters for a run live in one place and the
benchmark can reproduce a run's exact split from its config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingConfig:
    """Hyperparameters common to every experiment.

    Subclasses add the fields their model needs (image size, whether to freeze the
    backbone). seed and val_split are shared so the landmark and image runs split
    the same dataset the same way and the accuracy comparison is on matched folds
    """

    epochs: int
    lr: float
    weight_decay: float
    batch_size: int
    val_split: float = 0.2
    seed: int = 0
    device: str = field(default_factory=default_device)
    out_dir: Path = EXPORTS


def stratified_split(labels: list[int], val_split: float, seed: int):
    """Train/val index split, stratified by label, reproducible from the seed.

    Returns (train_idx, val_idx). Every experiment and the benchmark call this with
    the config's val_split and seed, so a model is always scored on the same
    held-out items it was validated on
    """
    idx = list(range(len(labels)))
    return train_test_split(
        idx, test_size=val_split, random_state=seed, stratify=labels
    )
