"""Pose image dataset and transforms.

Loads photos from data/raw/<pose-id>/, one folder per pose, with a stratified
train/val split. Augmentation softens the gap between studio photos at train time
and webcam frames at inference time.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_split(
    data_dir: Path, val_fraction: float = 0.2, seed: int = 0
) -> tuple[Subset, Subset, list[str]]:
    """Return stratified train and val subsets plus the class labels in order."""
    base = ImageFolder(str(data_dir))
    labels = list(base.classes)
    targets = [s[1] for s in base.samples]

    train_idx, val_idx = train_test_split(
        range(len(targets)),
        test_size=val_fraction,
        random_state=seed,
        stratify=targets,
    )

    train_ds = ImageFolder(str(data_dir), transform=train_transform())
    val_ds = ImageFolder(str(data_dir), transform=eval_transform())
    return Subset(train_ds, train_idx), Subset(val_ds, val_idx), labels
