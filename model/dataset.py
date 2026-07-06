"""Pose image dataset and transforms.

Loads photos from data/yoga-poses/<pose-id>/, one folder per pose, with a
stratified train/val split. Images are letterboxed (padded to square with black,
then resized) so nothing is cropped and the whole pose stays in frame, matching
what the browser does at inference. Augmentation softens the gap between studio
photos at train time and webcam frames at inference time.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Which files count as images, so a stray folder (e.g. the sorty .sorty cache)
# is never picked up as a class
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


class LetterboxPad:
    """Pad a PIL image to square with black, then resize to a fixed size.

    Keeps the whole pose in frame with no crop, so the aspect ratio is preserved
    and the model never loses a limb to a center crop. The browser reproduces the
    same geometry with a canvas cover-to-contain draw
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        side = max(w, h)
        padded = Image.new("RGB", (side, side), (0, 0, 0))
        padded.paste(img, ((side - w) // 2, (side - h) // 2))
        return padded.resize((self.size, self.size), Image.BILINEAR)


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            LetterboxPad(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            # fill=0 matches the black letterbox, so rotated-in corners blend in
            transforms.RandomRotation(15, fill=0),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            LetterboxPad(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _is_image(path: str) -> bool:
    return path.lower().endswith(IMG_EXTS)


class PoseFolder(ImageFolder):
    """ImageFolder that ignores dot-directories (e.g. the sorty .sorty cache)."""

    def find_classes(self, directory: str):
        classes = sorted(
            e.name
            for e in os.scandir(directory)
            if e.is_dir() and not e.name.startswith(".")
        )
        return classes, {name: i for i, name in enumerate(classes)}


def load_split(
    data_dir: Path, val_fraction: float = 0.2, seed: int = 0
) -> tuple[Subset, Subset, list[str]]:
    """Return stratified train and val subsets plus the class labels in order."""
    base = PoseFolder(str(data_dir), is_valid_file=_is_image)
    labels = list(base.classes)
    targets = [s[1] for s in base.samples]

    train_idx, val_idx = train_test_split(
        range(len(targets)),
        test_size=val_fraction,
        random_state=seed,
        stratify=targets,
    )

    train_ds = PoseFolder(
        str(data_dir), transform=train_transform(), is_valid_file=_is_image
    )
    val_ds = PoseFolder(
        str(data_dir), transform=eval_transform(), is_valid_file=_is_image
    )
    return Subset(train_ds, train_idx), Subset(val_ds, val_idx), labels
