"""Image dataset and transforms for the two image experiments.

Both image experiments read the same data/yoga-poses/<pose-id>/ folders the
landmark pipeline uses, so the classes are folder-driven and the comparison runs
on the same photos. Training augments, evaluation and prediction only preprocess.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from common.dataset import pose_folders

# ImageNet statistics the MobileNet backbone was pretrained with. The same values
# go into model_meta.json so any consumer normalizes an input frame identically
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform(size: int) -> transforms.Compose:
    """Augment then normalize, for the training split only.

    RandomResizedCrop and the flips and jitter widen the studio photos toward the
    framing and lighting of a live webcam. The horizontal flip is the image analog
    of the landmark mirror aug, so a pose facing either way trains the same
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int) -> transforms.Compose:
    """Resize and normalize with no augmentation, for validation and prediction.

    The exporter and the benchmark preprocess with this, so a served model sees the
    same pixel distribution it was validated on. Randomness at inference would make
    the score irreproducible and the prediction jittery
    """
    resize = int(size * 1.14)  # resize a touch larger, then center-crop to size
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class PoseImageDataset(Dataset):
    """Every image under data/yoga-poses/, labeled by its folder.

    The transform is applied lazily per __getitem__, so the same file list can back
    a training set (with augmentation) and a validation set (eval transform only)
    by passing a different transform
    """

    def __init__(self, transform):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        for label, files in pose_folders():
            self.classes.append(label)
            class_idx = len(self.classes) - 1
            for f in files:
                self.samples.append((f, class_idx))

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[i]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


class _TransformSubset(Dataset):
    """A subset of a PoseImageDataset with its own transform.

    Lets the train and val splits over one file list use augmentation and the eval
    transform independently, without loading the images twice
    """

    def __init__(self, base: PoseImageDataset, indices: list[int], transform):
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, label = self.base.samples[self.indices[i]]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
