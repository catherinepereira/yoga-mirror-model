"""Shared dataset layout for the scripts that walk the pose photos.

extract_landmarks, extract_landmarks_aug, and viz_poses all iterate the same
data/yoga-poses/<pose-id>/ tree and filter to the same image types.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _ROOT / "data" / "yoga-poses"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def pose_folders() -> list[tuple[str, list[Path]]]:
    """Each pose folder as (label, image files), sorted, skipping dot-dirs.

    The dot-dir skip drops the sorty tool's .sorty cache
    """
    out: list[tuple[str, list[Path]]] = []
    for folder in sorted(
        p for p in DATA_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        files = sorted(
            f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXT
        )
        out.append((folder.name, files))
    return out
