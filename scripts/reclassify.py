"""Re-run the classifier over a cached skeleton, without re-downloading the video.

process_video caches each video's world landmarks to exports/skeletons/<id>.json.
This reads that cache, re-runs the current classifier and angle math over it, and
rewrites the pose labels and angles in the frontend track. Use it after
retraining the model or changing the angle math to relabel existing tracks.

Run from the repo root:
    python scripts/reclassify.py VIDEO_ID          # one video
    python scripts/reclassify.py --all             # every cached skeleton
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from process_video import (
    SKELETON_DIR,
    MIN_LABEL_CONFIDENCE,
    _classify,
    _frame_angles,
    _load_classifier,
    _segment,
)
from track_common import track_path


def reclassify(video_id: str) -> dict:
    """Relabel one track from its cached skeleton. Returns a per-pose frame count."""
    skeleton_path = SKELETON_DIR / f"{video_id}.json"
    if not skeleton_path.exists():
        raise SystemExit(f"no cached skeleton for {video_id}, process the video first")
    track_file = track_path(video_id)
    if not track_file.exists():
        raise SystemExit(f"no track for {video_id}")

    worlds = json.loads(skeleton_path.read_text())["frames"]
    track = json.loads(track_file.read_text())
    frames = track["frames"]
    if len(worlds) != len(frames):
        raise SystemExit(
            f"{video_id}: {len(worlds)} cached frames vs {len(frames)} track frames"
        )

    session, labels = _load_classifier()
    counts: dict[str, int] = {}
    for frame, world in zip(frames, worlds):
        if world is None:
            frame["pose"] = None
            continue
        angles, visible = _frame_angles(world)
        label, conf = _classify(session, labels, world)
        frame["angles"] = angles
        frame["visible"] = visible
        frame["pose"] = label if conf >= MIN_LABEL_CONFIDENCE else None
        if frame["pose"]:
            counts[frame["pose"]] = counts.get(frame["pose"], 0) + 1

    # Smooth the labels and rebuild the timeline segments, same as process_video
    track["segments"] = _segment(frames)
    track_file.write_text(json.dumps(track))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-classify a track from its cached skeleton")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("video_id", nargs="?", help="Video id to re-classify")
    group.add_argument("--all", action="store_true", help="Re-classify every cached skeleton")
    args = parser.parse_args()

    ids = (
        [p.stem for p in sorted(SKELETON_DIR.glob("*.json"))]
        if args.all
        else [args.video_id]
    )
    if not ids:
        raise SystemExit(f"no cached skeletons in {SKELETON_DIR}")

    for video_id in ids:
        counts = reclassify(video_id)
        top = sorted(counts.items(), key=lambda kv: -kv[1])
        print(f"{video_id}: " + (", ".join(f"{p} ({n})" for p, n in top) or "no poses"))


if __name__ == "__main__":
    main()
