"""Process a YouTube video into a follow-along track for the lotus-pose frontend.

Downloads the video, samples frames, runs BlazePose over each, and writes a track
JSON the app plays against: per-frame joint angles (for scoring), 2D landmarks
(for the skeleton overlay), and a classified pose label. The track lands in the
frontend's public/tracks/ and the manifest is updated, so the app is fully static
with no backend.

Angles, landmark normalization, and the classifier are the same ones training
uses (model/angles, model/landmarks, exports/pose_classifier.onnx), so a track
matches what the app computes from the live webcam.

Run from the repo root:
    python scripts/process_video.py "https://youtu.be/VIDEO_ID"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.angles import JOINT_NAMES, compute_angles
from model.landmarks import normalize_landmarks
from blazepose import PoseDetector, LANDMARKER_PATH, world_landmarks
from track_common import FRONTEND_TRACKS, youtube_url, track_path, resolve_video_id

CLASSIFIER_PATH = ROOT / "exports" / "pose_classifier.onnx"
META_PATH = ROOT / "exports" / "model_meta.json"
SKELETON_DIR = ROOT / "exports" / "skeletons"

SAMPLE_FPS = 5
MAX_DURATION_SEC = 60 * 60
MIN_VISIBILITY = 0.5
MIN_LABEL_CONFIDENCE = 0.5
# A pose the classifier's top guess named but wasn't confident about, worth
# surfacing as a possible pose the dataset may be missing
LOW_CONFIDENCE_FLOOR = 0.25
# 4 decimals places a point on a 1080p frame and keeps the track from ballooning
# once 33 xy pairs land on every frame
COORD_PRECISION = 4

def _download(video_id: str, dest_dir: Path) -> tuple[Path, str]:
    """Download a YouTube video by id, returning its path and title."""
    from yt_dlp import YoutubeDL

    def duration_guard(info, *, incomplete=False):
        d = info.get("duration")
        if d is None:
            return "unknown duration"
        if d > MAX_DURATION_SEC:
            return f"{d}s is over the {MAX_DURATION_SEC}s cap"
        return None

    url = youtube_url(video_id)
    opts = {
        "format": "mp4/best",
        "outtmpl": str(dest_dir / "video.%(ext)s"),
        "match_filter": duration_guard,
        "quiet": True,
        "noprogress": True,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info)), info.get("title") or "Untitled"


def _sample_frames(video: Path, frame_dir: Path) -> None:
    """Sample the video to numbered JPEG frames at SAMPLE_FPS."""
    subprocess.run(
        [
            "ffmpeg", "-i", str(video),
            "-vf", f"fps={SAMPLE_FPS}",
            "-q:v", "3",
            str(frame_dir / "%05d.jpg"),
        ],
        check=True,
        capture_output=True,
    )


def _load_classifier() -> tuple[ort.InferenceSession, list[str]]:
    labels = json.loads(META_PATH.read_text())["labels"]
    session = ort.InferenceSession(str(CLASSIFIER_PATH))
    return session, labels


def _classify(session, labels, world) -> tuple[str, float]:
    """Top pose id and its softmax confidence for a frame's world landmarks."""
    feat = np.array([normalize_landmarks(world)], dtype=np.float32)
    logits = session.run(None, {session.get_inputs()[0].name: feat})[0][0]
    exps = np.exp(logits - logits.max())
    probs = exps / exps.sum()
    best = int(probs.argmax())
    return labels[best], float(probs[best])


def _frame_angles(world) -> tuple[list, list]:
    """8-angle vector and visibility flags in JOINT_NAMES order."""
    measured = compute_angles(world)
    angles, visible = [], []
    for name in JOINT_NAMES:
        a = measured[name]
        ok = a["angle"] is not None and a["visibility"] >= MIN_VISIBILITY
        angles.append(a["angle"] if ok else None)
        visible.append(ok)
    return angles, visible


def _flatten_screen(screen) -> list[float]:
    """Flatten the 33 image-space points to a rounded [x0,y0,x1,y1,...] vector."""
    out: list[float] = []
    for lm in screen:
        out.append(round(lm.x, COORD_PRECISION))
        out.append(round(lm.y, COORD_PRECISION))
    return out


def _empty_frame(t: float) -> dict:
    return {
        "t": t,
        "angles": [None] * len(JOINT_NAMES),
        "visible": [False] * len(JOINT_NAMES),
        "lm": None,
        "pose": None,
    }


def _extract(frame_dir: Path) -> tuple[list[dict], list, int, int]:
    """Run BlazePose over the sampled frames.

    Returns the track frames, the per-frame world landmarks (for the skeleton
    cache, None where no pose was found), and the frame pixel size
    """
    session, labels = _load_classifier()
    files = sorted(f for f in frame_dir.iterdir() if f.suffix.lower() == ".jpg")
    frames: list[dict] = []
    worlds: list = []
    frame_w = frame_h = 0

    with PoseDetector() as detect:
        for i, f in enumerate(files):
            t = i / SAMPLE_FPS
            try:
                result = detect(f)
            except Exception as e:
                print(f"warning: {f.name}: {e}", file=sys.stderr)
                frames.append(_empty_frame(t))
                worlds.append(None)
                continue

            if frame_w == 0:
                with Image.open(f) as img:
                    frame_w, frame_h = img.size

            world = world_landmarks(result)
            if world and result.pose_landmarks:
                angles, visible = _frame_angles(world)
                label, conf = _classify(session, labels, world)
                frames.append({
                    "t": t,
                    "angles": angles,
                    "visible": visible,
                    "lm": _flatten_screen(result.pose_landmarks[0]),
                    "pose": label if conf >= MIN_LABEL_CONFIDENCE else None,
                })
                worlds.append(world)
            else:
                frames.append(_empty_frame(t))
                worlds.append(None)

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(files)} frames")

    return frames, worlds, frame_w, frame_h


def _update_manifest(video_id: str, title: str) -> None:
    """Add or replace this video's entry in the frontend manifest."""
    manifest_path = FRONTEND_TRACKS / "manifest.json"
    entries = []
    if manifest_path.exists():
        entries = [e for e in json.loads(manifest_path.read_text()) if e["videoId"] != video_id]
    entries.append({
        "videoId": video_id,
        "title": title,
        "track": f"/tracks/{video_id}.json",
    })
    manifest_path.write_text(json.dumps(entries, indent=2) + "\n")


def _save_skeleton(video_id: str, worlds: list) -> Path:
    """Cache the per-frame world landmarks so the classifier can re-run offline.

    The classifier keys on world landmarks, which the track JSON does not carry.
    Saving them lets reclassify.py relabel a track without re-downloading the
    video. Rounded to keep the file small, None where no pose was found
    """
    SKELETON_DIR.mkdir(parents=True, exist_ok=True)
    frames = [
        [[round(v, COORD_PRECISION) for v in lm] for lm in world] if world else None
        for world in worlds
    ]
    out = SKELETON_DIR / f"{video_id}.json"
    out.write_text(json.dumps({"fps": SAMPLE_FPS, "frames": frames}))
    return out


def _pose_summary(frames: list[dict], worlds: list) -> dict:
    """Held poses and low-confidence visual guesses for the batch report.

    `detected` is the poses the classifier held with confidence, by frame count.
    `possible` is top guesses that cleared LOW_CONFIDENCE_FLOOR but not the label
    threshold, poses the model half-saw that the dataset may be missing
    """
    session, labels = _load_classifier()
    detected: dict[str, int] = {}
    possible: dict[str, int] = {}
    for frame, world in zip(frames, worlds):
        if frame["pose"]:
            detected[frame["pose"]] = detected.get(frame["pose"], 0) + 1
        elif world is not None:
            label, conf = _classify(session, labels, world)
            if conf >= LOW_CONFIDENCE_FLOOR:
                possible[label] = possible.get(label, 0) + 1
    return {"detected": detected, "possible": possible}


def process(source: str) -> dict:
    """Process a YouTube URL/id into a static track.

    Writes the track JSON, caches the skeleton for offline re-classifying, and
    returns a summary: video id, title, and the detected and possible poses
    """
    for path in (LANDMARKER_PATH, CLASSIFIER_PATH, META_PATH):
        if not path.exists():
            raise SystemExit(f"missing {path.relative_to(ROOT)}, run the training scripts first")

    video_id = resolve_video_id(source)
    if video_id is None:
        raise SystemExit(f"not a YouTube URL or id: {source}")
    FRONTEND_TRACKS.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        frame_dir = tmp_dir / "frames"
        frame_dir.mkdir()

        video, title = _download(video_id, tmp_dir)

        print(f"Sampling frames from {video.name}...")
        _sample_frames(video, frame_dir)
        print("Running BlazePose over the frames...")
        frames, worlds, frame_w, frame_h = _extract(frame_dir)

    track = {
        "title": title,
        "durationSec": len(frames) / SAMPLE_FPS,
        "fps": SAMPLE_FPS,
        "frameWidth": frame_w,
        "frameHeight": frame_h,
        "frames": frames,
        "videoId": video_id,
    }
    out_file = track_path(video_id)
    out_file.write_text(json.dumps(track))
    _update_manifest(video_id, title)
    skeleton = _save_skeleton(video_id, worlds)
    summary = _pose_summary(frames, worlds)

    measured = sum(1 for f in frames if f["pose"])
    print(f"\nWrote {len(frames)} frames ({measured} labeled) to {out_file}")
    print(f"Cached skeleton to {skeleton.relative_to(ROOT)}")
    return {"videoId": video_id, "title": title, **summary}


def print_poses(summary: dict) -> None:
    """Detected, possible-visual, and (if present) possible-spoken poses."""
    def line(counts: dict) -> str:
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        return ", ".join(f"{p} ({n})" for p, n in ranked) or "none"

    print(f"\n{summary['title']} ({summary['videoId']})")
    print("  detected:", line(summary["detected"]))
    print("  possible (visual):", line(summary["possible"]))
    if "spoken" in summary:
        print("  possible (spoken):", line(summary["spoken"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a YouTube video into a follow-along track")
    parser.add_argument("source", help="YouTube URL or id")
    args = parser.parse_args()
    print_poses(process(args.source))


if __name__ == "__main__":
    main()
