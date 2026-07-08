"""Process a video into a follow-along track for the yoga-mirror frontend.

Downloads a YouTube URL (or reads a local file), samples frames, runs BlazePose
over each, and writes a track JSON the app plays against: per-frame joint angles
(for scoring), 2D landmarks (for the skeleton overlay), and a classified pose
label. The track lands in the frontend's public/tracks/ and the manifest is
updated, so the app is fully static with no backend.

Angles, landmark normalization, and the classifier are the same ones training
uses (model/angles, model/landmarks, exports/pose_classifier.onnx), so a track
matches what the app computes from the live webcam.

Run from the repo root:
    python scripts/process_video.py "https://youtu.be/VIDEO_ID"
    python scripts/process_video.py path/to/local.mp4 --title "Morning Flow"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.angles import JOINT_NAMES, compute_angles
from model.landmarks import normalize_landmarks
from blazepose import PoseDetector, LANDMARKER_PATH, world_landmarks
from track_common import FRONTEND_TRACKS, youtube_url, track_path

CLASSIFIER_PATH = ROOT / "exports" / "pose_classifier.onnx"
META_PATH = ROOT / "exports" / "model_meta.json"

SAMPLE_FPS = 5
MAX_DURATION_SEC = 60 * 60
MIN_VISIBILITY = 0.5
MIN_LABEL_CONFIDENCE = 0.5
# 4 decimals places a point on a 1080p frame and keeps the track from ballooning
# once 33 xy pairs land on every frame
COORD_PRECISION = 4

_VIDEO_ID = re.compile(r"^[\w-]{11}$")
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def _video_id(url: str) -> str | None:
    """Resolve an 11-character YouTube id from a URL, or None for a local file."""
    raw = url.strip()
    if _VIDEO_ID.fullmatch(raw):
        return raw
    parsed = urlparse(raw)
    if parsed.hostname not in _YOUTUBE_HOSTS:
        return None
    if parsed.hostname == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif parsed.path == "/watch":
        candidate = parse_qs(parsed.query).get("v", [""])[0]
    else:
        parts = [p for p in parsed.path.split("/") if p]
        candidate = parts[1] if len(parts) >= 2 and parts[0] in ("embed", "shorts") else ""
    return candidate if _VIDEO_ID.fullmatch(candidate) else None


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


def _classify(session, labels, world) -> str | None:
    """Top pose id for a frame's world landmarks, or None below the threshold."""
    feat = np.array([normalize_landmarks(world)], dtype=np.float32)
    logits = session.run(None, {session.get_inputs()[0].name: feat})[0][0]
    exps = np.exp(logits - logits.max())
    probs = exps / exps.sum()
    best = int(probs.argmax())
    return labels[best] if probs[best] >= MIN_LABEL_CONFIDENCE else None


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


def _extract(frame_dir: Path) -> tuple[list[dict], int, int]:
    """Run BlazePose over the sampled frames, returning frames and frame size."""
    session, labels = _load_classifier()
    files = sorted(f for f in frame_dir.iterdir() if f.suffix.lower() == ".jpg")
    frames: list[dict] = []
    frame_w = frame_h = 0

    with PoseDetector() as detect:
        for i, f in enumerate(files):
            t = i / SAMPLE_FPS
            try:
                result = detect(f)
            except Exception as e:
                print(f"warning: {f.name}: {e}", file=sys.stderr)
                frames.append(_empty_frame(t))
                continue

            if frame_w == 0:
                with Image.open(f) as img:
                    frame_w, frame_h = img.size

            world = world_landmarks(result)
            if world and result.pose_landmarks:
                angles, visible = _frame_angles(world)
                frames.append({
                    "t": t,
                    "angles": angles,
                    "visible": visible,
                    "lm": _flatten_screen(result.pose_landmarks[0]),
                    "pose": _classify(session, labels, world),
                })
            else:
                frames.append(_empty_frame(t))

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(files)} frames")

    return frames, frame_w, frame_h


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


def process(source: str, title: str | None = None) -> str:
    """Process a URL or local file into a static track. Returns the track id."""
    for path in (LANDMARKER_PATH, CLASSIFIER_PATH, META_PATH):
        if not path.exists():
            raise SystemExit(f"missing {path.relative_to(ROOT)}, run the training scripts first")

    FRONTEND_TRACKS.mkdir(parents=True, exist_ok=True)
    video_id = _video_id(source)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        frame_dir = tmp_dir / "frames"
        frame_dir.mkdir()

        if video_id:
            video, fetched_title = _download(video_id, tmp_dir)
            title = title or fetched_title
        else:
            video = Path(source)
            if not video.exists():
                raise SystemExit(f"no such file: {source}")
            video_id = video.stem
            title = title or video.stem

        print(f"Sampling frames from {video.name}...")
        _sample_frames(video, frame_dir)
        print("Running BlazePose over the frames...")
        frames, frame_w, frame_h = _extract(frame_dir)

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

    measured = sum(1 for f in frames if f["pose"])
    print(f"\nWrote {len(frames)} frames ({measured} labeled) to {out_file}")
    return video_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a video into a follow-along track")
    parser.add_argument("source", help="YouTube URL/id or a local video file path")
    parser.add_argument("--title", help="Track title (defaults to the video title or filename)")
    args = parser.parse_args()
    process(args.source, args.title)


if __name__ == "__main__":
    main()
