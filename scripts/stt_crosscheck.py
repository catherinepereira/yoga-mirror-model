"""Transcribe a video's narration and cross-check it against the visual poses.

A follow-along track already carries a per-frame pose label from the visual
classifier. This step transcribes the instructor's narration with dinscribe
(Demucs denoise, VAD, Whisper), scans it for spoken pose names, and matches each
mention against the visual label near that timestamp. The two signals confirm or
disagree.

The result is folded back into the frontend track as `spokenPoses` (name and
time) and `crossCheck` (agreements and disagreements), which the app shows when
the instructor cues a pose. Run after process_video.py has built the track.

Run from the repo root:
    python scripts/stt_crosscheck.py VIDEO_ID [whisper_model]
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_common import youtube_url, track_path

# Pose names fed to Whisper as an initial prompt, so it decodes the yoga terms
# it otherwise mangles
VOCAB_FILE = Path(__file__).parent / "pose_vocab.txt"

# dinscribe prints box-drawing characters that the default Windows console
# codec (cp1252) cannot encode. Force UTF-8 so its progress output does not
# crash the run
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# A cue is checked against the visual track over a forward-leaning window: the
# instructor names a pose, the class takes a while to settle into it, then holds
# it. Look back a little for a cue said mid-pose, look well ahead for the settle
# and hold, since the pose can arrive many seconds after its name
MATCH_LOOKBACK_SEC = 3.0
MATCH_LOOKAHEAD_SEC = 20.0

# Pose ids the classifier can emit, with the spoken phrases that name each. The
# id's own words are matched too, these add the numbering and casual variants a
# narrator actually says
POSE_ALIASES: dict[str, list[str]] = {
    "boat-pose": ["boat"],
    "camel-pose": ["camel"],
    "cat-pose": ["cat"],
    "child-pose": ["child", "childs", "child's"],
    "cobra-pose": ["cobra"],
    "cow-pose": ["cow"],
    "downward-dog": ["downward dog", "down dog", "downward facing dog"],
    "eagle-pose": ["eagle"],
    "pigeon-pose": ["pigeon"],
    "tree-pose": ["tree"],
    "triangle-pose": ["triangle"],
    "warrior-i": ["warrior one", "warrior 1", "warrior i"],
    "warrior-ii": ["warrior two", "warrior 2", "warrior ii"],
}


def _download_audio(video_id: str, dest_dir: Path) -> Path:
    """Download the video's audio track as a wav for transcription."""
    from yt_dlp import YoutubeDL

    url = youtube_url(video_id)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "audio.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "quiet": True,
        "noprogress": True,
    }
    with YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)
    wav = dest_dir / "audio.wav"
    if not wav.exists():
        raise RuntimeError("audio extraction produced no wav")
    return wav


def _transcribe(wav: Path, work_dir: Path, model: str) -> list[dict]:
    """Run dinscribe over the wav, returning its transcription entries.

    Import is local so the heavy Whisper and Demucs stack only loads when STT
    actually runs.

    whole_file over per-VAD-segment: a yoga instructor cues poses continuously
    and the VAD path dropped nearly all of it (one mention per video). The vocab
    file biases Whisper toward the pose names, which turns "downward duck" back
    into "downward dog" often enough to matter
    """
    from dinscribe import process_file, PipelineConfig

    config = PipelineConfig.from_dict(
        {
            "transcribe": {
                "model": model,
                "whole_file": True,
                "word_timestamps": True,
                "vocab_file": str(VOCAB_FILE) if VOCAB_FILE.exists() else None,
            }
        }
    )
    ok = process_file(wav, work_dir, config=config, force=True)
    if not ok:
        raise RuntimeError("dinscribe failed")

    result = work_dir / wav.stem / f"{wav.stem}_transcription.json"
    data = json.loads(result.read_text(encoding="utf-8"))
    return data.get("transcription", [])


def _find_pose_mentions(entries: list[dict]) -> list[dict]:
    """Scan transcript entries for pose names, timestamped at the entry start.

    A "pose" or "posture" suffix on the alias is optional, so "into triangle"
    and "triangle pose" both match. The longest alias wins per entry so
    "downward dog" is not also counted as no match
    """
    mentions: list[dict] = []
    for entry in entries:
        text = entry.get("text", "").lower()
        start = entry.get("timestamp", {}).get("start", 0.0)
        for pose_id, aliases in POSE_ALIASES.items():
            for phrase in sorted(aliases, key=len, reverse=True):
                pattern = rf"\b{re.escape(phrase)}(\s+(pose|posture))?\b"
                if re.search(pattern, text):
                    mentions.append({"poseId": pose_id, "t": start})
                    break
    return mentions


def _window_frames(frames: list[dict], t: float) -> list[dict]:
    """Labeled frames in the forward-leaning window around cue time t."""
    lo = t - MATCH_LOOKBACK_SEC
    hi = t + MATCH_LOOKAHEAD_SEC
    return [f for f in frames if f.get("pose") and lo <= f["t"] <= hi]


def _visual_pose_at(frames: list[dict], t: float) -> str | None:
    """The visual label at the frame nearest t, for context on a disagreement."""
    best: str | None = None
    best_gap = float("inf")
    for f in frames:
        if not f.get("pose"):
            continue
        gap = abs(f["t"] - t)
        if gap < best_gap:
            best_gap = gap
            best = f["pose"]
    return best


def _cross_check(mentions: list[dict], frames: list[dict]) -> list[dict]:
    """Pair each spoken cue with the visual track over its window.

    A cue agrees if the spoken pose shows up anywhere in the window, since the
    instructor reaches it a beat after naming it. `visual` records the label at
    the cue instant, so a disagreement still shows what was on screen then
    """
    checks: list[dict] = []
    for m in mentions:
        seen = {f["pose"] for f in _window_frames(frames, m["t"])}
        checks.append(
            {
                "t": m["t"],
                "spoken": m["poseId"],
                "visual": _visual_pose_at(frames, m["t"]),
                "agree": m["poseId"] in seen,
            }
        )
    return checks


def annotate_track(video_id: str, model: str = "small") -> dict:
    """Transcribe the video and fold the STT cross-check into its track.

    Returns a summary dict. The track JSON gains `spokenPoses` and `crossCheck`.
    """
    path = track_path(video_id)
    if not path.exists():
        raise SystemExit(f"no track for {video_id}, run process_video.py first")
    track = json.loads(path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        wav = _download_audio(video_id, tmp_dir)
        entries = _transcribe(wav, tmp_dir, model)

    mentions = _find_pose_mentions(entries)
    checks = _cross_check(mentions, track.get("frames", []))

    track["spokenPoses"] = mentions
    track["crossCheck"] = checks
    path.write_text(json.dumps(track), encoding="utf-8")

    agree = sum(1 for c in checks if c["agree"])
    return {
        "videoId": video_id,
        "mentions": len(mentions),
        "agree": agree,
        "disagree": len(checks) - agree,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/stt_crosscheck.py <video_id> [whisper_model]")
        sys.exit(1)
    model = sys.argv[2] if len(sys.argv) > 2 else "small"
    print(json.dumps(annotate_track(sys.argv[1], model), indent=2))
