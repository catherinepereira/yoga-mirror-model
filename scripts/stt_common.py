"""Shared narration transcription for the STT scripts.

stt_crosscheck.py matches spoken cues against the visual track, mine_poses.py
scans narration for pose names not in the dataset. Both download a video's audio
and run it through dinscribe with the same vocab-primed Whisper config.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from track_common import youtube_url

# Pose names fed to Whisper as an initial prompt, so it decodes the yoga terms
# it otherwise mangles
VOCAB_FILE = Path(__file__).parent / "pose_vocab.txt"

# dinscribe prints box-drawing characters the default Windows console codec
# (cp1252) cannot encode. Force UTF-8 so its progress output does not crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

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


def download_audio(video_id: str, dest_dir: Path) -> Path:
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


def transcribe(wav: Path, work_dir: Path, model: str) -> list[dict]:
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
