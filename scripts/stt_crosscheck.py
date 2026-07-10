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
from track_common import track_path
from stt_common import POSE_ALIASES, download_audio, transcribe

# A cue is checked against the visual track over a forward-leaning window: the
# instructor names a pose, the class takes a while to settle into it, then holds
# it. Look back a little for a cue said mid-pose, look well ahead for the settle
# and hold, since the pose can arrive many seconds after its name
MATCH_LOOKBACK_SEC = 3.0
MATCH_LOOKAHEAD_SEC = 20.0


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
        wav = download_audio(video_id, tmp_dir)
        entries = transcribe(wav, tmp_dir, model)

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
