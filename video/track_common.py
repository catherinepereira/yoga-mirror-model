"""Shared helpers for the follow-along track scripts.

process_video.py builds a track from a video, stt_crosscheck.py annotates it with
narration cues. Both write into the frontend's public/tracks/ and address a
YouTube video by the same 11-character id.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# The frontend reads its static tracks from here. Both scripts write to it
FRONTEND_TRACKS = (
    Path(__file__).resolve().parents[2] / "yoga-mirror" / "public" / "tracks"
)

_VIDEO_ID = re.compile(r"^[\w-]{11}$")
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def resolve_video_id(url: str) -> str | None:
    """The 11-character YouTube id from a URL or bare id, or None if it isn't one."""
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


def track_path(video_id: str) -> Path:
    """The frontend track JSON for a video id."""
    return FRONTEND_TRACKS / f"{video_id}.json"
