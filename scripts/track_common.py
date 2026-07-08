"""Shared helpers for the follow-along track scripts.

process_video.py builds a track from a video, stt_crosscheck.py annotates it with
narration cues. Both write into the frontend's public/tracks/ and address a
YouTube video by the same 11-character id.
"""

from __future__ import annotations

from pathlib import Path

# The frontend reads its static tracks from here. Both scripts write to it
FRONTEND_TRACKS = (
    Path(__file__).resolve().parent.parent.parent / "yoga-mirror" / "public" / "tracks"
)


def youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def track_path(video_id: str) -> Path:
    """The frontend track JSON for a video id."""
    return FRONTEND_TRACKS / f"{video_id}.json"
