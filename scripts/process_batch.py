"""Process a list of YouTube videos into follow-along tracks.

Reads a file of YouTube URLs/ids (one per line, blanks and #-comments skipped)
and runs each through process_video. For every video it reports the poses the
classifier detected, the low-confidence visual guesses, and the pose names the
narration speaks that the dataset does not cover. The skeleton and transcript are
cached under exports/ so a video need not be re-downloaded to re-run either step.

Run from the repo root:
    python scripts/process_batch.py videos.txt [--model small]

A failed video is reported and skipped, the rest keep going.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from process_video import process, print_poses
from track_common import resolve_video_id
from mine_poses import transcribe_video, spoken_candidates


def _read_urls(path: Path) -> list[str]:
    """YouTube URLs from the list, skipping blanks, comments, and non-YouTube lines."""
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if resolve_video_id(line) is None:
            print(f"warning: skipping non-YouTube line: {line}", file=sys.stderr)
            continue
        urls.append(line)
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a list of YouTube videos into tracks")
    parser.add_argument("list_file", type=Path, help="File of YouTube URLs/ids, one per line")
    parser.add_argument("--model", default="small", help="Whisper model size")
    args = parser.parse_args()

    urls = _read_urls(args.list_file)
    if not urls:
        raise SystemExit(f"no YouTube URLs in {args.list_file}")

    summaries, failed = [], []
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] {url}")
        try:
            summary = process(url)
        except Exception as e:
            print(f"  failed: {e}", file=sys.stderr)
            failed.append(url)
            continue

        try:
            summary["spoken"] = spoken_candidates(transcribe_video(summary["videoId"], args.model))
        except Exception as e:
            print(f"  transcription failed: {e}", file=sys.stderr)
            summary["spoken"] = {}
        summaries.append(summary)

    print(f"\n{'=' * 40}\nProcessed {len(summaries)} tracks, {len(failed)} failed")
    for summary in summaries:
        print_poses(summary)
    for url in failed:
        print(f"  failed: {url}")


if __name__ == "__main__":
    main()
