"""Mine instructor narration for poses not yet in the dataset.

The classifier only knows the poses with a folder in data/yoga-poses/. To grow that set,
this transcribes a batch of instructor videos and reports pose names the
narration uses that the dataset does not cover, so you know what to collect next.

Two passes over each transcript:

  1. Phrase pass. Pull every "<word(s)> pose" / "<word> posture" / "warrior <n>"
     phrase, drop the ones that name a dataset pose, and rank the rest by how
     often they come up across the batch. High precision, these are named poses.

  2. Rare-word pass. Count every spoken word, drop stopwords and words the
     dataset already names, and list the least-frequent remainder. Lower
     precision, but it surfaces poses cued without the word "pose" (savasana,
     plank, malasana) that the phrase pass misses. For review, not automatic.

Run from the repo root:
    python -m video.mine_poses urls.txt [--model small] [--rare-words 40]

urls.txt is one YouTube URL or id per line. Blank lines and #-comments skipped.
Writes exports/candidate_poses.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from video.track_common import resolve_video_id
from video.stt_common import POSE_ALIASES, download_audio, transcribe

OUT_PATH = ROOT / "exports" / "candidate_poses.json"
TRANSCRIPT_DIR = ROOT / "exports" / "transcripts"

# A pose cue: up to two name words, then "pose"/"posture". Leading filler the
# grab picks up ("in tree", "to half moon") is trimmed after. "warrior <n>" is
# caught separately since it names a pose without the suffix
_POSE_PHRASE = re.compile(r"\b((?:\w+\s+){0,2}\w+)\s+(?:pose|posture)\b")
_WARRIOR = re.compile(r"\bwarrior\s+(one|two|three|1|2|3|i{1,3})\b")

# The words that make up the dataset pose names and their spoken aliases, so the
# phrase pass can tell a known pose from a candidate, and the rare-word pass can
# drop words the dataset already covers
_KNOWN_WORDS = {
    w
    for pose_id, aliases in POSE_ALIASES.items()
    for phrase in [pose_id.replace("-", " "), *aliases]
    for w in phrase.split()
}

# Common English and yoga-class filler that would otherwise dominate the
# rare-word tail's opposite end and clutter review. Not exhaustive, just the
# words a narrator says constantly that are never a pose
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for",
    "with", "your", "you", "we", "i", "it", "is", "are", "be", "here", "there",
    "this", "that", "these", "those", "so", "just", "now", "then", "up", "down",
    "left", "right", "side", "back", "front", "over", "into", "out", "let",
    "lets", "gonna", "going", "take", "bring", "keep", "hold", "breath",
    "breathe", "inhale", "exhale", "hand", "hands", "foot", "feet", "leg",
    "legs", "arm", "arms", "knee", "knees", "hip", "hips", "one", "two", "okay",
    "yeah", "little", "bit", "really", "nice", "good", "great", "feel", "feeling",
    # cueing verbs that lead a phrase grab ("begin in tree", "move to warrior")
    "begin", "move", "step", "come", "rise", "lower", "flow", "lift", "press",
    "reach", "fold", "extend", "sink", "settle", "transition", "enter",
}

# Pull real words only, drop digits and punctuation the transcript carries
_WORD = re.compile(r"[a-z']+")


def _read_urls(path: Path) -> list[str]:
    """Video ids from a file of URLs/ids, skipping blanks and #-comments."""
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vid = resolve_video_id(line)
        if vid is None:
            print(f"warning: skipping unrecognized line: {line}", file=sys.stderr)
            continue
        ids.append(vid)
    return ids


def _transcript_text(entries: list[dict]) -> str:
    return " ".join(e.get("text", "") for e in entries).lower()


def transcribe_video(video_id: str, model: str = "small") -> str:
    """Transcribe one video to lowercased text, caching to exports/transcripts/.

    A cached transcript is reused, so re-mining or re-checking a video does not
    re-download and re-run Whisper
    """
    cache = TRANSCRIPT_DIR / f"{video_id}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["text"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        wav = download_audio(video_id, tmp_dir)
        entries = transcribe(wav, tmp_dir, model)
    text = _transcript_text(entries)

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"text": text}), encoding="utf-8")
    return text


def spoken_candidates(text: str) -> dict[str, int]:
    """Pose names spoken in a transcript that the dataset does not cover."""
    return dict(_phrase_candidates(text))


def _phrase_candidates(text: str) -> Counter:
    """Named-pose phrases in one transcript that are not a dataset pose."""
    found: Counter = Counter()

    for m in _POSE_PHRASE.finditer(text):
        words = m.group(1).split()
        # The grab pulls up to two leading words, which are usually filler ("in
        # tree", "to half moon"). Drop leading stopwords so the phrase is just
        # the pose name
        while words and words[0] in _STOPWORDS:
            words.pop(0)
        if not words:
            continue
        # Known only if every remaining word names a dataset pose, so "triangle"
        # is skipped but "half moon" (unknown "moon") is kept
        if all(w in _KNOWN_WORDS for w in words):
            continue
        found[f"{' '.join(words)} pose"] += 1

    for m in _WARRIOR.finditer(text):
        n = m.group(1)
        if n in ("three", "3", "iii"):
            found["warrior three"] += 1

    return found


def _content_words(text: str) -> list[str]:
    """Words worth counting: long enough, not filler, not an existing pose name."""
    return [
        w
        for w in _WORD.findall(text)
        if len(w) > 3 and w not in _STOPWORDS and w not in _KNOWN_WORDS
    ]


def mine(url_file: Path, model: str, rare_word_limit: int) -> dict:
    """Transcribe every video in the file and report unknown pose candidates."""
    video_ids = _read_urls(url_file)
    if not video_ids:
        raise SystemExit(f"no usable video ids in {url_file}")

    phrase_counts: Counter = Counter()
    phrase_videos: dict[str, set[str]] = {}
    word_counts: Counter = Counter()

    for i, vid in enumerate(video_ids, 1):
        print(f"[{i}/{len(video_ids)}] {vid}: transcribing...")
        try:
            text = transcribe_video(vid, model)
        except Exception as e:
            print(f"  skipped {vid}: {e}", file=sys.stderr)
            continue

        for phrase, n in _phrase_candidates(text).items():
            phrase_counts[phrase] += n
            phrase_videos.setdefault(phrase, set()).add(vid)
        word_counts.update(_content_words(text))

    candidates = [
        {"phrase": phrase, "count": n, "videos": sorted(phrase_videos[phrase])}
        for phrase, n in phrase_counts.most_common()
    ]
    # Ascending count then alphabetical, so the rarest words surface first
    rare = sorted(word_counts.items(), key=lambda kv: (kv[1], kv[0]))[:rare_word_limit]

    report = {
        "videos": video_ids,
        "candidates": candidates,
        "rareWords": [{"word": w, "count": n} for w, n in rare],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _print_summary(report: dict) -> None:
    print(f"\nTranscribed {len(report['videos'])} videos.\n")

    print("Candidate poses (named as \"X pose\", not in the dataset):")
    if report["candidates"]:
        for c in report["candidates"]:
            print(f"  {c['count']:3d}  {c['phrase']}  ({len(c['videos'])} videos)")
    else:
        print("  none")

    print("\nRare words to review (poses cued without \"pose\" hide here):")
    print("  " + ", ".join(f"{r['word']}({r['count']})" for r in report["rareWords"]))
    print(f"\nWrote {OUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine narration for poses not in the dataset")
    parser.add_argument("url_file", type=Path, help="File of YouTube URLs/ids, one per line")
    parser.add_argument("--model", default="small", help="Whisper model size")
    parser.add_argument(
        "--rare-words", type=int, default=40, dest="rare_words",
        help="How many rare words to list for review",
    )
    args = parser.parse_args()
    report = mine(args.url_file, args.model, args.rare_words)
    _print_summary(report)


if __name__ == "__main__":
    main()
