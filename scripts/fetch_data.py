"""Keyless image fetch for the pose dataset.

prompt2dataset's `p2d add` resolves a free-text description into subjects via
Claude, which needs an API key. The pose list is already known, so this skips the
resolver and calls p2d's image sources directly, which need no key.

Images land in data/raw/<pose-id>/, one folder per pose. Re-running tops up
toward the target and is idempotent (files are named by content hash).

Run from the repo root:
    python scripts/fetch_data.py

Requires the local prompt2dataset checkout on PYTHONPATH, this script adds it.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "raw"

# Local prompt2dataset checkout. Defaults to a sibling of the Developer tree, set
# P2D_SRC to point elsewhere
P2D_SRC = Path(
    os.environ.get("P2D_SRC", ROOT.parent / "PythonPackages" / "prompt2dataset" / "src")
).resolve()
if not P2D_SRC.exists():
    raise SystemExit(
        f"prompt2dataset not found at {P2D_SRC}. Set P2D_SRC to its src directory."
    )

sys.path.insert(0, str(P2D_SRC))
from prompt2dataset.sources import fetch_all  # noqa: E402
from prompt2dataset.models import MediaType  # noqa: E402

# Sources that need no API key. iNaturalist is omitted because it indexes
# wildlife, not people
SOURCES = ["bing", "duckduckgo", "wikimedia_commons", "openverse"]
PER_SOURCE = 60

# pose id -> search query variants. Several phrasings per pose surface different
# images, since each source returns the same top results for a given query
POSES: dict[str, list[str]] = {
    "mountain": ["Tadasana mountain pose yoga", "mountain pose yoga woman"],
    "warrior-i": ["Virabhadrasana I yoga pose", "warrior 1 pose yoga"],
    "warrior-ii": ["Virabhadrasana II yoga pose", "warrior 2 pose yoga"],
    "warrior-iii": ["Virabhadrasana III yoga pose", "warrior 3 pose yoga balance"],
    "triangle": ["Trikonasana triangle yoga pose", "extended triangle pose yoga"],
    "tree": ["Vrksasana tree yoga pose", "tree pose yoga balance"],
    "chair": ["Utkatasana chair yoga pose", "chair pose yoga"],
    "standing-forward-fold": [
        "Uttanasana standing forward fold yoga",
        "standing forward bend yoga pose",
    ],
    "half-moon": ["Ardha Chandrasana half moon yoga", "half moon pose yoga balance"],
    "eagle": ["Garudasana eagle yoga pose", "eagle pose yoga"],
    "easy-pose": ["Sukhasana easy seated yoga pose", "cross legged seated yoga meditation"],
    "seated-forward-fold": [
        "Paschimottanasana seated forward fold yoga",
        "seated forward bend yoga pose",
    ],
    "bound-angle": ["Baddha Konasana bound angle yoga", "butterfly pose yoga seated"],
    "head-to-knee": ["Janu Sirsasana head to knee yoga", "head to knee forward bend yoga"],
    "boat": ["Navasana boat yoga pose", "boat pose yoga core"],
    "cobra": ["Bhujangasana cobra yoga pose", "cobra pose yoga backbend"],
    "upward-dog": [
        "Urdhva Mukha Svanasana upward facing dog yoga",
        "upward facing dog pose yoga",
    ],
    "bridge": ["Setu Bandha bridge yoga pose", "bridge pose yoga backbend"],
    "camel": ["Ustrasana camel yoga pose", "camel pose yoga backbend kneeling"],
    "bow": ["Dhanurasana bow yoga pose", "bow pose yoga backbend prone"],
    "wheel": ["Urdhva Dhanurasana wheel yoga pose", "wheel pose yoga backbend"],
    "downward-dog": [
        "Adho Mukha Svanasana downward facing dog yoga",
        "downward dog pose yoga",
        "downward facing dog stretch",
    ],
    "plank": ["plank pose yoga", "high plank yoga pose"],
    "side-plank": ["Vasisthasana side plank yoga", "side plank pose yoga"],
    "crow": ["Bakasana crow yoga pose", "crow pose yoga arm balance"],
    "headstand": ["Sirsasana headstand yoga pose", "supported headstand yoga"],
    "shoulder-stand": ["Sarvangasana shoulder stand yoga", "supported shoulder stand yoga"],
    "childs-pose": ["Balasana child's pose yoga", "child pose yoga resting"],
    "corpse": ["Savasana corpse pose yoga", "savasana relaxation yoga lying down"],
    "cat-cow": ["Marjaryasana cat cow yoga pose", "cat pose yoga tabletop"],
    "pigeon": ["Eka Pada Rajakapotasana pigeon yoga", "pigeon pose yoga hip opener"],
    "happy-baby": ["Ananda Balasana happy baby yoga", "happy baby pose yoga"],
}

EXT_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


async def download(
    client: httpx.AsyncClient, url: str, pose_id: str, folder: Path
) -> bool:
    try:
        resp = await client.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError:
        return False

    ext = EXT_BY_MIME.get(resp.headers.get("content-type", "").split(";")[0])
    if ext is None:
        return False

    # Name by content hash so a re-downloaded URL maps to the same file and never
    # duplicates, which keeps top-up runs idempotent
    digest = hashlib.sha1(resp.content).hexdigest()[:12]
    dest = folder / f"{pose_id}_{digest}{ext}"
    if dest.exists():
        return False
    dest.write_bytes(resp.content)
    return True


# Target images per pose. A pose already at this count is skipped, so the script
# is safe to re-run after an interruption and tops up toward the target
ENOUGH = 50


async def save_pose(
    client: httpx.AsyncClient,
    pose_id: str,
    by_source: dict[str, list[dict]],
) -> None:
    folder = DATASET / pose_id
    folder.mkdir(parents=True, exist_ok=True)

    urls: list[str] = []
    seen: set[str] = set()
    for items in by_source.values():
        for item in items:
            url = item["url"]
            if url not in seen:
                seen.add(url)
                urls.append(url)

    have = len(list(folder.glob("*")))
    # Fetch a margin beyond the target since some downloads fail or get rejected
    # later by the extractor. The extra is harmless
    want = max(0, int(ENOUGH * 1.4) - have)
    results = await asyncio.gather(
        *(download(client, url, pose_id, folder) for url in urls[: want * 2])
    )
    print(f"  {pose_id}: {have} + {sum(results)} new = {have + sum(results)}")


# Poses fetched per batch. Small batches keep Bing from rate-limiting the burst
# of concurrent searches, which silently returns empty for most poses
BATCH = 4

# A browser User-Agent, many image hosts reject hotlinks from unknown agents
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


async def main() -> None:
    pending = {
        pose_id: queries
        for pose_id, queries in POSES.items()
        if len(list((DATASET / pose_id).glob("*"))) < ENOUGH
    }
    if not pending:
        print("All poses already have enough images.")
        return

    print(f"Fetching {len(pending)} poses from {SOURCES} in batches of {BATCH}...")
    ids = list(pending)

    async with httpx.AsyncClient(headers={"User-Agent": UA}) as client:
        for start in range(0, len(ids), BATCH):
            batch = ids[start : start + BATCH]
            subjects = [q for pid in batch for q in pending[pid]]
            query_to_id = {q: pid for pid in batch for q in pending[pid]}
            fetched = await fetch_all(subjects, SOURCES, PER_SOURCE, MediaType.image)

            by_pose: dict[str, dict[str, list[dict]]] = {}
            for query, by_source in fetched.items():
                merged = by_pose.setdefault(query_to_id[query], {})
                for source, items in by_source.items():
                    merged.setdefault(source, []).extend(items)

            await asyncio.gather(
                *(save_pose(client, pid, by_pose.get(pid, {})) for pid in batch)
            )

    print(f"\nDone. Images in {DATASET}")


if __name__ == "__main__":
    asyncio.run(main())
