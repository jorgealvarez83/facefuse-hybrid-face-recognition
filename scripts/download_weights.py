#!/usr/bin/env python3
"""Fetch the YOLO11-L face-detector weights from the GitHub release.

The checkpoint is ~49 MB, which is too large to keep in Git history, so it is
attached to the release instead.

    python scripts/download_weights.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "models" / "yolo11l-face.pt"
RELEASE_TAG = "v1.0.0"
ASSET_NAME = "yolo11l-face.pt"
URL = (
    "https://github.com/ahmedsayed1911/facefuse-hybrid-face-recognition"
    f"/releases/download/{RELEASE_TAG}/{ASSET_NAME}"
)
# md5 of the published checkpoint; verified after download.
EXPECTED_MD5 = "cfb1d0b21aeed932e2fe2d97506be875"


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def report(done: int, total: int) -> None:
    if total > 0:
        pct = min(100.0, done * 100.0 / total)
        sys.stdout.write(f"\r  {pct:5.1f}%  ({done / 1e6:.1f} / {total / 1e6:.1f} MB)")
    else:
        sys.stdout.write(f"\r  {done / 1e6:.1f} MB")
    sys.stdout.flush()


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            report(done, total)
    print()
    tmp.replace(dest)
    return dest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--url", default=URL)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file already exists")
    args = parser.parse_args(argv)

    if args.dest.exists() and not args.force:
        print(f"Already present: {args.dest}")
        return 0

    download(args.url, args.dest)

    actual = md5sum(args.dest)
    if actual != EXPECTED_MD5:
        print(f"Checksum mismatch: expected {EXPECTED_MD5}, got {actual}", file=sys.stderr)
        return 1

    size_mb = args.dest.stat().st_size / 1e6
    print(f"Saved {args.dest} ({size_mb:.1f} MB, md5 verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
