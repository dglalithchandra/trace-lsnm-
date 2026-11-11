"""Download and verify the official, unaugmented Eye Disease Image Dataset archive."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_PAGE = "https://data.mendeley.com/datasets/s9bfhswzjb/1"
FILE_API = "https://data.mendeley.com/public-api/datasets/s9bfhswzjb/files?folder_id=root&version=1"
EXPECTED_NAME = "Original Dataset.zip"
EXPECTED_BYTES = 1_655_647_957
EXPECTED_SHA256 = "4cc006a9a769cb5172d552b69395d65fa1f1794af27baf32574a027051ea188d"
DESTINATION = ROOT / "data/external/EDID/Original_Dataset.zip"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Referer": DATASET_PAGE,
    "Accept": "application/json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    request = urllib.request.Request(FILE_API, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=40) as response:
        listing = json.load(response)
    matched = [row for row in listing if row.get("filename") == EXPECTED_NAME]
    if len(matched) != 1:
        raise RuntimeError(f"Expected exactly one {EXPECTED_NAME!r} file")
    item = matched[0]
    details = item["content_details"]
    if item["size"] != EXPECTED_BYTES or details["sha256_hash"] != EXPECTED_SHA256:
        raise RuntimeError("Official file metadata changed; review before downloading")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    part = DESTINATION.with_suffix(".zip.part")
    if DESTINATION.is_file():
        if DESTINATION.stat().st_size == EXPECTED_BYTES and sha256_file(DESTINATION) == EXPECTED_SHA256:
            print("Official original archive already verified", flush=True)
            return
        raise RuntimeError("Existing destination has unexpected size or hash")
    offset = part.stat().st_size if part.exists() else 0
    if offset > EXPECTED_BYTES:
        raise RuntimeError("Partial file is larger than official archive")
    headers = {key: value for key, value in HEADERS.items() if key != "Accept"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(details["download_url"], headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        if offset and response.status != 206:
            raise RuntimeError("Server did not honour resume range")
        mode = "ab" if offset else "wb"
        next_report = (offset // 100_000_000 + 1) * 100_000_000
        with part.open(mode) as stream:
            while block := response.read(8 * 1024 * 1024):
                stream.write(block)
                offset += len(block)
                if offset >= next_report:
                    print(f"Downloaded {offset:,}/{EXPECTED_BYTES:,} bytes", flush=True)
                    next_report += 100_000_000
    if part.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(f"Incomplete download: {part.stat().st_size} bytes")
    observed = sha256_file(part)
    if observed != EXPECTED_SHA256:
        raise RuntimeError(f"SHA-256 mismatch: {observed}")
    os.replace(part, DESTINATION)
    print(f"Verified {DESTINATION} SHA-256 {observed}", flush=True)


if __name__ == "__main__":
    main()
