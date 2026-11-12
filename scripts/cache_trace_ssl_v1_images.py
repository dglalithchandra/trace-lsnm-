"""Cache label-hidden RFMiD training images after frozen TRACE geometry/QC."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor

MANIFEST = ROOT / "data/protocols/trace_ssl_v1/unlabeled_train_manifest.csv"
ARCHIVE = ROOT / "A. RFMiD_All_Classes_Dataset.zip"
CACHE = ROOT / "data/processed/trace_ssl_v1"
INDEX = ROOT / "data/protocols/trace_ssl_v1/cached_image_index.csv"
SUMMARY = ROOT / "data/protocols/trace_ssl_v1/cache_summary.json"


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1917 or set(rows[0]) != {"record_id", "archive_member", "image_sha256"}:
        raise RuntimeError("Label-hidden SSL input changed")
    pipeline = TracePreprocessor(PreprocessingConfig(apply_photometric_correction=False))
    CACHE.mkdir(parents=True, exist_ok=True)
    index = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for number, row in enumerate(rows, 1):
            stem = hashlib.sha256(row["record_id"].encode()).hexdigest()[:24]
            target = CACHE / f"{stem}.png"
            if target.exists():
                rgb = cv2.cvtColor(cv2.imread(str(target), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                if rgb.shape != (448, 448, 3):
                    raise RuntimeError(f"Malformed existing cache: {target}")
                status = "cached_existing"
            else:
                raw = archive.read(row["archive_member"])
                if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
                    raise RuntimeError(f"Source image hash changed: {row['record_id']}")
                bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"Source image decode failed: {row['record_id']}")
                result = pipeline.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), image_id=row["record_id"])
                if result.status == ProcessingStatus.REJECT or result.processed_image is None:
                    raise RuntimeError(f"Frozen preprocessing rejected training image: {row['record_id']}")
                rgb = result.processed_image
                status = str(result.status)
                temp = target.with_suffix(".tmp.png")
                if not cv2.imwrite(str(temp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"Cannot write cache image: {temp}")
                temp.replace(target)
            index.append({"record_id": row["record_id"], "cache_path": str(target.relative_to(ROOT)).replace("\\", "/"), "source_image_sha256": row["image_sha256"], "cache_image_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "preprocessing_status": status})
            if number % 100 == 0 or number == len(rows):
                print(f"SSL cache {number}/{len(rows)}", flush=True)
    if len({item["cache_path"] for item in index}) != len(index):
        raise RuntimeError("Cache filename collision")
    with INDEX.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    summary = {"status": "FROZEN_PREPROCESSED_LABEL_HIDDEN_CACHE", "images": len(index), "by_preprocessing_status": dict(Counter(item["preprocessing_status"] for item in index)), "cache_bytes": sum((ROOT / item["cache_path"]).stat().st_size for item in index), "cache_index_sha256": hashlib.sha256(INDEX.read_bytes()).hexdigest(), "preprocessing": pipeline.config.to_dict(), "protected_test_images_opened": 0}
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
