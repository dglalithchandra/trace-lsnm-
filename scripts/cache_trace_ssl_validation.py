"""Cache RFMiD validation geometry once for repeated development extraction."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.loader import ManifestImageLoader

CACHE = ROOT / "data/processed/trace_ssl_validation_v1"
INDEX = ROOT / "data/protocols/trace_ssl_v1/cached_validation_index.csv"
SUMMARY = ROOT / "data/protocols/trace_ssl_v1/validation_cache_summary.json"


def main() -> None:
    pipeline = TracePreprocessor(PreprocessingConfig(apply_photometric_correction=False))
    loader = ManifestImageLoader(ROOT, allow_test=False)
    CACHE.mkdir(parents=True, exist_ok=True)
    entries = []
    try:
        ids = loader.record_ids(split="validation", dataset="RFMiD")
        if len(ids) != 639:
            raise RuntimeError("RFMiD validation count changed")
        for number, record_id in enumerate(ids, 1):
            target = CACHE / f"{hashlib.sha256(record_id.encode()).hexdigest()[:24]}.png"
            if not target.exists():
                image, row = loader.load_image(record_id)
                if row["_effective_split"] != "validation":
                    raise RuntimeError(f"Split mismatch: {record_id}")
                result = pipeline.process(image, image_id=record_id)
                if result.status == ProcessingStatus.REJECT or result.processed_image is None:
                    raise RuntimeError(f"Preprocessing rejected validation image: {record_id}")
                temp = target.with_suffix(".tmp.png")
                if not cv2.imwrite(str(temp), cv2.cvtColor(result.processed_image, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"Cache write failed: {target}")
                temp.replace(target)
            bgr = cv2.imread(str(target), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape != (448, 448, 3):
                raise RuntimeError(f"Invalid validation cache: {target}")
            entries.append({"record_id": record_id, "cache_path": str(target.relative_to(ROOT)).replace("\\", "/"), "cache_image_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
            if number % 100 == 0 or number == len(ids):
                print(f"Validation cache {number}/{len(ids)}", flush=True)
    finally:
        loader.close()
    with INDEX.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    summary = {"status": "DEVELOPMENT_VALIDATION_CACHE_ONLY", "images": len(entries), "cache_bytes": sum((ROOT / item["cache_path"]).stat().st_size for item in entries), "index_sha256": hashlib.sha256(INDEX.read_bytes()).hexdigest(), "preprocessing": pipeline.config.to_dict(), "test_records_accessed": 0}
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
