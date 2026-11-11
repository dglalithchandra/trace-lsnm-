"""Audit frozen geometry/QC acceptance on the EDID exploratory query set."""

from __future__ import annotations

import csv
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

from trace_preprocessing import PreprocessingConfig, TracePreprocessor

MANIFEST = ROOT / "data/manifests/edid_exploratory_manifest_v2.csv"
ARCHIVE = ROOT / "data/external/EDID/Original_Dataset.zip"
OUTPUT = ROOT / "data/protocols/trace_bench_v2/edid_preprocessing_audit.csv"
SUMMARY = ROOT / "data/protocols/trace_bench_v2/edid_preprocessing_audit_summary.json"
CLASSES = {"Macular Scar", "Retinitis Pigmentosa", "Disc Edema", "Healthy"}


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        query = [row for row in csv.DictReader(stream) if row["analysis_eligible"] == "True" and row["class_folder"] in CLASSES]
    if len(query) != 1333:
        raise RuntimeError(f"Frozen query count changed: {len(query)}")
    pipeline = TracePreprocessor(PreprocessingConfig(apply_photometric_correction=False))
    audit = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for number, row in enumerate(query, 1):
            member = row["archive_member"]
            bgr = cv2.imdecode(np.frombuffer(archive.read(member), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                status, reasons, warnings = "decode_failure", ["jpeg_decode_failure"], []
            else:
                result = pipeline.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), image_id=f"edid:{member}")
                status, reasons, warnings = str(result.status), result.reasons, result.warnings
            audit.append({"record_id": f"edid:{member}", "archive_member": member, "class_folder": row["class_folder"], "preprocessing_status": status, "reasons_json": json.dumps(reasons), "warnings_json": json.dumps(warnings)})
            if number % 100 == 0 or number == len(query):
                print(f"EDID preprocessing audit {number}/{len(query)}", flush=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    summary = {"status": "PREPROCESSING_AUDIT_BEFORE_MODEL_SCORES", "query_count": len(audit), "by_status": dict(Counter(row["preprocessing_status"] for row in audit)), "by_class_status": {name: dict(Counter(row["preprocessing_status"] for row in audit if row["class_folder"] == name)) for name in sorted(CLASSES)}, "preprocessing": pipeline.config.to_dict(), "protected_test_images_opened": 0}
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
