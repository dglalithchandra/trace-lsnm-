"""Inventory JSIEC-1000 classes and exact-image reuse without model access."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/external/JSIEC/1000images.zip"
RFMID_ARCHIVE = ROOT / "A. RFMiD_All_Classes_Dataset.zip"
RFMID_MANIFEST = ROOT / "data/manifests/trace_analysis_manifest.csv"
EDID_MANIFEST = ROOT / "data/manifests/edid_exploratory_manifest_v2.csv"
OUT = ROOT / "data/manifests/jsiec_exploratory_manifest.csv"
SUMMARY = ROOT / "data/manifests/jsiec_exploratory_summary.json"


def main() -> None:
    records = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for member in archive.namelist():
            if not member.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                continue
            parts = PurePosixPath(member).parts
            if len(parts) != 3 or parts[0] != "1000images":
                raise RuntimeError(f"Unexpected archive path: {member}")
            raw = archive.read(member)
            records.append({"archive_member": member, "class_folder": parts[1], "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    if len(records) != 1000:
        raise RuntimeError(f"Expected 1000 JSIEC images, found {len(records)}")
    grouped = defaultdict(list)
    for row in records:
        grouped[row["sha256"]].append(row)
    for group in grouped.values():
        categories = sorted({row["class_folder"] for row in group})
        for row in group:
            row["exact_group_size"] = len(group)
            row["exact_group_classes"] = "|".join(categories)
            if len(categories) > 1:
                row["status"] = "cross_class_exact_conflict"
                row["analysis_eligible"] = "False"
            elif row != min(group, key=lambda item: item["archive_member"]):
                row["status"] = "same_class_exact_extra_copy"
                row["analysis_eligible"] = "False"
            else:
                row["status"] = "eligible_after_exact_screen"
                row["analysis_eligible"] = "True"
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    eligible = [row for row in records if row["analysis_eligible"] == "True"]
    counts = Counter(row["class_folder"] for row in eligible)
    focus = {row["sha256"] for row in eligible if row["class_folder"] in {"15.0.Retinitis pigmentosa", "15.1.Bietti crystalline dystrophy", "0.0.Normal"}}
    with EDID_MANIFEST.open(newline="", encoding="utf-8") as stream:
        edid_hashes = {row["sha256"] for row in csv.DictReader(stream)}
    with RFMID_MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        train_members = [row["archive_member"] for row in csv.DictReader(stream) if row["dataset"] == "RFMiD" and row["final_split"] == "train"]
    if len(train_members) != 1917:
        raise RuntimeError("RFMiD v1 support pool changed")
    rfmid_hashes = set()
    with zipfile.ZipFile(RFMID_ARCHIVE) as archive:
        for member in train_members:
            rfmid_hashes.add(hashlib.sha256(archive.read(member)).hexdigest())
    summary = {"status": "EXACT_IMAGE_AUDIT_ONLY", "archive_images": len(records), "eligible_after_exact_screen": len(eligible), "exact_duplicate_groups": sum(len(group) > 1 for group in grouped.values()), "cross_class_exact_groups": sum(len({row['class_folder'] for row in group}) > 1 for group in grouped.values()), "target_counts": {"RP": counts["15.0.Retinitis pigmentosa"], "Bietti": counts["15.1.Bietti crystalline dystrophy"], "Normal": counts["0.0.Normal"]}, "focus_exact_overlap_edid": len(focus & edid_hashes), "focus_exact_overlap_rfmid_v1_training": len(focus & rfmid_hashes), "patient_ids_available": False, "per_image_camera_ids_available": False, "warning": "Exact byte matching does not detect re-encoded or cropped copies", "protected_test_images_opened": 0}
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
