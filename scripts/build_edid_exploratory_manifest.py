"""Exclude EDID exact cross-folder copies and retain one image per unambiguous hash."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/manifests/edid_original_records.csv"
OUTPUT = ROOT / "data/manifests/edid_exploratory_manifest.csv"
SUMMARY = ROOT / "data/manifests/edid_exploratory_manifest_summary.json"


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        groups[row["sha256"]].append(row)
    output = []
    for digest, group in groups.items():
        group.sort(key=lambda row: row["archive_member"])
        classes = sorted({row["class_folder"] for row in group})
        for index, row in enumerate(group):
            reason = (
                "cross_class_exact_copy" if len(classes) > 1 else
                "non_fundus_pterygium" if row["class_folder"] == "Pterygium" else
                "duplicate_copy_within_class" if index > 0 else
                "eligible_exploratory"
            )
            output.append({
                **row,
                "exact_group_class_folders": "|".join(classes),
                "exact_group_size": len(group),
                "status": reason,
                "analysis_eligible": reason == "eligible_exploratory",
            })
    output.sort(key=lambda row: row["archive_member"])
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    eligible = [row for row in output if row["analysis_eligible"]]
    counts = Counter(row["class_folder"] for row in eligible)
    status_counts = Counter(row["status"] for row in output)
    summary = {
        "status": "EXPLORATORY_MANIFEST_READY",
        "source_records": len(records),
        "exact_unique_image_groups": len(groups),
        "eligible_fundus_images": len(eligible),
        "eligible_counts_by_class": dict(sorted(counts.items())),
        "excluded_counts_by_reason": {
            key: value for key, value in sorted(status_counts.items())
            if key != "eligible_exploratory"
        },
        "labels_are_single_folder_assignments": True,
        "patient_ids_verified": False,
        "per_image_camera_ids_verified": False,
        "near_duplicate_screen_complete": False,
        "rfmid_cross_dataset_overlap_screen_complete": False,
        "model_evaluation_ready": False,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
