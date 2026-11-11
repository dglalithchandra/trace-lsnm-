"""Freeze a label-hidden RFMiD-v1 training manifest for TRACE SSL."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/manifests/trace_analysis_manifest.csv"
ARCHIVE = ROOT / "A. RFMiD_All_Classes_Dataset.zip"
OUTPUT_DIR = ROOT / "data/protocols/trace_ssl_v1"
OUTPUT = OUTPUT_DIR / "unlabeled_train_manifest.csv"
PROTOCOL = OUTPUT_DIR / "protocol.json"


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8-sig") as stream:
        original = [row for row in csv.DictReader(stream) if row["dataset"] == "RFMiD" and row["final_split"] == "train"]
    if len(original) != 1917:
        raise RuntimeError(f"RFMiD training split changed: {len(original)}")
    entries = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        available = set(archive.namelist())
        for row in sorted(original, key=lambda item: item["record_id"]):
            member = row["archive_member"]
            if member not in available:
                raise RuntimeError(f"Training member absent from archive: {member}")
            entries.append({"record_id": row["record_id"], "archive_member": member, "image_sha256": hashlib.sha256(archive.read(member)).hexdigest()})
    if len({entry["record_id"] for entry in entries}) != len(entries):
        raise RuntimeError("Duplicate training record IDs")
    if len({entry["image_sha256"] for entry in entries}) != len(entries):
        raise RuntimeError("Byte-identical images inside SSL training pool")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("record_id", "archive_member", "image_sha256"))
        writer.writeheader()
        writer.writerows(entries)
    summary = {
        "version": "trace-ssl-v1",
        "status": "FROZEN_LABEL_HIDDEN_TRAINING_INPUT",
        "source_dataset": "RFMiD v1",
        "source_split": "training_only",
        "image_count": len(entries),
        "manifest_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "manifest_fields": ["record_id", "archive_member", "image_sha256"],
        "encoder_start": "verified_local_DINOv2-S/14_pretrained_weights",
        "image_preprocessing": "frozen_448px_geometry_and_quality;photometric_harmonisation_disabled",
        "primary_comparison": "same_RFMiD_v1_training_images_updates_backbone_and_head_budget;plain_contrastive_vs_lesion_preserving_views",
        "candidate_extension": "bounded_neighbour_downweighting_with_random_downweighting_control_after_base_comparison",
        "evaluation_policy": "select_on_existing_development_episodes_only;exclude_EDID_JSIEC_and_protected_RFMiD_test_from_SSL_training",
        "warning": "The manifest has no disease labels, but archive images can still contain visible pathology; no patient IDs exist for false-negative screening.",
        "protected_test_images_opened": 0,
    }
    PROTOCOL.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
