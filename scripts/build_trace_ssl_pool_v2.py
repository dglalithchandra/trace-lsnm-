"""Build the core-only TRACE SSL v2 admission manifest without disease labels."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/protocols/trace_ssl_v1/cached_image_index.csv"
ANALYSIS = ROOT / "data/manifests/trace_analysis_manifest.csv"
OUTPUT = ROOT / "data/manifests/trace_ssl_pool_v2.csv"
SUMMARY = ROOT / "data/protocols/trace_source_balance_v1/pool_summary.json"

FIELDS = (
    "record_id",
    "dataset_id",
    "source_id",
    "source_split",
    "final_split",
    "cache_path",
    "source_image_sha256",
    "cache_image_sha256",
    "preprocessing_status",
    "patient_id",
    "patient_id_status",
    "eye_id",
    "split_group_id",
    "camera_model",
    "camera_domain",
    "ssl_v2_role",
    "admission_status",
    "admission_reason",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    cache_rows = read_csv(INDEX)
    if len(cache_rows) != 1917:
        raise RuntimeError(f"Expected 1917 frozen core rows, found {len(cache_rows)}")
    if len({row["record_id"] for row in cache_rows}) != len(cache_rows):
        raise RuntimeError("Duplicate record IDs in the frozen SSL cache index")
    if len({row["source_image_sha256"] for row in cache_rows}) != len(cache_rows):
        raise RuntimeError("Duplicate source-image hashes in the frozen SSL cache index")

    analysis_rows = read_csv(ANALYSIS)
    analysis = {row["record_id"]: row for row in analysis_rows}
    if len(analysis) != len(analysis_rows):
        raise RuntimeError("Duplicate record IDs in the analysis manifest")

    output_rows: list[dict[str, str]] = []
    for cached in sorted(cache_rows, key=lambda row: row["record_id"]):
        record_id = cached["record_id"]
        if not record_id.startswith("rfmid:training:"):
            raise RuntimeError(f"Non-training record in the frozen SSL index: {record_id}")
        metadata = analysis.get(record_id)
        if metadata is None:
            raise RuntimeError(f"Record missing from analysis manifest: {record_id}")
        if metadata["dataset"] != "RFMiD" or metadata["source_split"] != "training":
            raise RuntimeError(f"Unexpected source role for {record_id}")
        if metadata["final_split"] != "train":
            raise RuntimeError(f"Development/test member cannot enter SSL v2: {record_id}")
        if metadata["patient_id"]:
            patient_id = metadata["patient_id"]
        else:
            patient_id = ""
        output_rows.append(
            {
                "record_id": record_id,
                "dataset_id": "rfmid_v1",
                "source_id": "rfmid_v1_train",
                "source_split": "training",
                "final_split": "train",
                "cache_path": cached["cache_path"],
                "source_image_sha256": cached["source_image_sha256"],
                "cache_image_sha256": cached["cache_image_sha256"],
                "preprocessing_status": cached["preprocessing_status"],
                "patient_id": patient_id,
                "patient_id_status": metadata["patient_id_status"],
                "eye_id": "",
                "split_group_id": metadata["split_group_id"],
                "camera_model": metadata["camera_model"],
                "camera_domain": metadata["camera_domain"],
                "ssl_v2_role": "core_training",
                "admission_status": "admitted",
                "admission_reason": "frozen_rfmid_v1_training_member;not_development_or_test",
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    status_counts: dict[str, int] = {}
    camera_domain_counts: dict[str, int] = {}
    for row in output_rows:
        status_counts[row["preprocessing_status"]] = status_counts.get(row["preprocessing_status"], 0) + 1
        camera_domain_counts[row["camera_domain"]] = camera_domain_counts.get(row["camera_domain"], 0) + 1
    summary = {
        "version": "trace-ssl-pool-v2-core",
        "status": "CORE_ONLY_EXPANSION_BLOCKED",
        "record_count": len(output_rows),
        "dataset_counts": {"rfmid_v1": len(output_rows)},
        "source_counts": {"rfmid_v1_train": len(output_rows)},
        "preprocessing_status_counts": status_counts,
        "camera_domain_counts_for_audit_only": camera_domain_counts,
        "patient_ids_available": sum(bool(row["patient_id"]) for row in output_rows),
        "eye_ids_available": 0,
        "disease_label_fields_present": False,
        "input_cache_index_sha256": sha256(INDEX),
        "input_analysis_manifest_sha256": sha256(ANALYSIS),
        "output_manifest_sha256": sha256(OUTPUT),
        "protected_test_records_admitted": 0,
        "development_query_records_admitted": 0,
        "warning": "RFMiD provides no patient IDs; split-group controls cannot prove patient independence.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
