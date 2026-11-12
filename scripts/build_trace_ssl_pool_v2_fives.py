"""Build the leakage-audited RFMiD+FIVES training pool for A2/A3."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_trace_ssl_pool_v2 import FIELDS, read_csv, sha256
from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.loader import ManifestImageLoader

CORE = ROOT / "data/manifests/trace_ssl_pool_v2.csv"
FIVES = ROOT / "data/manifests/fives_manifest.csv"
TRACE = ROOT / "data/manifests/trace_analysis_manifest.csv"
RFMID_ARCHIVE = ROOT / "A. RFMiD_All_Classes_Dataset.zip"
RFMID_SIGNATURES = ROOT / "tmp/trace_final_v1_audit_v1_signatures.json"
OUTPUT = ROOT / "data/manifests/trace_ssl_pool_v2_fives.csv"
CACHE = ROOT / "data/processed/trace_ssl_pool_v2_fives"
SUMMARY = ROOT / "data/protocols/trace_source_balance_v1/pool_fives_summary.json"


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def main() -> None:
    core_rows = read_csv(CORE)
    if len(core_rows) != 1917 or any(row["admission_status"] != "admitted" for row in core_rows):
        raise RuntimeError("Frozen core pool changed")
    if any(row["final_split"] != "train" for row in core_rows):
        raise RuntimeError("Core pool contains a non-training record")

    fives = pd.read_csv(FIVES, dtype=str, keep_default_na=False)
    selected = fives.loc[
        fives.source_split.eq("train")
        & fives.is_primary_analysis_record.str.lower().eq("true")
    ].copy()
    if len(selected) != 595:
        raise RuntimeError(f"Expected 595 unique FIVES train photographs, found {len(selected)}")
    if not selected.license.eq("CC-BY-4.0").all():
        raise RuntimeError("Unexpected FIVES licence")
    if selected.image_sha256.duplicated().any():
        raise RuntimeError("FIVES primary training hashes are not unique")
    if selected.record_id.duplicated().any():
        raise RuntimeError("FIVES training record IDs are not unique")

    trace = pd.read_csv(TRACE, dtype=str, keep_default_na=False)
    rfmid = trace.loc[trace.dataset.eq("RFMiD")].copy()
    if len(rfmid) != 3200:
        raise RuntimeError("RFMiD manifest record count changed")
    signature_cache = json.loads(RFMID_SIGNATURES.read_text(encoding="utf-8"))
    if signature_cache.get("archive_sha256") != sha256(RFMID_ARCHIVE):
        raise RuntimeError("RFMiD signature cache archive hash mismatch")
    if signature_cache.get("manifest_sha256") != sha256(TRACE):
        raise RuntimeError("RFMiD signature cache manifest hash mismatch")
    signatures = signature_cache.get("records", [])
    if len(signatures) != 3200 or {item["id"] for item in signatures} != set(rfmid.record_id):
        raise RuntimeError("RFMiD signature cache is incomplete")
    rfmid_exact_hashes = {item["sha256"] for item in signatures}
    exact_cross_source = sorted(set(selected.image_sha256) & rfmid_exact_hashes)
    if exact_cross_source:
        raise RuntimeError("Exact FIVES/RFMiD image overlap detected")
    trace_hashes = [int(item["dhash"]) for item in signatures]
    nearest_distances = {
        row.record_id: min(hamming_hex(row.perceptual_hash_dhash256, f"{value:064x}") for value in trace_hashes)
        for row in selected.itertuples(index=False)
    }
    if min(nearest_distances.values()) <= 32:
        raise RuntimeError("FIVES/RFMiD perceptual-overlap candidate requires review")

    pipeline = TracePreprocessor(PreprocessingConfig(apply_photometric_correction=False))
    loader = ManifestImageLoader(ROOT, FIVES, allow_test=False)
    CACHE.mkdir(parents=True, exist_ok=True)
    fives_rows: list[dict[str, str]] = []
    try:
        for number, row in enumerate(selected.sort_values("record_id").itertuples(index=False), 1):
            source_path = ROOT / row.image_path
            if sha256(source_path) != row.image_sha256:
                raise RuntimeError(f"FIVES source hash changed: {row.record_id}")
            target = CACHE / f"{hashlib.sha256(row.record_id.encode()).hexdigest()[:24]}.png"
            image, metadata = loader.load_image(row.record_id)
            if metadata["_effective_split"] != "train":
                raise RuntimeError(f"Non-training FIVES record reached cache: {row.record_id}")
            result = pipeline.process(
                image,
                image_id=row.record_id,
                metadata={"dataset": "FIVES", "source_split": "train"},
            )
            if result.status == ProcessingStatus.REJECT or result.processed_image is None:
                raise RuntimeError(f"Frozen preprocessing rejected FIVES training image: {row.record_id}")
            preprocessing_status = result.status.value
            if target.exists():
                bgr = cv2.imread(str(target), cv2.IMREAD_COLOR)
                if bgr is None or bgr.shape != (448, 448, 3):
                    raise RuntimeError(f"Malformed existing FIVES cache: {target}")
                cached_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if not np.array_equal(cached_rgb, result.processed_image):
                    raise RuntimeError(f"FIVES cache differs from frozen preprocessing: {target}")
            else:
                temp = target.with_suffix(".tmp.png")
                if not cv2.imwrite(str(temp), cv2.cvtColor(result.processed_image, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"Cannot write FIVES cache: {temp}")
                temp.replace(target)
            fives_rows.append({
                "record_id": row.record_id,
                "dataset_id": "fives",
                "source_id": "fives_train",
                "source_split": "train",
                "final_split": "train",
                "cache_path": str(target.relative_to(ROOT)).replace("\\", "/"),
                "source_image_sha256": row.image_sha256,
                "cache_image_sha256": sha256(target),
                "preprocessing_status": preprocessing_status,
                "patient_id": "",
                "patient_id_status": row.patient_id_status,
                "eye_id": "",
                "split_group_id": row.duplicate_group_id or row.record_id,
                "camera_model": "",
                "camera_domain": "fives_dataset_camera_not_provided",
                "ssl_v2_role": "expanded_training",
                "admission_status": "admitted",
                "admission_reason": (
                    "official_fives_train;primary_unique_photo;cc_by_4_0;"
                    "no_exact_or_low_distance_rfmid_overlap;fives_test_excluded"
                ),
            })
            if number % 50 == 0 or number == len(selected):
                print(f"FIVES SSL cache {number}/{len(selected)}", flush=True)
    finally:
        loader.close()

    output_rows = sorted(core_rows + fives_rows, key=lambda row: row["record_id"])
    if len(output_rows) != 2512:
        raise RuntimeError("Expanded pool record count changed")
    if len({row["record_id"] for row in output_rows}) != len(output_rows):
        raise RuntimeError("Expanded pool record IDs are not unique")
    if len({row["source_image_sha256"] for row in output_rows}) != len(output_rows):
        raise RuntimeError("Expanded pool contains an exact source-image duplicate")
    if any(row["final_split"] != "train" for row in output_rows):
        raise RuntimeError("Expanded pool contains a non-training record")

    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    status_counts = Counter(row["preprocessing_status"] for row in fives_rows)
    split_counts = Counter(rfmid.final_split)
    summary = {
        "version": "trace-ssl-pool-v2-rfmid-fives",
        "status": "EXPANDED_POOL_READY_FOR_A2_A3",
        "records": len(output_rows),
        "source_counts": {"rfmid_v1_train": len(core_rows), "fives_train": len(fives_rows)},
        "fives_release_rows": len(fives),
        "fives_train_rows": int(fives.source_split.eq("train").sum()),
        "fives_train_unique_primary_rows_admitted": len(fives_rows),
        "fives_test_rows_opened": 0,
        "fives_test_rows_admitted": 0,
        "fives_licence": "CC-BY-4.0",
        "fives_source_doi": "10.6084/m9.figshare.19688169.v1",
        "cross_source_exact_matches": 0,
        "minimum_fives_to_trace_dhash256_distance": min(nearest_distances.values()),
        "perceptual_candidates_at_or_below_32": sum(value <= 32 for value in nearest_distances.values()),
        "trace_hash_metadata_rows_screened_by_final_split": dict(sorted(split_counts.items())),
        "protected_trace_test_images_opened": 0,
        "fives_preprocessing_status_counts": dict(sorted(status_counts.items())),
        "disease_label_fields_present": False,
        "patient_ids_available_for_fives": False,
        "camera_ids_available_for_fives": False,
        "core_manifest_sha256": sha256(CORE),
        "fives_manifest_sha256": sha256(FIVES),
        "trace_manifest_sha256": sha256(TRACE),
        "rfmid_signature_cache_sha256": sha256(RFMID_SIGNATURES),
        "rfmid_signature_cache_records_reused": len(signatures),
        "protected_trace_test_images_opened_during_this_build": 0,
        "output_manifest_sha256": sha256(OUTPUT),
        "role_decision": (
            "FIVES train is encoder-SSL training only; FIVES test remains unopened and excluded. "
            "FIVES is retired from future encoder evaluation. Earlier preprocessing/QC analyses are separate."
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
