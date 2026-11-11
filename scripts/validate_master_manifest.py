"""Validate TRACE's canonical manifest against the untouched source archives.

This is intentionally separate from ``build_master_manifest.py`` so a build bug
cannot silently pass by reusing the exact same checks.  It validates structure,
known source counts, annotation coverage, duplicate grouping, and every ZIP
member referenced by the manifest.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd


EXPECTED_DATASETS = {
    "RFMiD": 3200,
    "IDRiD_Grading_Localisation": 516,
    "IDRiD_Segmentation": 81,
}

EXPECTED_SPLITS = {
    ("RFMiD", "training"): 1920,
    ("RFMiD", "validation"): 640,
    ("RFMiD", "testing"): 640,
    ("IDRiD_Grading_Localisation", "training"): 413,
    ("IDRiD_Grading_Localisation", "testing"): 103,
    ("IDRiD_Segmentation", "training"): 54,
    ("IDRiD_Segmentation", "testing"): 27,
}

EXPECTED_MASK_COUNTS = {
    "mask_microaneurysms": 81,
    "mask_haemorrhages": 80,
    "mask_hard_exudates": 81,
    "mask_soft_exudates": 40,
    "mask_optic_disc": 81,
}

REQUIRED_COLUMNS = {
    "record_id",
    "dataset",
    "archive_path",
    "archive_member",
    "source_split",
    "source_image_id",
    "patient_id_status",
    "camera_model",
    "camera_domain",
    "camera_domain_source",
    "width",
    "height",
    "channels",
    "disease_labels_json",
    "dr_grade",
    "dme_risk",
    "duplicate_group_id",
    "duplicate_status",
    "final_split",
    "final_split_status",
} | set(EXPECTED_MASK_COUNTS)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def count_not_blank(series: pd.Series) -> int:
    return int(series.fillna("").astype(str).str.strip().ne("").sum())


def validate(workspace: Path) -> dict[str, object]:
    manifest_path = workspace / "data" / "manifests" / "master_manifest.csv"
    pairs_path = workspace / "data" / "manifests" / "duplicate_pairs.csv"
    check(manifest_path.is_file(), f"Missing manifest: {manifest_path}")
    check(pairs_path.is_file(), f"Missing duplicate-pair table: {pairs_path}")

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    pairs = pd.read_csv(pairs_path, dtype=str, keep_default_na=False)

    check(REQUIRED_COLUMNS <= set(manifest.columns), "Manifest is missing required columns")
    check(len(manifest) == 3797, f"Expected 3797 records, found {len(manifest)}")
    check(manifest["record_id"].is_unique, "record_id values are not unique")
    check(manifest["archive_member"].ne("").all(), "One or more image members are blank")

    dataset_counts = manifest["dataset"].value_counts().to_dict()
    check(dataset_counts == EXPECTED_DATASETS, f"Unexpected dataset counts: {dataset_counts}")

    split_counts = Counter(zip(manifest["dataset"], manifest["source_split"]))
    check(dict(split_counts) == EXPECTED_SPLITS, f"Unexpected source splits: {dict(split_counts)}")

    rfmid = manifest[manifest["dataset"] == "RFMiD"]
    parsed_label_lengths = []
    for raw in rfmid["disease_labels_json"]:
        labels = json.loads(raw)
        check(isinstance(labels, dict), "RFMiD disease_labels_json is not an object")
        parsed_label_lengths.append(len(labels))
    check(set(parsed_label_lengths) == {45}, "Every RFMiD record must contain all 45 labels")
    check(count_not_blank(rfmid["disease_risk"]) == 3200, "RFMiD disease-risk labels incomplete")

    grading = manifest[manifest["dataset"] == "IDRiD_Grading_Localisation"]
    check(count_not_blank(grading["dr_grade"]) == 516, "IDRiD DR grades incomplete")
    check(count_not_blank(grading["dme_risk"]) == 516, "IDRiD DME labels incomplete")
    for coord in ("optic_disc_x", "optic_disc_y", "fovea_x", "fovea_y"):
        check(count_not_blank(grading[coord]) == 516, f"IDRiD coordinate coverage incomplete: {coord}")

    mask_counts = {column: count_not_blank(manifest[column]) for column in EXPECTED_MASK_COUNTS}
    check(mask_counts == EXPECTED_MASK_COUNTS, f"Unexpected mask counts: {mask_counts}")

    # Camera labels may be derived from official acquisition tables, but the
    # downloaded RFMiD archive does not exactly match the paper's camera counts.
    # Keep the entire 4288 x 2848 RFMiD group unresolved rather than inventing
    # 30 per-image assignments.
    rfmid_camera_counts = rfmid["camera_model"].replace("", "unresolved").value_counts().to_dict()
    check(
        rfmid_camera_counts
        == {
            "TOPCON 3D OCT-2000": 2427,
            "unresolved": 497,
            "TOPCON TRC-NW300": 276,
        },
        f"Unexpected RFMiD camera-evidence counts: {rfmid_camera_counts}",
    )
    check(
        rfmid.loc[rfmid["camera_model"].eq(""), "camera_domain"].eq(
            "rfmid_resolution_4288x2848_unresolved"
        ).all(),
        "Unresolved RFMiD camera records are not isolated",
    )
    idrid = manifest[manifest["dataset"].str.startswith("IDRiD")]
    check(
        idrid["camera_model"].eq("Kowa VX-10 alpha").all(),
        "IDRiD camera mapping does not match the official specification",
    )
    allowed_camera_sources = {
        "official_rfmid_paper_resolution_mapping_count_consistent",
        "official_rfmid_paper_resolution_mapping_archive_count_mismatch",
        "official_rfmid_paper_resolution_mapping_archive_count_mismatch_unresolved",
        "official_idrid_camera_specification",
    }
    check(
        set(manifest["camera_domain_source"]) <= allowed_camera_sources,
        "Unexpected camera evidence source",
    )
    check(manifest["patient_id_status"].eq("not_provided").all(), "Unexpected patient-ID status")

    check(manifest["final_split"].eq("").all(), "final_split must stay blank before group assignment")
    check(
        manifest["final_split_status"].eq("pending_duplicate_aware_assignment").all(),
        "Unexpected final-split status",
    )

    # Confirm every referenced image/mask exists in its declared ZIP.  Opening
    # each archive once keeps this check fast even for thousands of references.
    missing_members: list[str] = []
    mask_columns = list(EXPECTED_MASK_COUNTS)
    for archive_name, rows in manifest.groupby("archive_path"):
        archive_path = workspace / archive_name
        check(archive_path.is_file(), f"Missing source archive: {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            requested = set(rows["archive_member"])
            for column in mask_columns:
                requested.update(value for value in rows[column] if value)
            for member in sorted(requested - names):
                missing_members.append(f"{archive_name}::{member}")
    check(not missing_members, f"Manifest references missing ZIP members: {missing_members[:10]}")

    confirmed_pairs = pairs[pairs["relation"] == "confirmed_perceptual_duplicate"]
    check(len(confirmed_pairs) == 60, f"Expected 60 confirmed duplicate pairs, found {len(confirmed_pairs)}")
    confirmed_records = manifest[manifest["duplicate_status"] == "confirmed_perceptual_duplicate"]
    check(len(confirmed_records) == 116, "Expected 116 confirmed-duplicate records")
    group_sizes = confirmed_records.groupby("duplicate_group_id").size().to_dict()
    check(len(group_sizes) == 56, f"Expected 56 confirmed groups, found {len(group_sizes)}")
    check(Counter(group_sizes.values()) == {2: 52, 3: 4}, f"Unexpected duplicate group sizes: {group_sizes}")

    report: dict[str, object] = {
        "status": "PASS",
        "canonical_records": len(manifest),
        "unique_record_ids": int(manifest["record_id"].nunique()),
        "records_by_dataset": dataset_counts,
        "records_by_source_split": {
            f"{dataset}/{split}": count
            for (dataset, split), count in sorted(split_counts.items())
        },
        "rfmid_label_columns_per_record": 45,
        "idrid_grading_records_with_labels_and_coordinates": 516,
        "segmentation_mask_counts": mask_counts,
        "confirmed_duplicate_pairs": len(confirmed_pairs),
        "confirmed_duplicate_groups": len(group_sizes),
        "confirmed_duplicate_records": len(confirmed_records),
        "camera_identity": "derived from official acquisition tables with RFMiD 4288x2848 unresolved",
        "rfmid_camera_model_counts": rfmid_camera_counts,
        "camera_domains": "two RFMiD models derived, one RFMiD resolution group unresolved, IDRiD Kowa specified",
        "final_split": "pending duplicate-aware assignment",
        "archive_members_checked": True,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Workspace containing the archives and data/manifests directory",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    report = validate(workspace)
    output = workspace / "data" / "manifests" / "manifest_validation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
