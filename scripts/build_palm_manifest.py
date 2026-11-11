"""Build and validate a TRACE manifest for the public PALM archive.

The archive remains untouched.  Images, spreadsheets and masks are read directly
from the ZIP.  The official train/validation/test split is preserved because the
PALM paper reports that images from the same patient do not cross those splits.
Patient identifiers are not included in the release, so that property is recorded
as source-reported rather than independently re-derived.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import openpyxl
from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[1]
ARCHIVE_RELATIVE = Path("data/external/PALM/PALM.zip")
OUTPUT_DIR = WORKSPACE / "data" / "manifests"
EXPECTED_BYTES = 1_659_253_782
EXPECTED_MD5 = "28636601413706884ea89ec1be835e94"

SPLITS = {
    "Training": "train",
    "Validation": "validation",
    "Testing": "test",
}
DEVICE = {
    1: ("ZEISS Visucam 500", "palm_camera_zeiss_visucam_500"),
    2: ("Canon CR-2", "palm_camera_canon_cr_2"),
}
SHOOTING_CENTER = {
    1: "optic_disc",
    2: "fovea",
    3: "midpoint_optic_disc_and_fovea",
}


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec - integrity check against publisher metadata
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workbook_rows(archive: zipfile.ZipFile, member: str) -> list[dict[str, Any]]:
    workbook = openpyxl.load_workbook(
        io.BytesIO(archive.read(member)), read_only=True, data_only=True
    )
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))
    headers = [str(value).strip() if value is not None else "" for value in rows[0]]
    result: list[dict[str, Any]] = []
    for raw in rows[1:]:
        row = {headers[index]: value for index, value in enumerate(raw) if headers[index]}
        if any(value is not None for value in row.values()):
            result.append(row)
    return result


def keyed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["imgName"]).strip(): row for row in rows}


def image_info_and_sha256(
    archive: zipfile.ZipFile, member: str
) -> tuple[int, int, str, str, str]:
    raw = archive.read(member)
    sha256 = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as image:
        width, height = image.size
        mode = image.mode
        image_format = image.format or PurePosixPath(member).suffix.lstrip(".")
    return width, height, mode, image_format.upper(), sha256


def mask_info(
    archive: zipfile.ZipFile, member: str | None
) -> tuple[int | str, int | str, int | str]:
    if not member:
        return "", "", ""
    with archive.open(member) as stream:
        with Image.open(stream) as image:
            array = np.asarray(image)
            width, height = image.size
    return width, height, int(np.count_nonzero(array))


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    archive_path = WORKSPACE / ARCHIVE_RELATIVE
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if archive_path.stat().st_size != EXPECTED_BYTES:
        raise ValueError(f"PALM size mismatch: {archive_path.stat().st_size}")
    md5 = file_md5(archive_path)
    if md5 != EXPECTED_MD5:
        raise ValueError(f"PALM MD5 mismatch: {md5}")

    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"PALM ZIP CRC failure: {bad_member}")
        names = set(archive.namelist())

        for source_name, final_split in SPLITS.items():
            base = f"PALM/{source_name}"
            labels = keyed(
                workbook_rows(archive, f"{base}/Classification Labels.xlsx")
            )
            supplementary = keyed(
                workbook_rows(archive, f"{base}/Supplementary Information.xlsx")
            )
            fovea = keyed(workbook_rows(archive, f"{base}/Fovea Localization.xlsx"))
            image_members = sorted(
                name
                for name in names
                if name.startswith(f"{base}/Images/")
                and name.lower().endswith(".jpg")
            )
            if not (
                len(image_members)
                == len(labels)
                == len(supplementary)
                == len(fovea)
                == 400
            ):
                raise ValueError(
                    f"PALM {source_name} metadata mismatch: "
                    f"images={len(image_members)} labels={len(labels)} "
                    f"supplementary={len(supplementary)} fovea={len(fovea)}"
                )

            for image_member in image_members:
                image_name = PurePosixPath(image_member).name
                stem = PurePosixPath(image_name).stem
                label_row = labels[image_name]
                supplementary_row = supplementary[image_name]
                fovea_row = fovea[image_name]
                device_code = int(
                    supplementary_row["Device(1:Zeiss, 2: Canon)"]
                )
                camera_model, camera_domain = DEVICE[device_code]
                center_code = int(
                    supplementary_row[
                        "Shooting Center(1: OD, 2: fovea, 3: midpoint of OD and fovea)"
                    ]
                )
                width, height, mode, image_format, image_sha256 = (
                    image_info_and_sha256(archive, image_member)
                )
                mask_candidates = {
                    "disc_mask": f"{base}/Disc Masks/{stem}.bmp",
                    "atrophy_mask": f"{base}/Lesion Masks/Atrophy/{stem}.bmp",
                    "detachment_mask": f"{base}/Lesion Masks/Detachment/{stem}.bmp",
                }
                masks = {
                    key: member if member in names else ""
                    for key, member in mask_candidates.items()
                }
                mask_metrics: dict[str, Any] = {}
                for key, member in masks.items():
                    mask_width, mask_height, foreground = mask_info(
                        archive, member or None
                    )
                    if member and (mask_width, mask_height) != (width, height):
                        raise ValueError(
                            f"Mask size mismatch for {image_name}: {key} "
                            f"{mask_width}x{mask_height} vs {width}x{height}"
                        )
                    mask_metrics[f"{key}_width"] = mask_width
                    mask_metrics[f"{key}_height"] = mask_height
                    mask_metrics[f"{key}_foreground_pixels"] = foreground

                info = archive.getinfo(image_member)
                records.append(
                    {
                        "record_id": f"palm:{source_name.lower()}:{stem}",
                        "dataset": "PALM",
                        "modality": "color_fundus_photograph",
                        "archive_path": ARCHIVE_RELATIVE.as_posix(),
                        "archive_member": image_member,
                        "source_split": source_name.lower(),
                        "source_image_id": stem,
                        "patient_id": "",
                        "patient_id_status": "not_provided_official_split_reported_patient_disjoint",
                        "camera_model": camera_model,
                        "camera_domain": camera_domain,
                        "camera_domain_source": "official_palm_supplementary_information",
                        "device_code": device_code,
                        "shooting_center_code": center_code,
                        "shooting_center": SHOOTING_CENTER[center_code],
                        "width": width,
                        "height": height,
                        "channels": 3 if mode == "RGB" else "",
                        "image_mode": mode,
                        "image_format": image_format,
                        "zip_crc32": f"{info.CRC:08x}",
                        "uncompressed_bytes": info.file_size,
                        "image_sha256": image_sha256,
                        "license": "CC0-1.0",
                        "source_doi": "10.6084/m9.figshare.21299148.v1",
                        "pathologic_myopia": int(label_row["Label"]),
                        "fovea_x": float(fovea_row["Fovea_X"]),
                        "fovea_y": float(fovea_row["Fovea_Y"]),
                        **masks,
                        **mask_metrics,
                        "final_split": final_split,
                        "final_split_status": "official_palm_split_source_reported_patient_disjoint",
                    }
                )

    records.sort(key=lambda row: row["record_id"])

    # The public archive contains byte-identical photographs under different IDs,
    # including copies that cross the official splits.  Preserve every source row
    # for audit, but retain exactly one analysis record per SHA-256.  Test has the
    # highest priority so no duplicate of a protected test photograph can enter
    # development; validation then has priority over training.
    by_sha256: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_sha256[row["image_sha256"]].append(row)
    duplicate_groups = {
        sha256: rows for sha256, rows in by_sha256.items() if len(rows) > 1
    }
    split_priority = {"test": 0, "validation": 1, "train": 2}
    for group_number, (sha256, rows) in enumerate(
        sorted(duplicate_groups.items()), start=1
    ):
        group_id = f"palm_exact_duplicate_{group_number:04d}"
        primary = min(
            rows,
            key=lambda row: (split_priority[row["final_split"]], row["record_id"]),
        )
        source_splits = {row["final_split"] for row in rows}
        crosses_split = len(source_splits) > 1
        for row in rows:
            row["exact_duplicate_group_id"] = group_id
            row["exact_duplicate_crosses_official_splits"] = crosses_split
            if row is primary:
                row["exact_duplicate_status"] = "primary_analysis_record"
                row["is_primary_analysis_record"] = True
                row["effective_split"] = row["final_split"]
                row["effective_split_status"] = (
                    "duplicate_group_primary_test_priority"
                    if crosses_split
                    else "duplicate_group_primary_within_split"
                )
            else:
                row["exact_duplicate_status"] = "excluded_exact_duplicate"
                row["is_primary_analysis_record"] = False
                row["effective_split"] = "excluded"
                row["effective_split_status"] = (
                    "excluded_cross_split_exact_duplicate"
                    if crosses_split
                    else "excluded_within_split_exact_duplicate"
                )
    for sha256, rows in by_sha256.items():
        if sha256 in duplicate_groups:
            continue
        row = rows[0]
        row["exact_duplicate_group_id"] = ""
        row["exact_duplicate_crosses_official_splits"] = False
        row["exact_duplicate_status"] = "unique"
        row["is_primary_analysis_record"] = True
        row["effective_split"] = row["final_split"]
        row["effective_split_status"] = "official_split_unique_image"

    primary_records = [row for row in records if row["is_primary_analysis_record"]]
    cross_split_groups = sum(
        len({row["final_split"] for row in rows}) > 1
        for rows in duplicate_groups.values()
    )
    summary = {
        "status": "PASS",
        "archive_path": ARCHIVE_RELATIVE.as_posix(),
        "archive_bytes": EXPECTED_BYTES,
        "archive_md5": md5,
        "zip_crc_check": "PASS",
        "records": len(records),
        "unique_image_sha256": len({row["image_sha256"] for row in records}),
        "primary_analysis_records": len(primary_records),
        "exact_duplicate_groups": len(duplicate_groups),
        "exact_duplicate_records": sum(len(rows) for rows in duplicate_groups.values()),
        "cross_official_split_exact_duplicate_groups": cross_split_groups,
        "excluded_exact_duplicate_records": sum(
            not row["is_primary_analysis_record"] for row in records
        ),
        "records_by_source_split": dict(Counter(row["source_split"] for row in records)),
        "records_by_final_split": dict(Counter(row["final_split"] for row in records)),
        "primary_records_by_effective_split": dict(
            Counter(row["effective_split"] for row in primary_records)
        ),
        "excluded_duplicate_records_by_source_split": dict(
            Counter(
                row["final_split"]
                for row in records
                if not row["is_primary_analysis_record"]
            )
        ),
        "records_by_camera_model": dict(Counter(row["camera_model"] for row in records)),
        "records_by_pathologic_myopia": dict(
            Counter(str(row["pathologic_myopia"]) for row in records)
        ),
        "records_by_shooting_center": dict(
            Counter(row["shooting_center"] for row in records)
        ),
        "disc_masks": sum(bool(row["disc_mask"]) for row in records),
        "atrophy_masks": sum(bool(row["atrophy_mask"]) for row in records),
        "detachment_masks": sum(bool(row["detachment_mask"]) for row in records),
        "empty_present_masks": sum(
            bool(row[mask]) and int(row[f"{mask}_foreground_pixels"]) == 0
            for row in records
            for mask in ("disc_mask", "atrophy_mask", "detachment_mask")
        ),
        "patient_split_note": (
            "The paper reports patient separation, but patient identifiers are not "
            "present for independent verification. Exact image duplicates cross the "
            "official splits, so a leakage-safe primary-analysis view excludes duplicate "
            "copies and gives protected test records priority."
        ),
        "test_use_policy": (
            "Do not tune preprocessing or thresholds on PALM test records. Use only "
            "is_primary_analysis_record=True rows, and never use effective_split=excluded."
        ),
    }
    if summary["records"] != 1200:
        raise ValueError(f"Expected 1200 PALM records, found {summary['records']}")
    if summary["primary_analysis_records"] != 1184:
        raise ValueError(
            f"Expected 1184 unique PALM images, found {summary['primary_analysis_records']}"
        )
    if summary["cross_official_split_exact_duplicate_groups"] != 11:
        raise ValueError(
            "Unexpected PALM cross-split duplicate count: "
            f"{summary['cross_official_split_exact_duplicate_groups']}"
        )
    if summary["records_by_camera_model"] != {
        "ZEISS Visucam 500": 1047,
        "Canon CR-2": 153,
    }:
        raise ValueError(f"Unexpected PALM device counts: {summary['records_by_camera_model']}")
    if summary["records_by_pathologic_myopia"] != {"0": 563, "1": 637}:
        raise ValueError(
            f"Unexpected PALM label counts: {summary['records_by_pathologic_myopia']}"
        )
    if (
        summary["disc_masks"],
        summary["atrophy_masks"],
        summary["detachment_masks"],
    ) != (1144, 870, 24):
        raise ValueError("Unexpected PALM mask coverage")
    return records, summary


def main() -> None:
    records, summary = build()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "palm_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUTPUT_DIR / "palm_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
