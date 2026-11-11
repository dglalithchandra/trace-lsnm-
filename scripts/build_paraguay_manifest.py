"""Build and validate the external-development manifest for Paraguay DR V03."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[1]
ARCHIVE = (
    WORKSPACE
    / "data"
    / "external"
    / "ParaguayDR"
    / "Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V03.zip"
)
OUTPUT_DIR = WORKSPACE / "data" / "manifests"
MANIFEST_PATH = OUTPUT_DIR / "paraguay_dr_manifest.csv"
SUMMARY_PATH = OUTPUT_DIR / "paraguay_dr_manifest_summary.json"
WORKBOOK_MEMBER = "Annotations of the classifications.xlsx"
EXPECTED_MD5 = "1d9109e010533c408ed362ac9b1bbc9d"
EXPECTED_BYTES = 1_523_619_607

FOLDER_TO_LABEL = {
    "1. No DR signs": (0, "No DR signs"),
    "2. Mild (or early) NPDR": (1, "Mild (or early) NPDR"),
    "3. Moderate NPDR": (2, "Moderate NPDR"),
    "4. Severe NPDR": (3, "Severe NPDR"),
    "5. Very Severe NPDR": (4, "Very Severe NPDR"),
    "6. PDR": (5, "PDR"),
    "7. Advanced PDR": (6, "Advanced PDR"),
}


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def known_manifest_hash_sources() -> dict[str, list[str]]:
    sources: dict[str, list[str]] = defaultdict(list)
    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        if path == MANIFEST_PATH:
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "image_sha256" not in reader.fieldnames:
                continue
            for row in reader:
                value = (row.get("image_sha256") or "").strip().lower()
                if value:
                    sources[value].append(path.name)
    return sources


def main() -> None:
    if not ARCHIVE.is_file() or ARCHIVE.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError("Paraguay archive missing or byte size mismatch")
    md5 = digest_file(ARCHIVE, "md5")
    if md5 != EXPECTED_MD5:
        raise RuntimeError(f"Paraguay archive MD5 mismatch: {md5}")
    archive_sha256 = digest_file(ARCHIVE, "sha256")
    existing_hashes = known_manifest_hash_sources()

    rows: list[dict[str, object]] = []
    decode_failures: list[str] = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP integrity failed at: {bad_member}")
        workbook = pd.read_excel(io.BytesIO(archive.read(WORKBOOK_MEMBER)))
        if workbook.columns.tolist() != ["Image", "Format", "Status"]:
            raise RuntimeError(f"Unexpected Paraguay workbook columns: {workbook.columns}")
        if len(workbook) != 1437 or workbook["Image"].duplicated().any():
            raise RuntimeError("Unexpected Paraguay workbook identifier integrity")
        workbook_by_id = workbook.set_index("Image")

        image_members = sorted(
            (name for name in archive.namelist() if name.lower().endswith(".jpg")),
            key=lambda name: int(PurePosixPath(name).stem),
        )
        if len(image_members) != 1437:
            raise RuntimeError(f"Unexpected Paraguay image count: {len(image_members)}")
        for index, member in enumerate(image_members, start=1):
            path = PurePosixPath(member)
            folder = path.parts[0]
            if folder not in FOLDER_TO_LABEL:
                raise RuntimeError(f"Unexpected Paraguay label folder: {folder}")
            image_id = int(path.stem)
            if image_id not in workbook_by_id.index:
                raise RuntimeError(f"Image missing from workbook: {image_id}")
            grade, expected_status = FOLDER_TO_LABEL[folder]
            workbook_row = workbook_by_id.loc[image_id]
            if str(workbook_row["Format"]).lower() != "jpg":
                raise RuntimeError(f"Unexpected format for image {image_id}")
            if str(workbook_row["Status"]) != expected_status:
                raise RuntimeError(
                    f"Folder/workbook label mismatch for image {image_id}: "
                    f"{expected_status} != {workbook_row['Status']}"
                )
            payload = archive.read(member)
            image_sha256 = hashlib.sha256(payload).hexdigest()
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    width, height = image.size
                    mode = image.mode
                    image.verify()
            except Exception:
                decode_failures.append(member)
                continue
            info = archive.getinfo(member)
            rows.append(
                {
                    "record_id": f"paraguay_dr:{image_id:04d}",
                    "dataset": "Paraguay DR V03",
                    "modality": "color_fundus_photograph",
                    "archive_path": str(ARCHIVE.relative_to(WORKSPACE)).replace("\\", "/"),
                    "archive_member": member,
                    "source_image_id": image_id,
                    "patient_id": "",
                    "patient_id_status": "not_provided",
                    "camera_model": "ZEISS Visucam 500",
                    "camera_metadata_source": "official Zenodo record 4891308",
                    "acquisition_site": "Department of Ophthalmology, Hospital de Clinicas, Universidad Nacional de Asuncion, Paraguay",
                    "dr_grade": grade,
                    "dr_grade_name": expected_status,
                    "any_dr": int(grade >= 1),
                    "referable_dr": int(grade >= 2),
                    "proliferative_dr": int(grade >= 5),
                    "width": width,
                    "height": height,
                    "image_mode": mode,
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "zip_crc32": f"{info.CRC:08x}",
                    "image_sha256": image_sha256,
                    "exact_match_in_existing_manifest": int(image_sha256 in existing_hashes),
                    "exact_match_manifest_sources": json.dumps(
                        sorted(set(existing_hashes.get(image_sha256, [])))
                    ),
                    "license": "CC BY 4.0",
                    "source_doi": "10.5281/zenodo.4891308",
                    "analysis_split": "external_development_validation",
                    "protected_final_test": False,
                }
            )
            if index % 100 == 0 or index == len(image_members):
                print(f"paraguay_manifest_progress {index}/{len(image_members)}", flush=True)

    if decode_failures or len(rows) != 1437:
        raise RuntimeError(f"Paraguay decode failures: {decode_failures[:10]}")
    frame = pd.DataFrame(rows)
    hash_counts = frame["image_sha256"].value_counts()
    duplicate_hashes = set(hash_counts[hash_counts > 1].index)
    frame["exact_duplicate_within_dataset"] = frame["image_sha256"].isin(
        duplicate_hashes
    )
    group_by_hash = {
        value: f"paraguay_exact_duplicate_{index:04d}"
        for index, value in enumerate(sorted(duplicate_hashes), start=1)
    }
    frame["exact_duplicate_group_id"] = frame["image_sha256"].map(group_by_hash).fillna("")
    frame["is_primary_analysis_record"] = ~frame.duplicated(
        subset=["image_sha256"], keep="first"
    )
    frame.loc[~frame["is_primary_analysis_record"], "analysis_split"] = "excluded_duplicate"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MANIFEST_PATH, index=False)
    dimension_counts = Counter(
        f"{width}x{height}" for width, height in zip(frame["width"], frame["height"], strict=True)
    )
    summary = {
        "status": "PASS",
        "archive": str(ARCHIVE.relative_to(WORKSPACE)).replace("\\", "/"),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_md5": md5,
        "archive_sha256": archive_sha256,
        "zip_integrity": "PASS",
        "records": len(frame),
        "primary_analysis_records": int(frame["is_primary_analysis_record"].sum()),
        "label_counts": {
            str(name): int(count)
            for name, count in frame["dr_grade_name"].value_counts().sort_index().items()
        },
        "any_dr_positive": int(frame["any_dr"].sum()),
        "referable_dr_positive": int(frame["referable_dr"].sum()),
        "proliferative_dr_positive": int(frame["proliferative_dr"].sum()),
        "dimension_counts": dict(dimension_counts.most_common()),
        "within_dataset_duplicate_hashes": len(duplicate_hashes),
        "within_dataset_duplicate_records": int(
            frame["exact_duplicate_within_dataset"].sum()
        ),
        "exact_matches_in_existing_manifests": int(
            frame["exact_match_in_existing_manifest"].sum()
        ),
        "patient_ids_available": False,
        "camera_model": "ZEISS Visucam 500",
        "acquisition_site": "Hospital de Clinicas, Paraguay",
        "role": "external development validation for DR site/domain shift; not rare-disease training",
        "protected_test_records_accessed": 0,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
