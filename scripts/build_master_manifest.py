"""Build a duplicate-aware manifest directly from the downloaded ZIP archives.

This script deliberately runs before extraction.  Camera evidence is derived only
where an official acquisition table links a native resolution to a named camera.
The RFMiD 4288 x 2848 group stays unresolved because the downloaded archive has
30 more images at that resolution than the paper's Kowa count and 30 fewer at the
TRC-NW300 resolution.  It merges IDRiD grading and localisation annotations because
those archives contain byte-identical copies of the same 516 photographs.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
from PIL import Image


RFMID_ZIP = "A. RFMiD_All_Classes_Dataset.zip"
IDRID_SEG_ZIP = "A. Segmentation.zip"
IDRID_GRADE_ZIP = "B. Disease Grading.zip"
IDRID_LOC_ZIP = "C. Localization.zip"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SPLIT_BY_FOLDER = {
    "a. Training Set": "training",
    "b. Validation Set": "validation",
    "b. Testing Set": "testing",
    "c. Testing Set": "testing",
}
RFMID_CAMERA_BY_RESOLUTION = {
    (2144, 1424): {
        "model": "TOPCON 3D OCT-2000",
        "domain": "rfmid_camera_topcon_3d_oct_2000",
        "source": "official_rfmid_paper_resolution_mapping_count_consistent",
    },
    (2048, 1536): {
        "model": "TOPCON TRC-NW300",
        "domain": "rfmid_camera_topcon_trc_nw300",
        "source": "official_rfmid_paper_resolution_mapping_archive_count_mismatch",
    },
}
RFMID_UNRESOLVED_4288_DOMAIN = "rfmid_resolution_4288x2848_unresolved"
RFMID_UNRESOLVED_4288_SOURCE = (
    "official_rfmid_paper_resolution_mapping_archive_count_mismatch_unresolved"
)
IDRID_CAMERA_MODEL = "Kowa VX-10 alpha"
IDRID_CAMERA_DOMAIN = "idrid_camera_kowa_vx_10_alpha"
IDRID_CAMERA_SOURCE = "official_idrid_camera_specification"
MASK_SUFFIX_TO_COLUMN = {
    "MA": "mask_microaneurysms",
    "HE": "mask_haemorrhages",
    "EX": "mask_hard_exudates",
    "SE": "mask_soft_exudates",
    "OD": "mask_optic_disc",
}


@dataclass
class Thumbnail:
    record_id: str
    dhash_hex: str
    pixels: np.ndarray


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/manifests")
    )
    parser.add_argument(
        "--skip-perceptual-overlap",
        action="store_true",
        help="Skip the slower RFMiD/IDRiD perceptual overlap scan.",
    )
    return parser.parse_args()


def read_csv_member(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    text = archive.read(member).decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return []
    headers = [header.strip() for header in rows[0]]
    result: list[dict[str, str]] = []
    for raw in rows[1:]:
        padded = list(raw) + [""] * max(0, len(headers) - len(raw))
        row = {
            header: padded[index].strip()
            for index, header in enumerate(headers)
            if header
        }
        if any(row.values()):
            result.append(row)
    return result


def find_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one member ending in {suffix!r}, found {matches}")
    return matches[0]


def image_metadata(
    archive: zipfile.ZipFile, member: str
) -> tuple[int, int, str, str]:
    with archive.open(member) as stream:
        with Image.open(stream) as image:
            width, height = image.size
            mode = image.mode
            image_format = image.format or PurePosixPath(member).suffix.lstrip(".")
    return width, height, mode, image_format.upper()


def thumbnail_and_hash(archive: zipfile.ZipFile, member: str, record_id: str) -> Thumbnail:
    with archive.open(member) as stream:
        with Image.open(stream) as image:
            rgb = image.convert("RGB").resize((256, 170), Image.Resampling.BILINEAR)
            pixels = np.asarray(rgb, dtype=np.uint8)
            gray = image.convert("L").resize((17, 16), Image.Resampling.LANCZOS)
            gray_array = np.asarray(gray, dtype=np.int16)
    bits = gray_array[:, :-1] > gray_array[:, 1:]
    packed = np.packbits(bits.reshape(-1))
    return Thumbnail(record_id, packed.tobytes().hex(), pixels)


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def similarity(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    a = left.astype(np.float32).reshape(-1)
    b = right.astype(np.float32).reshape(-1)
    a_std = float(a.std())
    b_std = float(b.std())
    correlation = 0.0 if a_std == 0 or b_std == 0 else float(np.corrcoef(a, b)[0, 1])
    normalized_mae = float(np.mean(np.abs(a - b)) / 255.0)
    return correlation, normalized_mae


def split_from_member(member: str) -> str:
    for folder, split in SPLIT_BY_FOLDER.items():
        if f"/{folder}/" in member:
            return split
    raise ValueError(f"Could not determine split for {member}")


def base_record(
    *,
    record_id: str,
    dataset: str,
    archive_path: Path,
    archive: zipfile.ZipFile,
    member: str,
    source_split: str,
    source_image_id: str,
    width: int,
    height: int,
    mode: str,
    image_format: str,
) -> dict[str, Any]:
    info = archive.getinfo(member)
    resolution = f"{width}x{height}"
    camera_model = ""
    if dataset == "RFMiD":
        evidence = RFMID_CAMERA_BY_RESOLUTION.get((width, height))
        if evidence:
            camera_model = evidence["model"]
            domain = evidence["domain"]
            domain_source = evidence["source"]
        elif (width, height) == (4288, 2848):
            domain = RFMID_UNRESOLVED_4288_DOMAIN
            domain_source = RFMID_UNRESOLVED_4288_SOURCE
        else:
            domain = f"rfmid_resolution_{resolution}_unresolved"
            domain_source = "native_resolution_not_in_official_camera_table"
    elif dataset.startswith("IDRiD"):
        if (width, height) == (4288, 2848):
            camera_model = IDRID_CAMERA_MODEL
            domain = IDRID_CAMERA_DOMAIN
            domain_source = IDRID_CAMERA_SOURCE
        else:
            domain = f"idrid_resolution_{resolution}_unresolved"
            domain_source = "native_resolution_not_in_official_camera_table"
    else:
        domain = f"unknown_resolution_{resolution}"
        domain_source = "native_resolution_proxy_unverified"
    return {
        "record_id": record_id,
        "dataset": dataset,
        "modality": "color_fundus_photograph",
        "archive_path": archive_path.name,
        "archive_member": member,
        "source_split": source_split,
        "source_image_id": source_image_id,
        "patient_id": "",
        "patient_id_status": "not_provided",
        "camera_model": camera_model,
        "camera_domain": domain,
        "camera_domain_source": domain_source,
        "width": width,
        "height": height,
        "channels": 3 if mode == "RGB" else "",
        "image_mode": mode,
        "image_format": image_format,
        "zip_crc32": f"{info.CRC:08x}",
        "uncompressed_bytes": info.file_size,
        "license": "CC-BY-4.0",
        "disease_labels_json": "{}",
        "disease_risk": "",
        "dr_grade": "",
        "dme_risk": "",
        "optic_disc_x": "",
        "optic_disc_y": "",
        "fovea_x": "",
        "fovea_y": "",
        "mask_microaneurysms": "",
        "mask_haemorrhages": "",
        "mask_hard_exudates": "",
        "mask_soft_exudates": "",
        "mask_optic_disc": "",
        "quality_label": "",
        "quality_label_status": "not_provided",
        "perceptual_hash_dhash256": "",
        "duplicate_group_id": "",
        "duplicate_status": "none_detected",
        "related_record_id": "",
        "related_status": "",
        "related_similarity": "",
        "final_split": "",
        "final_split_status": "pending_duplicate_aware_assignment",
    }


def rfmid_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, Thumbnail]]:
    path = root / RFMID_ZIP
    records: list[dict[str, Any]] = []
    thumbnails: dict[str, Thumbnail] = {}
    with zipfile.ZipFile(path) as archive:
        label_member_by_split = {
            "training": find_member(archive, "RFMiD_Training_Labels.csv"),
            "validation": find_member(archive, "RFMiD_Validation_Labels.csv"),
            "testing": find_member(archive, "RFMiD_Testing_Labels.csv"),
        }
        labels_by_split: dict[str, dict[str, dict[str, str]]] = {}
        for split, member in label_member_by_split.items():
            labels_by_split[split] = {
                row["ID"]: row for row in read_csv_member(archive, member)
            }

        image_members = [
            name
            for name in archive.namelist()
            if "1. Original Images" in name
            and PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS
        ]
        for member in sorted(image_members):
            split = split_from_member(member)
            image_id = PurePosixPath(member).stem
            label = labels_by_split[split].get(image_id)
            if label is None:
                raise ValueError(f"Missing RFMiD label for {split}/{image_id}")
            width, height, mode, image_format = image_metadata(archive, member)
            record_id = f"rfmid:{split}:{image_id}"
            record = base_record(
                record_id=record_id,
                dataset="RFMiD",
                archive_path=path,
                archive=archive,
                member=member,
                source_split=split,
                source_image_id=image_id,
                width=width,
                height=height,
                mode=mode,
                image_format=image_format,
            )
            disease_risk = label.get("Disease_Risk", "")
            disease_labels = {
                key: int(value)
                for key, value in label.items()
                if key not in {"ID", "Disease_Risk"} and value != ""
            }
            record["disease_risk"] = disease_risk
            record["disease_labels_json"] = json.dumps(
                disease_labels, sort_keys=True, separators=(",", ":")
            )
            if (width, height) == (4288, 2848):
                thumb = thumbnail_and_hash(archive, member, record_id)
                record["perceptual_hash_dhash256"] = thumb.dhash_hex
                thumbnails[record_id] = thumb
            records.append(record)
    if len(records) != 3200:
        raise ValueError(f"Expected 3200 RFMiD records, found {len(records)}")
    return records, thumbnails


def localisation_maps(root: Path) -> tuple[dict[tuple[str, str], tuple[str, str]], dict[tuple[str, str], tuple[str, str]]]:
    path = root / IDRID_LOC_ZIP
    optic_disc: dict[tuple[str, str], tuple[str, str]] = {}
    fovea: dict[tuple[str, str], tuple[str, str]] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".csv"):
                continue
            split = "training" if "Training" in PurePosixPath(member).name else "testing"
            rows = read_csv_member(archive, member)
            target = optic_disc if "OD_Center" in member else fovea
            for row in rows:
                image_id = row.get("Image No", "")
                x = row.get("X- Coordinate", "")
                y = row.get("Y - Coordinate", "")
                if image_id:
                    target[(split, image_id)] = (x, y)
    return optic_disc, fovea


def idrid_grading_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, Thumbnail]]:
    path = root / IDRID_GRADE_ZIP
    optic_disc, fovea = localisation_maps(root)
    records: list[dict[str, Any]] = []
    thumbnails: dict[str, Thumbnail] = {}
    with zipfile.ZipFile(path) as archive:
        grade_members = {
            "training": find_member(archive, "IDRiD_Disease Grading_Training Labels.csv"),
            "testing": find_member(archive, "IDRiD_Disease Grading_Testing Labels.csv"),
        }
        labels_by_split: dict[str, dict[str, dict[str, str]]] = {}
        for split, member in grade_members.items():
            labels_by_split[split] = {
                row["Image name"]: row for row in read_csv_member(archive, member)
            }
        image_members = [
            name
            for name in archive.namelist()
            if "1. Original Images" in name
            and PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS
        ]
        for member in sorted(image_members):
            split = split_from_member(member)
            image_id = PurePosixPath(member).stem
            label = labels_by_split[split].get(image_id)
            if label is None:
                raise ValueError(f"Missing IDRiD grading label for {split}/{image_id}")
            width, height, mode, image_format = image_metadata(archive, member)
            record_id = f"idrid_grading:{split}:{image_id}"
            record = base_record(
                record_id=record_id,
                dataset="IDRiD_Grading_Localisation",
                archive_path=path,
                archive=archive,
                member=member,
                source_split=split,
                source_image_id=image_id,
                width=width,
                height=height,
                mode=mode,
                image_format=image_format,
            )
            record["dr_grade"] = label.get("Retinopathy grade", "")
            record["dme_risk"] = label.get("Risk of macular edema", "")
            od = optic_disc.get((split, image_id), ("", ""))
            fv = fovea.get((split, image_id), ("", ""))
            record["optic_disc_x"], record["optic_disc_y"] = od
            record["fovea_x"], record["fovea_y"] = fv
            thumb = thumbnail_and_hash(archive, member, record_id)
            record["perceptual_hash_dhash256"] = thumb.dhash_hex
            thumbnails[record_id] = thumb
            records.append(record)
    if len(records) != 516:
        raise ValueError(f"Expected 516 IDRiD grading records, found {len(records)}")
    return records, thumbnails


def segmentation_mask_map(archive: zipfile.ZipFile) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    pattern = re.compile(r"(?P<image>IDRiD_\d+)_(?P<kind>MA|HE|EX|SE|OD)\.tif$", re.I)
    for member in archive.namelist():
        match = pattern.search(member)
        if not match:
            continue
        split = split_from_member(member)
        image_id = match.group("image")
        column = MASK_SUFFIX_TO_COLUMN[match.group("kind").upper()]
        result[(split, image_id)][column] = member
    return result


def idrid_segmentation_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, Thumbnail]]:
    path = root / IDRID_SEG_ZIP
    records: list[dict[str, Any]] = []
    thumbnails: dict[str, Thumbnail] = {}
    with zipfile.ZipFile(path) as archive:
        masks = segmentation_mask_map(archive)
        image_members = [
            name
            for name in archive.namelist()
            if "1. Original Images" in name
            and PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS
        ]
        for member in sorted(image_members):
            split = split_from_member(member)
            image_id = PurePosixPath(member).stem
            width, height, mode, image_format = image_metadata(archive, member)
            record_id = f"idrid_segmentation:{split}:{image_id}"
            record = base_record(
                record_id=record_id,
                dataset="IDRiD_Segmentation",
                archive_path=path,
                archive=archive,
                member=member,
                source_split=split,
                source_image_id=image_id,
                width=width,
                height=height,
                mode=mode,
                image_format=image_format,
            )
            record.update(masks.get((split, image_id), {}))
            thumb = thumbnail_and_hash(archive, member, record_id)
            record["perceptual_hash_dhash256"] = thumb.dhash_hex
            thumbnails[record_id] = thumb
            records.append(record)
    if len(records) != 81:
        raise ValueError(f"Expected 81 IDRiD segmentation records, found {len(records)}")
    return records, thumbnails


def detect_overlap(
    records: list[dict[str, Any]],
    rfmid: dict[str, Thumbnail],
    grading: dict[str, Thumbnail],
    segmentation: dict[str, Thumbnail],
) -> list[dict[str, Any]]:
    by_id = {record["record_id"]: record for record in records}
    pairs: list[dict[str, Any]] = []
    union = UnionFind()

    rf_values = list(rfmid.values())
    for grade in grading.values():
        nearest = min(
            rf_values,
            key=lambda candidate: (
                hamming_hex(grade.dhash_hex, candidate.dhash_hex),
                candidate.record_id,
            ),
        )
        distance = hamming_hex(grade.dhash_hex, nearest.dhash_hex)
        if distance > 12:
            continue
        correlation, normalized_mae = similarity(grade.pixels, nearest.pixels)
        status = "confirmed_perceptual_duplicate" if correlation >= 0.995 and normalized_mae <= 0.02 else "candidate_only"
        pairs.append(
            {
                "left_record_id": grade.record_id,
                "right_record_id": nearest.record_id,
                "relation": status,
                "dhash_hamming_256": distance,
                "pixel_correlation": f"{correlation:.8f}",
                "normalized_mae": f"{normalized_mae:.8f}",
            }
        )
        if status == "confirmed_perceptual_duplicate":
            union.union(grade.record_id, nearest.record_id)

    grading_by_number: dict[tuple[str, int], Thumbnail] = {}
    for item in grading.values():
        _, split, image_id = item.record_id.split(":", 2)
        grading_by_number[(split, int(image_id.rsplit("_", 1)[1]))] = item
    for segment in segmentation.values():
        _, split, image_id = segment.record_id.split(":", 2)
        number = int(image_id.rsplit("_", 1)[1])
        candidate = grading_by_number.get((split, number))
        if candidate is None:
            continue
        distance = hamming_hex(segment.dhash_hex, candidate.dhash_hex)
        correlation, normalized_mae = similarity(segment.pixels, candidate.pixels)
        relation = "confirmed_perceptual_duplicate" if correlation >= 0.995 and normalized_mae <= 0.02 else "possible_related_same_source_number"
        pairs.append(
            {
                "left_record_id": segment.record_id,
                "right_record_id": candidate.record_id,
                "relation": relation,
                "dhash_hamming_256": distance,
                "pixel_correlation": f"{correlation:.8f}",
                "normalized_mae": f"{normalized_mae:.8f}",
            }
        )
        if relation == "confirmed_perceptual_duplicate":
            union.union(segment.record_id, candidate.record_id)
        else:
            record = by_id[segment.record_id]
            record["related_record_id"] = candidate.record_id
            record["related_status"] = relation
            record["related_similarity"] = f"corr={correlation:.6f};nmae={normalized_mae:.6f};dhash={distance}"

    groups: dict[str, list[str]] = defaultdict(list)
    for item in union.parent:
        groups[union.find(item)].append(item)
    numbered_groups = {
        member: f"duplicate_group_{index:04d}"
        for index, members in enumerate(
            sorted((sorted(value) for value in groups.values() if len(value) > 1)), start=1
        )
        for member in members
    }
    for record_id, group_id in numbered_groups.items():
        by_id[record_id]["duplicate_group_id"] = group_id
        by_id[record_id]["duplicate_status"] = "confirmed_perceptual_duplicate"
    return pairs


def verify_localisation_image_identity(root: Path) -> tuple[int, int]:
    grade_path = root / IDRID_GRADE_ZIP
    loc_path = root / IDRID_LOC_ZIP
    with zipfile.ZipFile(grade_path) as grade, zipfile.ZipFile(loc_path) as loc:
        grade_images = [name for name in grade.namelist() if PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS]
        loc_names = set(loc.namelist())
        matches = 0
        for member in grade_images:
            relative = member.split("/", 1)[1]
            loc_member = f"C. Localization/{relative}"
            if loc_member in loc_names:
                left = grade.getinfo(member)
                right = loc.getinfo(loc_member)
                if left.CRC == right.CRC and left.file_size == right.file_size:
                    matches += 1
        return matches, len(grade_images)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    required = [RFMID_ZIP, IDRID_SEG_ZIP, IDRID_GRADE_ZIP, IDRID_LOC_ZIP]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required archives: {missing}")

    rfmid_records_list, rfmid_thumbs = rfmid_records(root)
    print(f"Loaded RFMiD: {len(rfmid_records_list)} records", flush=True)
    grading_records_list, grading_thumbs = idrid_grading_records(root)
    print(f"Loaded IDRiD grading/localisation: {len(grading_records_list)} records", flush=True)
    segmentation_records_list, segmentation_thumbs = idrid_segmentation_records(root)
    print(f"Loaded IDRiD segmentation: {len(segmentation_records_list)} records", flush=True)
    records = rfmid_records_list + grading_records_list + segmentation_records_list

    duplicate_pairs: list[dict[str, Any]] = []
    if not args.skip_perceptual_overlap:
        print("Scanning cross-dataset perceptual overlap...", flush=True)
        duplicate_pairs = detect_overlap(
            records, rfmid_thumbs, grading_thumbs, segmentation_thumbs
        )

    records.sort(key=lambda row: row["record_id"])
    fields = list(records[0].keys())
    write_csv(output_dir / "master_manifest.csv", records, fields)
    pair_fields = [
        "left_record_id",
        "right_record_id",
        "relation",
        "dhash_hamming_256",
        "pixel_correlation",
        "normalized_mae",
    ]
    write_csv(output_dir / "duplicate_pairs.csv", duplicate_pairs, pair_fields)

    localisation_identical, localisation_total = verify_localisation_image_identity(root)
    summary = {
        "manifest_schema_version": 1,
        "total_canonical_image_records": len(records),
        "records_by_dataset": dict(Counter(row["dataset"] for row in records)),
        "records_by_resolution": dict(Counter(f'{row["width"]}x{row["height"]}' for row in records)),
        "records_by_camera_model": dict(Counter(row["camera_model"] or "unresolved" for row in records)),
        "records_by_camera_domain": dict(Counter(row["camera_domain"] for row in records)),
        "confirmed_duplicate_groups": len({row["duplicate_group_id"] for row in records if row["duplicate_group_id"]}),
        "confirmed_duplicate_records": sum(bool(row["duplicate_group_id"]) for row in records),
        "candidate_or_related_pairs": len(duplicate_pairs),
        "idrid_localisation_exact_duplicates_of_grading": localisation_identical,
        "idrid_localisation_total_images": localisation_total,
        "camera_identity_warning": (
            "Camera models are derived from official dataset acquisition tables, not embedded per-image metadata. "
            "RFMiD 4288x2848 remains unresolved because the archive has 497 such images versus 467 reported for Kowa, "
            "while 2048x1536 has 276 images versus 306 reported for TRC-NW300."
        ),
        "split_warning": "final_split is intentionally blank until duplicate-aware assignment is generated.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

