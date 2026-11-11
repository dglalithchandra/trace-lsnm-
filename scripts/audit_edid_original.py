"""Audit the official EDID original archive without extracting or splitting images."""

from __future__ import annotations

import hashlib
import csv
import json
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from itertools import combinations
from pathlib import Path, PurePosixPath

from PIL import ExifTags, Image

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/external/EDID/Original_Dataset.zip"
EXPECTED_SHA256 = "4cc006a9a769cb5172d552b69395d65fa1f1794af27baf32574a027051ea188d"
OUTPUT = ROOT / "data/manifests/edid_original_audit.json"
RECORDS = ROOT / "data/manifests/edid_original_records.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if file_sha256(ARCHIVE) != EXPECTED_SHA256:
        raise RuntimeError("EDID archive does not match Mendeley SHA-256")
    class_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    exif_counts: Counter[str] = Counter()
    content_hashes: dict[str, list[str]] = defaultdict(list)
    broken: list[str] = []
    image_names: list[str] = []
    record_rows: list[dict[str, str]] = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for number, member in enumerate(archive.infolist(), start=1):
            if member.is_dir():
                continue
            name = PurePosixPath(member.filename)
            if name.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image_names.append(member.filename)
            class_counts[name.parent.name] += 1
            suffix_counts[name.suffix.lower()] += 1
            try:
                content = archive.read(member)
                digest = hashlib.sha256(content).hexdigest()
                content_hashes[digest].append(member.filename)
                record_rows.append({"archive_member": member.filename, "class_folder": name.parent.name, "sha256": digest})
                with Image.open(BytesIO(content)) as image:
                    size_counts[f"{image.width}x{image.height}"] += 1
                    exif = image.getexif()
                    for tag in exif:
                        exif_counts[ExifTags.TAGS.get(tag, str(tag))] += 1
                    image.verify()
            except Exception as error:
                broken.append(f"{member.filename}: {type(error).__name__}: {error}")
            if number % 1000 == 0:
                print(f"Audited {number} ZIP members", flush=True)
    duplicate_groups = [names for names in content_hashes.values() if len(names) > 1]
    cross_class_groups = [
        names for names in duplicate_groups
        if len({PurePosixPath(name).parent.name for name in names}) > 1
    ]
    healthy_conflict_groups = [
        names for names in cross_class_groups
        if "Healthy" in {PurePosixPath(name).parent.name for name in names}
    ]
    class_pair_counts: Counter[str] = Counter()
    for names in cross_class_groups:
        categories = sorted({PurePosixPath(name).parent.name for name in names})
        for left, right in combinations(categories, 2):
            class_pair_counts[f"{left} || {right}"] += 1
    unambiguous_unique_counts: Counter[str] = Counter()
    for names in content_hashes.values():
        categories = {PurePosixPath(name).parent.name for name in names}
        if len(categories) == 1:
            unambiguous_unique_counts[next(iter(categories))] += 1
    summary = {
        "status": "PASS" if not broken else "FAILED_IMAGE_READS",
        "source_archive": str(ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": EXPECTED_SHA256,
        "image_members": len(image_names),
        "class_counts": dict(sorted(class_counts.items())),
        "dimension_counts": dict(size_counts.most_common()),
        "extension_counts": dict(sorted(suffix_counts.items())),
        "exif_tag_counts": dict(sorted(exif_counts.items())),
        "exact_duplicate_groups": len(duplicate_groups),
        "exact_duplicate_extra_images": sum(len(group) - 1 for group in duplicate_groups),
        "cross_class_exact_duplicate_groups": len(cross_class_groups),
        "healthy_cross_class_exact_duplicate_groups": len(healthy_conflict_groups),
        "cross_class_pairs": dict(class_pair_counts.most_common()),
        "unique_images_with_one_class_folder": dict(sorted(unambiguous_unique_counts.items())),
        "all_unique_bytewise_images": len(content_hashes),
        "exact_duplicate_examples": duplicate_groups[:20],
        "broken_image_count": len(broken),
        "broken_image_examples": broken[:20],
        "sample_names_by_class": {
            category: sorted(name for name in image_names if PurePosixPath(name).parent.name == category)[:3]
            for category in sorted(class_counts)
        },
        "patient_ids_verified": False,
        "per_image_camera_ids_verified": False,
        "augmented_archive_opened": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with RECORDS.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["archive_member", "class_folder", "sha256"])
        writer.writeheader()
        writer.writerows(record_rows)
    OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {
        "sample_names_by_class", "exact_duplicate_examples", "broken_image_examples"
    }}, indent=2))


if __name__ == "__main__":
    main()
