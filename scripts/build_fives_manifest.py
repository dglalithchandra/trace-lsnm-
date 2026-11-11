"""Build and validate a manifest for the official FIVES release."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, ImageChops


DISEASE_NAMES = {
    "A": "age_related_macular_degeneration",
    "D": "diabetic_retinopathy",
    "G": "glaucoma",
    "N": "normal",
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_dataset_root(extracted_root: Path) -> Path:
    candidates = [
        path
        for path in extracted_root.iterdir()
        if path.is_dir() and (path / "Quality Assessment.xlsx").is_file()
    ]
    check(len(candidates) == 1, f"Expected one extracted FIVES root, found {candidates}")
    return candidates[0]


def build_manifest(workspace: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    external_root = workspace / "data" / "external" / "FIVES"
    dataset_root = locate_dataset_root(external_root / "extracted")
    workbook = dataset_root / "Quality Assessment.xlsx"

    records: list[dict[str, object]] = []
    for source_split, sheet_name, expected_count in (
        ("train", "Train", 600),
        ("test", "Test", 200),
    ):
        quality = pd.read_excel(workbook, sheet_name=sheet_name)
        check(list(quality.columns) == ["Disease", "Number", "IC", "Blur", "LC"],
              f"Unexpected {sheet_name} workbook columns: {list(quality.columns)}")
        check(len(quality) == expected_count, f"Unexpected {source_split} workbook count")

        original_dir = dataset_root / source_split / "Original"
        mask_dir = dataset_root / source_split / "Ground truth"
        originals = {path.name: path for path in original_dir.glob("*.png")}
        masks = {path.name: path for path in mask_dir.glob("*.png")}
        check(len(originals) == expected_count, f"Unexpected {source_split} image count")
        check(set(originals) == set(masks), f"Image/mask filename mismatch in {source_split}")

        expected_filenames: set[str] = set()
        for row in quality.itertuples(index=False):
            disease_code = str(row.Disease).strip()
            check(disease_code in DISEASE_NAMES, f"Unexpected disease code: {disease_code}")
            number = int(row.Number)
            filename = f"{number}_{disease_code}.png"
            expected_filenames.add(filename)
            check(filename in originals, f"Workbook image is missing: {source_split}/{filename}")

            image_path = originals[filename]
            mask_path = masks[filename]
            with Image.open(image_path) as image:
                width, height = image.size
                mode = image.mode
                image_format = image.format
                dhash = str(imagehash.dhash(image.convert("RGB"), hash_size=16))
            with Image.open(mask_path) as mask:
                mask_width, mask_height = mask.size
                mask_mode = mask.mode
                red, green, blue = mask.convert("RGB").split()
                red_histogram = red.histogram()

            check((width, height) == (2048, 2048), f"Unexpected image size: {filename}")
            check(mode == "RGB", f"Unexpected original mode: {filename} -> {mode}")
            check((mask_width, mask_height) == (2048, 2048), f"Unexpected mask size: {filename}")
            check(mask_mode in {"RGB", "RGBA"}, f"Unexpected mask mode: {filename} -> {mask_mode}")
            check(sum(red_histogram[1:255]) == 0,
                  f"Mask has non-binary values: {filename}")
            check(ImageChops.difference(red, green).getbbox() is None
                  and ImageChops.difference(red, blue).getbbox() is None,
                  f"Mask RGB channels differ: {filename}")
            mask_foreground_pixels = int(red_histogram[255])
            mask_is_nonempty = mask_foreground_pixels > 0

            for value, name in ((row.IC, "IC"), (row.Blur, "Blur"), (row.LC, "LC")):
                check(int(value) in {0, 1}, f"Unexpected {name} value for {filename}: {value}")

            records.append(
                {
                    "record_id": f"fives:{source_split}:{Path(filename).stem}",
                    "dataset": "FIVES",
                    "modality": "color_fundus_photograph",
                    "source_split": source_split,
                    "source_image_id": Path(filename).stem,
                    "patient_id": "",
                    "patient_id_status": "not_provided_in_public_release",
                    "camera_model": "",
                    "camera_metadata_status": "not_provided_in_public_release",
                    "image_path": str(image_path.relative_to(workspace)).replace("\\", "/"),
                    "vessel_mask_path": str(mask_path.relative_to(workspace)).replace("\\", "/"),
                    "width": width,
                    "height": height,
                    "channels": 3,
                    "image_mode": mode,
                    "image_format": image_format,
                    "image_bytes": image_path.stat().st_size,
                    "image_sha256": sha256_file(image_path),
                    "mask_mode": mask_mode,
                    "mask_bytes": mask_path.stat().st_size,
                    "mask_sha256": sha256_file(mask_path),
                    "mask_foreground_pixels": mask_foreground_pixels,
                    "mask_is_nonempty": mask_is_nonempty,
                    "mask_validation_status": (
                        "valid_binary_nonempty"
                        if mask_is_nonempty
                        else "empty_mask_exclude_from_vessel_metrics"
                    ),
                    "disease_code": disease_code,
                    "disease_label": DISEASE_NAMES[disease_code],
                    "quality_illumination_colour_good": int(row.IC),
                    "quality_blur_good": int(row.Blur),
                    "quality_low_contrast_good": int(row.LC),
                    "quality_flag_interpretation": "1=good;0=poor",
                    "quality_label_source": "published_automatic_quality_algorithms",
                    "perceptual_hash_dhash256": dhash,
                    "license": "CC-BY-4.0",
                    "source_doi": "10.6084/m9.figshare.19688169.v1",
                    "final_split": "",
                    "final_split_status": "source_split_only_patient_ids_unavailable",
                }
            )

        check(expected_filenames == set(originals),
              f"Files not represented in the {sheet_name} workbook")

    manifest = pd.DataFrame(records).sort_values("record_id").reset_index(drop=True)
    check(len(manifest) == 800, f"Expected 800 records, found {len(manifest)}")
    check(manifest["record_id"].is_unique, "FIVES record IDs are not unique")
    check(Counter(manifest["source_split"]) == {"train": 600, "test": 200},
          "Unexpected source split counts")
    check(Counter(manifest["disease_code"]) == {"A": 200, "D": 200, "G": 200, "N": 200},
          "Unexpected disease counts")

    manifest["duplicate_group_id"] = ""
    manifest["duplicate_status"] = "none_detected"
    manifest["is_primary_analysis_record"] = True
    exact_groups = [
        group.sort_values("record_id")
        for _, group in manifest.groupby("image_sha256")
        if len(group) > 1
    ]
    for group_number, group in enumerate(exact_groups, start=1):
        group_id = f"fives_exact_duplicate_{group_number:04d}"
        manifest.loc[group.index, "duplicate_group_id"] = group_id
        manifest.loc[group.index, "duplicate_status"] = "confirmed_exact_file_duplicate"
        manifest.loc[group.index[1:], "is_primary_analysis_record"] = False

    exact_duplicate_records = sum(len(group) for group in exact_groups)
    duplicate_groups_with_different_masks = sum(
        group["mask_sha256"].nunique() > 1 for group in exact_groups
    )
    check(len(exact_groups) == 6, f"Expected six exact duplicate groups, found {len(exact_groups)}")
    check(exact_duplicate_records == 12, "Expected 12 records in exact duplicate groups")
    check(duplicate_groups_with_different_masks == 6,
          "Each known exact-image duplicate pair should have different mask files")

    quality_counts = {
        column: {
            "good": int((manifest[column] == 1).sum()),
            "poor": int((manifest[column] == 0).sum()),
        }
        for column in (
            "quality_illumination_colour_good",
            "quality_blur_good",
            "quality_low_contrast_good",
        )
    }

    summary: dict[str, object] = {
        "status": "PASS",
        "records": len(manifest),
        "source_splits": manifest["source_split"].value_counts().sort_index().to_dict(),
        "disease_counts": manifest["disease_label"].value_counts().sort_index().to_dict(),
        "quality_counts": quality_counts,
        "image_mask_pairs": 800,
        "usable_nonempty_vessel_masks": int(manifest["mask_is_nonempty"].sum()),
        "empty_vessel_masks": int((~manifest["mask_is_nonempty"]).sum()),
        "empty_vessel_mask_records": manifest.loc[
            ~manifest["mask_is_nonempty"], "record_id"
        ].tolist(),
        "image_size": "2048x2048",
        "mask_pixel_values": [0, 255],
        "unique_photographs_by_sha256": int(manifest["image_sha256"].nunique()),
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_records": exact_duplicate_records,
        "duplicate_groups_with_different_masks": duplicate_groups_with_different_masks,
        "primary_analysis_records_after_exact_deduplication": int(
            manifest["is_primary_analysis_record"].sum()
        ),
        "patient_id_warning": "Patient IDs are not present in the public release.",
        "camera_warning": "Per-image camera identity is not present in the public release.",
        "split_warning": "Published train/test split retained as source metadata only.",
    }
    return manifest, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    manifest, summary = build_manifest(workspace)
    output_dir = workspace / "data" / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "fives_manifest.csv", index=False)
    (output_dir / "fives_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
