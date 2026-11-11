"""Run a small, non-test PALM integration smoke test and visual review sheet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_preprocessing import PreprocessingConfig, TracePreprocessor
from trace_preprocessing.loader import ManifestImageLoader


MANIFEST = WORKSPACE / "data" / "manifests" / "palm_manifest.csv"
OUTPUT = WORKSPACE / "outputs" / "palm_preprocessing_smoke"


def fit(image: np.ndarray, size: int = 300) -> Image.Image:
    source = Image.fromarray(image).convert("RGB")
    source.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "black")
    canvas.paste(source, ((size - source.width) // 2, (size - source.height) // 2))
    return canvas


def choose_records(loader: ManifestImageLoader) -> list[str]:
    frame = loader.manifest
    frame = frame[
        frame["_effective_split"].isin(["train", "validation"])
        & frame["is_primary_analysis_record"].eq("True")
    ]
    choices: list[str] = []
    selectors = [
        ("train", "Canon CR-2", "disc_mask"),
        ("train", "ZEISS Visucam 500", "atrophy_mask"),
        ("validation", "Canon CR-2", "disc_mask"),
        ("validation", "ZEISS Visucam 500", "detachment_mask"),
    ]
    for split, camera, annotation_column in selectors:
        candidates = frame[
            frame["_effective_split"].eq(split)
            & frame["camera_model"].eq(camera)
            & frame[annotation_column].ne("")
        ]
        if candidates.empty:
            raise RuntimeError(f"No PALM smoke candidate: {split}/{camera}/{annotation_column}")
        choices.append(str(candidates.iloc[0]["record_id"]))
    return choices


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pipeline = TracePreprocessor(PreprocessingConfig(apply_photometric_correction=False))
    expected_output_shape = [
        pipeline.config.geometry.output_size,
        pipeline.config.geometry.output_size,
        3,
    ]
    records: list[dict[str, object]] = []
    images: list[tuple[np.ndarray, np.ndarray, str]] = []
    with ManifestImageLoader(WORKSPACE, MANIFEST) as loader:
        record_ids = choose_records(loader)
        for record_id in record_ids:
            image, row = loader.load_image(record_id)
            if row["_effective_split"] == "test":
                raise RuntimeError(f"Protected PALM test record reached smoke test: {record_id}")
            masks = loader.load_annotation_masks(record_id)
            result = pipeline.process(
                image,
                image_id=record_id,
                annotation_masks=masks,
                metadata={
                    "dataset": "PALM",
                    "effective_split": row["_effective_split"],
                    "camera_model": row["camera_model"],
                },
            )
            if result.geometry_only_image is None:
                raise RuntimeError(f"PALM geometry failed: {record_id}")
            preservation = result.annotation_preservation
            lost = [
                name
                for name, item in preservation.items()
                if item.source_foreground_pixels > 0
                and item.transformed_foreground_pixels == 0
            ]
            records.append(
                {
                    "record_id": record_id,
                    "effective_split": row["_effective_split"],
                    "camera_model": row["camera_model"],
                    "source_shape": list(image.shape),
                    "output_shape": list(result.geometry_only_image.shape),
                    "status": result.status.value,
                    "warnings": result.warnings,
                    "annotation_names": sorted(masks),
                    "nonempty_annotations_lost": lost,
                    "fundus_detection_confidence": result.metadata.get(
                        "fundus_detection_confidence"
                    ),
                }
            )
            images.append(
                (
                    image,
                    result.geometry_only_image,
                    f"{record_id} | {row['camera_model']} | {row['_effective_split']}",
                )
            )

    panel = 300
    label_height = 46
    sheet = Image.new("RGB", (panel * 2, (panel + label_height) * len(images)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (original, geometry, label) in enumerate(images):
        top = index * (panel + label_height)
        sheet.paste(fit(original, panel), (0, top))
        sheet.paste(fit(geometry, panel), (panel, top))
        draw.text((6, top + panel + 5), label, fill="black", font=font)
        draw.text((6, top + panel + 22), "original", fill="black", font=font)
        draw.text((panel + 6, top + panel + 22), "geometry normalized", fill="black", font=font)
    sheet.save(OUTPUT / "palm_smoke_review.png")

    summary = {
        "status": "PASS",
        "records": records,
        "records_processed": len(records),
        "test_records_accessed": 0,
        "photometric_correction_enabled": False,
        "expected_output_shape": expected_output_shape,
        "all_outputs_expected_rgb": all(
            row["output_shape"] == expected_output_shape for row in records
        ),
        "total_nonempty_annotations_lost": sum(
            len(row["nonempty_annotations_lost"]) for row in records
        ),
        "purpose": "integration smoke test only; not a performance result",
    }
    if not summary["all_outputs_expected_rgb"]:
        raise RuntimeError("PALM smoke output size failure")
    if summary["total_nonempty_annotations_lost"]:
        raise RuntimeError("PALM smoke lost an annotation")
    (OUTPUT / "palm_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
