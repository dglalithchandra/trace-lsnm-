"""Audit TRACE geometry, quality, and photometry on PALM development records.

Only duplicate-controlled primary train/validation records are permitted. The
protected test split is never requested. Quality thresholds remain provisional;
the audit measures and surfaces cases for review rather than claiming clinical
acceptability.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_preprocessing import PreprocessingConfig, TracePreprocessor
from trace_preprocessing.loader import ManifestImageLoader


MANIFEST = WORKSPACE / "data" / "manifests" / "palm_manifest.csv"
OUTPUT = WORKSPACE / "outputs" / "palm_preprocessing_audit"
METRICS = [
    "fundus_detection_confidence",
    "source_retinal_coverage",
    "common_mask_retinal_loss_fraction",
    "normalized_detected_retina_fill_fraction",
    "blur_laplacian_variance",
    "noise_mad",
    "glare_fraction",
    "compression_blockiness",
    "red_green_ratio",
    "blue_green_ratio",
    "saturation_median",
    "brightness_median",
    "contrast_p95_p05",
    "gamma_to_midgray",
    "illumination_nonuniformity",
]


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def permitted_record_ids(loader: ManifestImageLoader) -> list[str]:
    frame = loader.manifest
    primary = frame.get(
        "is_primary_analysis_record", pd.Series("", index=frame.index, dtype=str)
    ).astype(str).str.lower().eq("true")
    allowed = frame[frame["_effective_split"].isin(["train", "validation"]) & primary]
    if allowed["_effective_split"].eq("test").any():
        raise RuntimeError("protected_test_record_entered_palm_audit_plan")
    return allowed["record_id"].tolist()


def audit_record(
    loader: ManifestImageLoader,
    pipeline: TracePreprocessor,
    record_id: str,
) -> dict[str, Any]:
    image, source = loader.load_image(record_id)
    if source["_effective_split"] == "test":
        raise RuntimeError(f"protected_test_record_loaded:{record_id}")
    masks = loader.load_annotation_masks(record_id)
    result = pipeline.process(
        image,
        image_id=record_id,
        annotation_masks=masks,
        metadata={
            "dataset": "PALM",
            "effective_split": source["_effective_split"],
            "camera_model": source.get("camera_model", ""),
        },
    )
    metadata = result.metadata
    quality = result.quality_before
    stats = result.stats_before
    preservation = result.annotation_preservation
    lost = [
        name
        for name, item in preservation.items()
        if item.source_foreground_pixels > 0
        and item.transformed_foreground_pixels == 0
    ]
    row: dict[str, Any] = {
        "record_id": record_id,
        "effective_split": source["_effective_split"],
        "camera_model": source.get("camera_model", "not_provided"),
        "pathologic_myopia": source.get("pathologic_myopia", ""),
        "source_width": image.shape[1],
        "source_height": image.shape[0],
        "output_width": (
            result.geometry_only_image.shape[1]
            if result.geometry_only_image is not None
            else ""
        ),
        "output_height": (
            result.geometry_only_image.shape[0]
            if result.geometry_only_image is not None
            else ""
        ),
        "status": result.status.value,
        "reasons": "|".join(result.reasons),
        "warnings": "|".join(result.warnings),
        "fundus_detection_confidence": safe_float(
            metadata.get("fundus_detection_confidence")
        ),
        "source_retinal_coverage": safe_float(metadata.get("source_retinal_coverage")),
        "source_crop_retinal_loss_fraction": safe_float(
            metadata.get("source_crop_retinal_loss_fraction")
        ),
        "common_mask_retinal_loss_fraction": safe_float(
            metadata.get("common_mask_retinal_loss_fraction")
        ),
        "normalized_detected_retina_fill_fraction": safe_float(
            metadata.get("normalized_detected_retina_fill_fraction")
        ),
        "annotation_masks_present": len(masks),
        "nonempty_annotations_lost": len(lost),
        "lost_annotation_names": "|".join(lost),
    }
    if quality is not None:
        row.update(quality.to_dict())
    if stats is not None:
        row.update(stats.to_dict())
    return row


def fit_image(array: np.ndarray, size: int = 280) -> Image.Image:
    image = Image.fromarray(array).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "black")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def review_sheet(
    selection: pd.DataFrame,
    loader: ManifestImageLoader,
    output: Path,
    title: str,
) -> None:
    selection = selection.head(8)
    if selection.empty:
        return
    panel = 280
    label_height = 52
    sheet = Image.new(
        "RGB", (panel * 2, 40 + (panel + label_height) * len(selection)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 10), title, fill="black", font=font)
    pipeline = TracePreprocessor(PreprocessingConfig(apply_photometric_correction=False))
    for index, row in enumerate(selection.itertuples(index=False)):
        image, source = loader.load_image(row.record_id)
        if source["_effective_split"] == "test":
            raise RuntimeError("protected_test_record_entered_review_sheet")
        result = pipeline.process(image, image_id=row.record_id)
        if result.geometry_only_image is None:
            continue
        top = 40 + index * (panel + label_height)
        sheet.paste(fit_image(image, panel), (0, top))
        sheet.paste(fit_image(result.geometry_only_image, panel), (panel, top))
        label = (
            f"{row.record_id} | {row.camera_model} | {row.effective_split} | "
            f"status={row.status}"
        )
        draw.text((6, top + panel + 5), label[:95], fill="black", font=font)
        draw.text((6, top + panel + 22), "original", fill="black", font=font)
        draw.text((panel + 6, top + panel + 22), "geometry normalized", fill="black", font=font)
    sheet.save(output)


def metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for camera, group in frame.groupby("camera_model", dropna=False):
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "camera_model": camera,
                    "metric": metric,
                    "count": len(values),
                    "mean": float(values.mean()),
                    "p05": float(values.quantile(0.05)),
                    "median": float(values.median()),
                    "p95": float(values.quantile(0.95)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pipeline = TracePreprocessor(PreprocessingConfig(apply_photometric_correction=False))
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ManifestImageLoader(WORKSPACE, MANIFEST, allow_test=False) as loader:
        record_ids = permitted_record_ids(loader)
        for index, record_id in enumerate(record_ids, start=1):
            try:
                records.append(audit_record(loader, pipeline, record_id))
            except Exception as error:
                errors.append(
                    {
                        "record_id": record_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            if index % 50 == 0 or index == len(record_ids):
                print(
                    f"palm_audit_progress={index}/{len(record_ids)} "
                    f"completed={len(records)} errors={len(errors)}",
                    flush=True,
                )

        frame = pd.DataFrame(records)
        error_frame = pd.DataFrame(errors, columns=["record_id", "error_type", "error"])
        frame.to_csv(OUTPUT / "palm_preprocessing_records.csv", index=False)
        error_frame.to_csv(OUTPUT / "palm_preprocessing_errors.csv", index=False)
        metric_summary(frame).to_csv(OUTPUT / "palm_camera_metric_summary.csv", index=False)

        if not frame.empty:
            review_sheet(
                frame.sort_values("fundus_detection_confidence"),
                loader,
                OUTPUT / "review_lowest_detection_confidence.png",
                "Lowest fundus-detection confidence",
            )
            review_sheet(
                frame.sort_values("blur_laplacian_variance"),
                loader,
                OUTPUT / "review_lowest_blur_score.png",
                "Lowest blur score (softest images)",
            )
            review_sheet(
                frame.sort_values("glare_fraction", ascending=False),
                loader,
                OUTPUT / "review_highest_glare.png",
                "Highest measured glare fraction",
            )
            review_sheet(
                frame.sort_values("common_mask_retinal_loss_fraction", ascending=False),
                loader,
                OUTPUT / "review_highest_common_mask_loss.png",
                "Highest common-mask retinal loss",
            )

    warning_counts: Counter[str] = Counter()
    for value in frame.get("warnings", pd.Series(dtype=str)).fillna(""):
        warning_counts.update(item for item in value.split("|") if item)
    expected_size = pipeline.config.geometry.output_size
    summary = {
        "status": "PASS" if not errors else "FAIL_WITH_PROCESSING_ERRORS",
        "scope": "PALM duplicate-controlled primary train/validation records only",
        "planned_records": len(record_ids),
        "completed_records": len(frame),
        "processing_errors": len(errors),
        "test_records_accessed": 0,
        "records_by_split": {
            str(key): int(value)
            for key, value in frame["effective_split"].value_counts().items()
        },
        "records_by_camera": {
            str(key): int(value)
            for key, value in frame["camera_model"].value_counts().items()
        },
        "pipeline_status_counts": {
            str(key): int(value) for key, value in frame["status"].value_counts().items()
        },
        "warning_counts": dict(sorted(warning_counts.items())),
        "output_size": expected_size,
        "output_size_failures": int(
            (
                (pd.to_numeric(frame["output_width"], errors="coerce") != expected_size)
                | (pd.to_numeric(frame["output_height"], errors="coerce") != expected_size)
            ).sum()
        ),
        "annotation_masks_evaluated": int(frame["annotation_masks_present"].sum()),
        "nonempty_annotations_lost": int(frame["nonempty_annotations_lost"].sum()),
        "photometric_correction_enabled": False,
        "quality_thresholds_are_provisional": True,
        "important_limit": (
            "Quality warnings and cross-camera metric differences are audit signals, "
            "not clinically validated acceptance thresholds or proof of generalization."
        ),
    }
    (OUTPUT / "palm_preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
