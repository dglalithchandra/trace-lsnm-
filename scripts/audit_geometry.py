"""Audit TRACE geometric normalisation across all permitted development data.

The final test split is never requested.  The script saves compact metrics and
small review sheets rather than materialising every processed photograph.
"""

from __future__ import annotations

import argparse
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

from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.loader import ManifestImageLoader


REVIEW_LIMITS = {
    "detection_confidence_below": 0.75,
    "source_crop_loss_above": 0.001,
    "common_mask_loss_above": 0.05,
    "crop_padding_above": 0.15,
    "normalized_fill_below": 0.80,
}


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def audit_one(
    loader: ManifestImageLoader,
    pipeline: TracePreprocessor,
    record_id: str,
) -> dict[str, Any]:
    image, source = loader.load_image(record_id)
    annotation_masks = loader.load_annotation_masks(record_id)
    effective_split = source.get("_effective_split", "")
    if effective_split == "test":
        raise RuntimeError(f"test_record_reached_geometry_audit:{record_id}")
    result = pipeline.process(
        image,
        image_id=record_id,
        annotation_masks=annotation_masks,
        metadata={
            "dataset": source.get("dataset", ""),
            "effective_split": effective_split,
            "camera_domain": source.get("camera_domain", ""),
        },
    )

    metadata = result.metadata
    transform = result.geometry_transform
    preservation = result.annotation_preservation
    nonempty_source_annotations = sum(
        item.source_foreground_pixels > 0 for item in preservation.values()
    )
    empty_transformed_annotations = sum(
        item.source_foreground_pixels > 0 and item.transformed_foreground_pixels == 0
        for item in preservation.values()
    )
    max_annotation_crop_loss = max(
        (item.source_region_clipped_fraction for item in preservation.values()),
        default=0.0,
    )
    review_flags: list[str] = []
    informational_flags: list[str] = []
    confidence = _safe_float(metadata.get("fundus_detection_confidence"))
    source_crop_loss = _safe_float(metadata.get("source_crop_retinal_loss_fraction"))
    common_mask_loss = _safe_float(metadata.get("common_mask_retinal_loss_fraction"))
    crop_padding = _safe_float(metadata.get("crop_padding_fraction"))
    normalized_fill = _safe_float(metadata.get("normalized_detected_retina_fill_fraction"))
    if result.status == ProcessingStatus.REJECT:
        review_flags.append("pipeline_rejected")
    if np.isfinite(confidence) and confidence < REVIEW_LIMITS["detection_confidence_below"]:
        review_flags.append("low_detection_confidence")
    if np.isfinite(source_crop_loss) and source_crop_loss > REVIEW_LIMITS["source_crop_loss_above"]:
        review_flags.append("source_retina_outside_crop")
    if np.isfinite(common_mask_loss) and common_mask_loss > REVIEW_LIMITS["common_mask_loss_above"]:
        review_flags.append("common_mask_removes_detected_retina")
    if np.isfinite(crop_padding) and crop_padding > REVIEW_LIMITS["crop_padding_above"]:
        informational_flags.append("large_crop_padding_preserves_square_field")
    if np.isfinite(normalized_fill) and normalized_fill < REVIEW_LIMITS["normalized_fill_below"]:
        review_flags.append("low_normalized_retinal_fill")
    if empty_transformed_annotations:
        review_flags.append("nonempty_annotation_lost")
    if max_annotation_crop_loss > REVIEW_LIMITS["source_crop_loss_above"]:
        review_flags.append("annotation_outside_crop")

    row = {
        "record_id": record_id,
        "dataset": source.get("dataset", ""),
        "effective_split": effective_split,
        "source_split": source.get("source_split", ""),
        "camera_domain": source.get("camera_domain", "") or "not_provided",
        "camera_domain_source": source.get("camera_domain_source", "") or "not_provided",
        "source_width": image.shape[1],
        "source_height": image.shape[0],
        "status": result.status.value,
        "reasons": "|".join(result.reasons),
        "warnings": "|".join(result.warnings),
        "review_flags": "|".join(sorted(set(review_flags))),
        "informational_flags": "|".join(sorted(set(informational_flags))),
        "needs_visual_review": bool(review_flags),
        "fundus_detection_confidence": confidence,
        "source_retinal_coverage": _safe_float(metadata.get("source_retinal_coverage")),
        "fundus_bbox_width": _safe_float(metadata.get("fundus_bbox_width")),
        "fundus_bbox_height": _safe_float(metadata.get("fundus_bbox_height")),
        "fundus_radius": _safe_float(metadata.get("fundus_radius")),
        "crop_x": transform.crop_x if transform else "",
        "crop_y": transform.crop_y if transform else "",
        "crop_size": transform.crop_size if transform else "",
        "crop_padding_fraction": crop_padding,
        "source_crop_retinal_loss_fraction": source_crop_loss,
        "common_mask_retinal_loss_fraction": common_mask_loss,
        "normalized_detected_retina_fill_fraction": normalized_fill,
        "output_width": result.geometry_only_image.shape[1] if result.geometry_only_image is not None else "",
        "output_height": result.geometry_only_image.shape[0] if result.geometry_only_image is not None else "",
        "annotation_masks_present": len(annotation_masks),
        "nonempty_source_annotations": nonempty_source_annotations,
        "nonempty_annotations_lost": empty_transformed_annotations,
        "maximum_annotation_crop_loss_fraction": max_annotation_crop_loss,
        "analysis_primary": source.get("is_primary_analysis_record", ""),
        "duplicate_status": source.get("duplicate_status", ""),
    }
    return row


def _fit_image(array: np.ndarray, size: int = 280) -> Image.Image:
    image = Image.fromarray(array).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def make_review_sheet(
    selections: pd.DataFrame,
    loaders: dict[str, ManifestImageLoader],
    output: Path,
    title: str,
) -> None:
    if selections.empty:
        return
    selections = selections.head(8)
    panel = 280
    label_height = 54
    row_height = panel + label_height
    sheet = Image.new("RGB", (panel * 2, 46 + row_height * len(selections)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 10), title, fill=(15, 15, 15), font=font)
    for row_index, row in enumerate(selections.itertuples(index=False)):
        loader = loaders[row.loader_key]
        image, _ = loader.load_image(row.record_id)
        result = TracePreprocessor(
            PreprocessingConfig(apply_photometric_correction=False)
        ).process(image, image_id=row.record_id)
        if result.geometry_only_image is None:
            continue
        top = 46 + row_index * row_height
        sheet.paste(_fit_image(image, panel), (0, top))
        sheet.paste(_fit_image(result.geometry_only_image, panel), (panel, top))
        label = (
            f"{row.record_id} | {row.dataset} | conf={row.fundus_detection_confidence:.3f} "
            f"| mask_loss={row.common_mask_retinal_loss_fraction:.3f}"
        )
        draw.text((6, top + panel + 5), label[:92], fill=(15, 15, 15), font=font)
        draw.text((6, top + panel + 22), "original", fill=(60, 60, 60), font=font)
        draw.text((panel + 6, top + panel + 22), "geometry output", fill=(60, 60, 60), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output_dir = workspace / "outputs" / "geometry_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_loader = ManifestImageLoader(workspace, allow_test=False)
    fives_loader = ManifestImageLoader(
        workspace,
        workspace / "data" / "manifests" / "fives_manifest.csv",
        allow_test=False,
    )
    loaders = {"trace": trace_loader, "fives": fives_loader}
    planned: list[tuple[str, str]] = []
    for split in ("train", "validation"):
        planned.extend(("trace", record_id) for record_id in trace_loader.record_ids(split=split))
    planned.extend(("fives", record_id) for record_id in fives_loader.record_ids(split="train"))

    pipeline = TracePreprocessor(PreprocessingConfig(apply_photometric_correction=False))
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, (loader_key, record_id) in enumerate(planned, start=1):
        try:
            row = audit_one(loaders[loader_key], pipeline, record_id)
            row["loader_key"] = loader_key
            records.append(row)
        except Exception as error:  # continue to produce a complete failure inventory
            errors.append(
                {
                    "loader_key": loader_key,
                    "record_id": record_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        if index % args.progress_every == 0 or index == len(planned):
            print(
                f"geometry_audit_progress={index}/{len(planned)} "
                f"completed={len(records)} errors={len(errors)}",
                flush=True,
            )

    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "geometry_audit_records.csv", index=False)
    pd.DataFrame(errors).to_csv(output_dir / "geometry_audit_errors.csv", index=False)

    if not frame.empty:
        make_review_sheet(
            frame.sort_values("fundus_detection_confidence", ascending=True),
            loaders,
            output_dir / "review_lowest_detection_confidence.png",
            "Lowest fundus-detection confidence: original vs geometry output",
        )
        make_review_sheet(
            frame.sort_values("common_mask_retinal_loss_fraction", ascending=False),
            loaders,
            output_dir / "review_highest_common_mask_loss.png",
            "Highest common-mask loss: original vs geometry output",
        )
        make_review_sheet(
            frame[frame["needs_visual_review"]].sort_values(
                ["dataset", "fundus_detection_confidence"]
            ),
            loaders,
            output_dir / "review_flagged_cases.png",
            "Automatically flagged geometry cases: original vs geometry output",
        )
        make_review_sheet(
            frame.sort_values("normalized_detected_retina_fill_fraction", ascending=True),
            loaders,
            output_dir / "review_lowest_normalized_retinal_fill.png",
            "Lowest normalized retinal fill: original vs geometry output",
        )

    flag_counts: Counter[str] = Counter()
    for value in frame.get("review_flags", pd.Series(dtype=str)).fillna(""):
        flag_counts.update(item for item in value.split("|") if item)
    informational_counts: Counter[str] = Counter()
    for value in frame.get("informational_flags", pd.Series(dtype=str)).fillna(""):
        informational_counts.update(item for item in value.split("|") if item)
    fives_rows = frame[frame["dataset"] == "FIVES"] if not frame.empty else frame
    summary: dict[str, Any] = {
        "status": "PASS" if not errors else "FAIL_WITH_PROCESSING_ERRORS",
        "scope": "all_permitted_development_geometry_records",
        "planned_records": len(planned),
        "completed_records": len(frame),
        "processing_errors": len(errors),
        "test_records_accessed": 0,
        "records_by_dataset": _counts(frame, "dataset") if not frame.empty else {},
        "records_by_effective_split": _counts(frame, "effective_split") if not frame.empty else {},
        "pipeline_status_counts": _counts(frame, "status") if not frame.empty else {},
        "automatic_review_flag_counts": dict(sorted(flag_counts.items())),
        "informational_geometry_counts": dict(sorted(informational_counts.items())),
        "records_needing_visual_review": int(frame["needs_visual_review"].sum()) if not frame.empty else 0,
        "output_size_failures": int(
            ((frame["output_width"] != 448) | (frame["output_height"] != 448)).sum()
        ) if not frame.empty else 0,
        "nonempty_annotations_lost": int(frame["nonempty_annotations_lost"].sum()) if not frame.empty else 0,
        "annotation_masks_evaluated": int(frame["annotation_masks_present"].sum()) if not frame.empty else 0,
        "fives_released_rows_audited": len(fives_rows),
        "fives_primary_analysis_rows": int(
            fives_rows["analysis_primary"].astype(str).str.lower().eq("true").sum()
        ) if not fives_rows.empty else 0,
        "review_thresholds_are_provisional": REVIEW_LIMITS,
        "important_limit": (
            "Automatic flags identify cases for visual review; they are not clinically validated rejection thresholds."
        ),
    }
    (output_dir / "geometry_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    for loader in loaders.values():
        loader.close()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
