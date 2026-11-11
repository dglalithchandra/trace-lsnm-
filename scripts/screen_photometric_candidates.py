"""Screen train-derived photometric references on RFMiD validation images.

This stage uses image-safety and photometric-domain metrics only. It does not
select or freeze a final filter; selected candidates must next pass the frozen
disease-model non-inferiority evaluation.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.loader import ManifestImageLoader
from trace_preprocessing.measurements import measure_photometry, measure_quality
from trace_preprocessing.photometric import (
    apply_corrections,
    derive_corrections,
    scale_corrections,
)
from trace_preprocessing.quality import compare_images, decide_quality
from trace_preprocessing.reference import REFERENCE_FIELDS, reference_from_dict


SEED = 20261108
STRENGTHS = (0.25, 0.50, 0.75, 1.00)
REFERENCE_NAMES = (
    "rfmid_model_source_reference",
    "pooled_training_reference",
    "balanced_proxy_quality_reference",
)


def domain_metrics(
    values: np.ndarray,
    domains: np.ndarray,
    baseline_scale: np.ndarray,
) -> dict[str, float]:
    standardized = values / baseline_scale
    names = sorted(set(domains))
    centroids = {
        name: standardized[domains == name].mean(axis=0) for name in names
    }
    distances = [
        float(
            np.linalg.norm(centroids[left] - centroids[right])
            / np.sqrt(values.shape[1])
        )
        for left, right in combinations(names, 2)
    ]
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        ),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    predicted = cross_val_predict(probe, values, domains, cv=folds)
    return {
        "mean_pairwise_standardized_centroid_distance": float(np.mean(distances)),
        "photometric_probe_balanced_accuracy": float(
            balanced_accuracy_score(domains, predicted)
        ),
        "photometric_probe_macro_f1": float(
            f1_score(domains, predicted, average="macro")
        ),
    }


def main() -> None:
    output_dir = WORKSPACE / "outputs" / "photometric_screening"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = WORKSPACE / "outputs" / "photometric_reference"
    references = {
        name: reference_from_dict(
            json.loads((reference_dir / f"{name}.json").read_text(encoding="utf-8"))
        )
        for name in REFERENCE_NAMES
    }
    specifications = [
        (name, strength) for name in REFERENCE_NAMES for strength in STRENGTHS
    ]

    audit = pd.read_csv(
        WORKSPACE / "outputs" / "geometry_audit" / "geometry_audit_records.csv",
        usecols=["record_id", "effective_split", "dataset", "camera_domain"],
    )
    audit = audit[
        (audit["effective_split"] == "validation") & (audit["dataset"] == "RFMiD")
    ]
    domain_by_id = dict(zip(audit["record_id"].astype(str), audit["camera_domain"]))

    loader = ManifestImageLoader(WORKSPACE, allow_test=False)
    record_ids = loader.record_ids(split="validation", dataset="RFMiD")
    domains = np.asarray([domain_by_id[record_id] for record_id in record_ids])
    config = PreprocessingConfig(apply_photometric_correction=False)
    geometry_pipeline = TracePreprocessor(config)
    before_values = np.zeros((len(record_ids), len(REFERENCE_FIELDS)), dtype=np.float64)
    after_values = {
        (name, strength): np.zeros_like(before_values)
        for name, strength in specifications
    }
    record_rows: list[dict[str, object]] = []
    try:
        for record_index, record_id in enumerate(record_ids):
            image, row = loader.load_image(record_id)
            if row["_effective_split"] != "validation":
                raise RuntimeError(f"Split mismatch: {record_id}")
            geometry = geometry_pipeline.process(image, image_id=record_id)
            if (
                geometry.status == ProcessingStatus.REJECT
                or geometry.geometry_only_image is None
                or geometry.retinal_mask is None
                or geometry.stats_before is None
                or geometry.quality_before is None
            ):
                raise RuntimeError(f"Geometry failed during screening: {record_id}")
            before_values[record_index] = [
                getattr(geometry.stats_before, field) for field in REFERENCE_FIELDS
            ]
            for reference_name, strength in specifications:
                full = derive_corrections(
                    geometry.stats_before,
                    references[reference_name],
                    config.photometric_bounds,
                )
                corrections = scale_corrections(full, strength)
                processed = apply_corrections(
                    geometry.geometry_only_image,
                    geometry.retinal_mask,
                    corrections,
                )
                stats_after = measure_photometry(processed, geometry.retinal_mask)
                quality_after = measure_quality(processed, geometry.retinal_mask)
                comparison = compare_images(
                    geometry.geometry_only_image,
                    processed,
                    geometry.retinal_mask,
                    geometry.stats_before,
                    stats_after,
                )
                status, reasons, warnings = decide_quality(
                    comparison,
                    geometry.quality_before,
                    quality_after,
                    config.quality,
                )
                after_values[(reference_name, strength)][record_index] = [
                    getattr(stats_after, field) for field in REFERENCE_FIELDS
                ]
                record_rows.append(
                    {
                        "record_id": record_id,
                        "proxy_domain": domains[record_index],
                        "reference": reference_name,
                        "strength": strength,
                        "status": status.value,
                        "reasons": "|".join(reasons),
                        "warnings": "|".join(warnings),
                        **comparison.to_dict(),
                        **corrections.to_dict(),
                    }
                )
            completed = record_index + 1
            if completed % 50 == 0 or completed == len(record_ids):
                print(
                    f"photometric_screen_progress {completed}/{len(record_ids)}",
                    flush=True,
                )
    finally:
        loader.close()

    records = pd.DataFrame(record_rows)
    records.to_csv(output_dir / "screening_records.csv", index=False)
    baseline_scale = np.maximum(before_values.std(axis=0), 1e-6)
    baseline_domain = domain_metrics(before_values, domains, baseline_scale)
    summaries: list[dict[str, object]] = []
    for reference_name, strength in specifications:
        selected = records[
            (records["reference"] == reference_name)
            & np.isclose(records["strength"], strength)
        ]
        candidate_domain = domain_metrics(
            after_values[(reference_name, strength)], domains, baseline_scale
        )
        reject_count = int((selected["status"] == ProcessingStatus.REJECT.value).sum())
        summary = {
            "reference": reference_name,
            "profile_id": references[reference_name].profile_id,
            "strength": strength,
            "records": int(len(selected)),
            "reject_count": reject_count,
            "reject_fraction": reject_count / len(selected),
            "warning_count": int(
                (selected["status"] == ProcessingStatus.WARNING.value).sum()
            ),
            "mean_structural_similarity": float(
                selected["structural_similarity"].mean()
            ),
            "p05_structural_similarity": float(
                selected["structural_similarity"].quantile(0.05)
            ),
            "mean_absolute_change": float(selected["mean_absolute_change"].mean()),
            "p95_of_mean_absolute_change": float(
                selected["mean_absolute_change"].quantile(0.95)
            ),
            **candidate_domain,
            "centroid_gap_change": (
                candidate_domain["mean_pairwise_standardized_centroid_distance"]
                - baseline_domain["mean_pairwise_standardized_centroid_distance"]
            ),
            "photometric_probe_balanced_accuracy_change": (
                candidate_domain["photometric_probe_balanced_accuracy"]
                - baseline_domain["photometric_probe_balanced_accuracy"]
            ),
        }
        summaries.append(summary)

    selected_for_disease_model: list[dict[str, object]] = []
    maximum_screening_fallback_fraction = 0.02
    for reference_name in REFERENCE_NAMES:
        feasible = [
            item
            for item in summaries
            if item["reference"] == reference_name
            and item["reject_fraction"] <= maximum_screening_fallback_fraction
        ]
        if not feasible:
            reference_items = [
                item for item in summaries if item["reference"] == reference_name
            ]
            minimum_fallback = min(item["reject_fraction"] for item in reference_items)
            feasible = [
                item
                for item in reference_items
                if item["reject_fraction"] == minimum_fallback
            ]
        chosen = min(
            feasible,
            key=lambda item: (
                item["photometric_probe_balanced_accuracy"],
                item["mean_pairwise_standardized_centroid_distance"],
                -item["mean_structural_similarity"],
            ),
        )
        selected_for_disease_model.append(
            {
                "reference": chosen["reference"],
                "profile_id": chosen["profile_id"],
                "strength": chosen["strength"],
                "selection_role": (
                    "lowest residual photometric-domain probe among settings with "
                    "at most 2% quality-gated fallback; not yet disease-model approved"
                ),
            }
        )

    summary_payload = {
        "status": "PASS",
        "scope": "RFMiD validation photometric and image-safety screening",
        "records": len(record_ids),
        "candidate_count": len(summaries),
        "baseline_photometric_domain_metrics": baseline_domain,
        "candidates": summaries,
        "selected_for_frozen_disease_model_evaluation": selected_for_disease_model,
        "screening_fallback_fraction_limit": maximum_screening_fallback_fraction,
        "deployment_safety_rule": (
            "if a correction fails quality policy, use the geometry-only image"
        ),
        "selection_status": "screened only; no final filter selected or frozen",
        "proxy_disclaimer": "native-resolution groups are not verified camera identities",
        "test_records_accessed": 0,
    }
    (output_dir / "screening_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
