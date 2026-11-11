"""Evaluate screened TRACE filters with the frozen RFMiD disease model.

The RFMiD final test split is never loaded. A correction that fails the image
quality policy falls back to the geometry-only model input.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_models import load_rfmid_bundle, probabilities_from_embedding
from trace_models.rfmid_dinov2 import DECISION_TOLERANCE
from trace_models.dinov2 import load_dinov2_vits14
from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.adapters import prepare_dinov2_tensor
from trace_preprocessing.loader import ManifestImageLoader
from trace_preprocessing.measurements import measure_photometry, measure_quality
from trace_preprocessing.photometric import (
    apply_corrections,
    derive_corrections,
    scale_corrections,
)
from trace_preprocessing.quality import compare_images, decide_quality
from trace_preprocessing.reference import reference_from_dict


SEED = 20261108
BOOTSTRAPS = 1000


def macro_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_mask: np.ndarray,
) -> tuple[float, float]:
    indices = np.flatnonzero(class_mask)
    return (
        float(
            np.mean(
                [
                    average_precision_score(labels[:, index], probabilities[:, index])
                    for index in indices
                ]
            )
        ),
        float(
            np.mean(
                [
                    roc_auc_score(labels[:, index], probabilities[:, index])
                    for index in indices
                ]
            )
        ),
    )


def threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float | None],
) -> dict[str, float | int | None]:
    selected = np.asarray([value is not None for value in thresholds])
    values = np.asarray(
        [float(value) if value is not None else 0.5 for value in thresholds]
    )
    truth = labels[:, selected].astype(bool).ravel()
    predicted = (
        probabilities[:, selected]
        >= values[selected][None, :] - DECISION_TOLERANCE
    ).ravel()
    tp = int(np.sum(truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    tn = int(np.sum(~truth & ~predicted))
    fp = int(np.sum(~truth & predicted))
    return {
        "thresholded_classes": int(selected.sum()),
        "micro_sensitivity": tp / (tp + fn) if tp + fn else None,
        "micro_specificity": tn / (tn + fp) if tn + fp else None,
    }


def paired_macro_auprc_delta_ci(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    class_mask: np.ndarray,
) -> dict[str, float | int]:
    rng = np.random.default_rng(SEED)
    indices = np.flatnonzero(class_mask)
    deltas: list[float] = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.integers(0, len(labels), size=len(labels))
        class_deltas: list[float] = []
        for class_index in indices:
            sampled_labels = labels[sampled, class_index]
            if sampled_labels.min() == sampled_labels.max():
                continue
            class_deltas.append(
                average_precision_score(
                    sampled_labels, candidate[sampled, class_index]
                )
                - average_precision_score(
                    sampled_labels, baseline[sampled, class_index]
                )
            )
        if class_deltas:
            deltas.append(float(np.mean(class_deltas)))
    low, median, high = np.percentile(deltas, [2.5, 50.0, 97.5])
    return {
        "bootstrap_replicates": len(deltas),
        "paired_delta_ci95_lower": float(low),
        "paired_delta_median": float(median),
        "paired_delta_ci95_upper": float(high),
    }


def representation_domain_metrics(
    embeddings: np.ndarray,
    domains: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> dict[str, object]:
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        ),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    predicted = cross_val_predict(probe, embeddings, domains, cv=folds)
    standardized = (embeddings - feature_mean) / feature_scale
    names = sorted(set(domains))
    centroids = {
        name: standardized[domains == name].mean(axis=0) for name in names
    }
    distances = {
        f"{left}__vs__{right}": float(
            np.linalg.norm(centroids[left] - centroids[right])
            / np.sqrt(embeddings.shape[1])
        )
        for left, right in combinations(names, 2)
    }
    return {
        "probe_balanced_accuracy": float(
            balanced_accuracy_score(domains, predicted)
        ),
        "probe_macro_f1": float(f1_score(domains, predicted, average="macro")),
        "mean_pairwise_standardized_centroid_distance": float(
            np.mean(list(distances.values()))
        ),
        "pairwise_standardized_centroid_distance": distances,
    }


def main() -> None:
    output_dir = WORKSPACE / "outputs" / "photometric_model_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    screening = json.loads(
        (
            WORKSPACE
            / "outputs"
            / "photometric_screening"
            / "screening_summary.json"
        ).read_text(encoding="utf-8")
    )
    selected = screening["selected_for_frozen_disease_model_evaluation"]
    specs = [(item["reference"], float(item["strength"])) for item in selected]
    references = {
        name: reference_from_dict(
            json.loads(
                (
                    WORKSPACE
                    / "outputs"
                    / "photometric_reference"
                    / f"{name}.json"
                ).read_text(encoding="utf-8")
            )
        )
        for name, _ in specs
    }

    feature_path = (
        WORKSPACE
        / "data"
        / "features"
        / "rfmid_validation_dinov2_vits14_geometry_v1.npz"
    )
    with np.load(feature_path, allow_pickle=False) as payload:
        record_ids = payload["record_ids"].astype(str)
        baseline_embeddings = payload["embeddings"].astype(np.float32)
        labels = payload["labels"].astype(np.uint8)
        class_order = payload["class_order"].astype(str).tolist()
    with np.load(
        WORKSPACE / "outputs" / "rfmid_baseline" / "baseline_predictions.npz",
        allow_pickle=False,
    ) as payload:
        if not np.array_equal(record_ids, payload["record_ids"].astype(str)):
            raise RuntimeError("Baseline prediction record order mismatch")
        baseline_probabilities = payload["probabilities"].astype(np.float32)
        domains = payload["proxy_domains"].astype(str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for full candidate evaluation")
    encoder = load_dinov2_vits14(WORKSPACE, device=device)
    head, bundle = load_rfmid_bundle(
        WORKSPACE
        / "models"
        / "rfmid_dinov2_linear_v1"
        / "rfmid_dinov2_linear_v1.pt",
        device=device,
    )
    if bundle["class_order"] != class_order:
        raise RuntimeError("Model and feature class order mismatch")

    config = PreprocessingConfig(apply_photometric_correction=False)
    geometry_pipeline = TracePreprocessor(config)
    candidate_embeddings = {
        spec: np.zeros_like(baseline_embeddings) for spec in specs
    }
    pending_tensors: list[torch.Tensor] = []
    pending_locations: list[tuple[tuple[str, float], int]] = []
    safety_rows: list[dict[str, object]] = []

    def flush_batch() -> None:
        if not pending_tensors:
            return
        inputs = torch.stack(pending_tensors).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            output = encoder(inputs)
        output_array = output.float().cpu().numpy()
        if not np.isfinite(output_array).all():
            raise RuntimeError("Non-finite filtered DINOv2 embedding")
        for embedding, (spec, index) in zip(output_array, pending_locations):
            candidate_embeddings[spec][index] = embedding
        pending_tensors.clear()
        pending_locations.clear()

    loader = ManifestImageLoader(WORKSPACE, allow_test=False)
    loader_ids = loader.record_ids(split="validation", dataset="RFMiD")
    if loader_ids != record_ids.tolist():
        raise RuntimeError("Validation loader and feature record order mismatch")
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
                raise RuntimeError(f"Geometry failed: {record_id}")
            for spec in specs:
                reference_name, strength = spec
                corrections = scale_corrections(
                    derive_corrections(
                        geometry.stats_before,
                        references[reference_name],
                        config.photometric_bounds,
                    ),
                    strength,
                )
                attempted = apply_corrections(
                    geometry.geometry_only_image,
                    geometry.retinal_mask,
                    corrections,
                )
                stats_after = measure_photometry(attempted, geometry.retinal_mask)
                quality_after = measure_quality(attempted, geometry.retinal_mask)
                comparison = compare_images(
                    geometry.geometry_only_image,
                    attempted,
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
                fallback = status == ProcessingStatus.REJECT
                safety_rows.append(
                    {
                        "record_id": record_id,
                        "proxy_domain": domains[record_index],
                        "reference": reference_name,
                        "strength": strength,
                        "attempted_status": status.value,
                        "fallback_to_geometry": fallback,
                        "reasons": "|".join(reasons),
                        "warnings": "|".join(warnings),
                        **comparison.to_dict(),
                    }
                )
                if fallback:
                    candidate_embeddings[spec][record_index] = baseline_embeddings[
                        record_index
                    ]
                else:
                    pending_tensors.append(prepare_dinov2_tensor(attempted))
                    pending_locations.append((spec, record_index))
                    if len(pending_tensors) >= 8:
                        flush_batch()
            completed = record_index + 1
            if completed % 50 == 0 or completed == len(record_ids):
                print(
                    f"disease_candidate_progress {completed}/{len(record_ids)}",
                    flush=True,
                )
        flush_batch()
    finally:
        loader.close()

    safety = pd.DataFrame(safety_rows)
    safety.to_csv(output_dir / "candidate_safety_records.csv", index=False)
    supported = np.asarray(
        [
            bundle["support_status"][name]
            != "unsupported_no_training_positives"
            for name in class_order
        ]
    )
    validation_positive = labels.sum(axis=0)
    robust = supported & (validation_positive >= 5) & (
        validation_positive < len(labels)
    )
    baseline_auprc, baseline_auroc = macro_metrics(
        labels, baseline_probabilities, robust
    )
    feature_mean = torch.as_tensor(bundle["feature_mean"]).cpu().numpy()
    feature_scale = torch.as_tensor(bundle["feature_scale"]).cpu().numpy()
    baseline_domain = representation_domain_metrics(
        baseline_embeddings, domains, feature_mean, feature_scale
    )
    baseline_metrics = {
        "macro_auprc_robust_21_classes": baseline_auprc,
        "macro_auroc_robust_21_classes": baseline_auroc,
        "threshold_metrics": threshold_metrics(
            labels, baseline_probabilities, bundle["thresholds"]
        ),
        "representation_domain_metrics": baseline_domain,
    }

    candidate_summaries: list[dict[str, object]] = []
    for spec in specs:
        reference_name, strength = spec
        embeddings = candidate_embeddings[spec]
        probabilities = (
            probabilities_from_embedding(
                head, torch.from_numpy(embeddings).to(device), bundle
            )
            .detach()
            .cpu()
            .numpy()
        )
        auprc, auroc = macro_metrics(labels, probabilities, robust)
        domain_result = representation_domain_metrics(
            embeddings, domains, feature_mean, feature_scale
        )
        selected_safety = safety[
            (safety["reference"] == reference_name)
            & np.isclose(safety["strength"], strength)
        ]
        fallback_count = int(selected_safety["fallback_to_geometry"].sum())
        summary = {
            "reference": reference_name,
            "profile_id": references[reference_name].profile_id,
            "strength": strength,
            "records": len(record_ids),
            "fallback_count": fallback_count,
            "fallback_fraction": fallback_count / len(record_ids),
            "macro_auprc_robust_21_classes": auprc,
            "macro_auprc_delta": auprc - baseline_auprc,
            "macro_auroc_robust_21_classes": auroc,
            "macro_auroc_delta": auroc - baseline_auroc,
            "threshold_metrics": threshold_metrics(
                labels, probabilities, bundle["thresholds"]
            ),
            "mean_absolute_prediction_change": float(
                np.mean(np.abs(probabilities - baseline_probabilities))
            ),
            "paired_macro_auprc_delta_bootstrap": paired_macro_auprc_delta_ci(
                labels,
                baseline_probabilities,
                probabilities,
                robust,
            ),
            "representation_domain_metrics": domain_result,
            "domain_probe_balanced_accuracy_delta": (
                domain_result["probe_balanced_accuracy"]
                - baseline_domain["probe_balanced_accuracy"]
            ),
            "centroid_distance_delta": (
                domain_result["mean_pairwise_standardized_centroid_distance"]
                - baseline_domain["mean_pairwise_standardized_centroid_distance"]
            ),
        }
        candidate_summaries.append(summary)
        safe_name = f"{reference_name}_strength_{strength:.2f}".replace(".", "p")
        np.savez_compressed(
            output_dir / f"{safe_name}.npz",
            record_ids=record_ids,
            embeddings=embeddings.astype(np.float32),
            probabilities=probabilities.astype(np.float32),
            labels=labels,
            class_order=np.asarray(class_order, dtype=np.str_),
            proxy_domains=domains.astype(np.str_),
            split=np.asarray(["validation"], dtype=np.str_),
        )

    pareto = [
        item
        for item in candidate_summaries
        if item["fallback_fraction"] <= 0.02
        and item["macro_auprc_delta"] >= 0.0
        and item["domain_probe_balanced_accuracy_delta"] < 0.0
    ]
    if pareto:
        chosen = min(
            pareto,
            key=lambda item: (
                item["representation_domain_metrics"]["probe_balanced_accuracy"],
                -item["macro_auprc_robust_21_classes"],
            ),
        )
        decision = {
            "status": "provisional_joint_improvement_observed",
            "selected_reference": chosen["reference"],
            "selected_strength": chosen["strength"],
            "important_limit": (
                "Selection is validation-derived and still requires untouched final-test evaluation."
            ),
        }
    else:
        decision = {
            "status": "no_candidate_frozen",
            "reason": (
                "No screened candidate simultaneously preserved or improved macro AUPRC, "
                "reduced representation-domain probe accuracy, and kept fallback at or below 2%."
            ),
        }

    result = {
        "status": "PASS",
        "scope": "RFMiD validation frozen-model comparison",
        "records": len(record_ids),
        "baseline": baseline_metrics,
        "candidates": candidate_summaries,
        "decision": decision,
        "bootstrap_replicates": BOOTSTRAPS,
        "proxy_disclaimer": "native-resolution groups are not verified camera identities",
        "score_disclaimer": "model scores are not clinical certainty",
        "validation_reuse_limit": (
            "Validation was already used for head selection/calibration; results are development evidence, not independent performance."
        ),
        "test_records_accessed": 0,
    }
    (output_dir / "model_evaluation_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
