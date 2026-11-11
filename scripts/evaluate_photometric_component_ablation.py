"""Ablate the six TRACE photometric operations with the frozen disease model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts.evaluate_photometric_candidates_with_model import (
    BOOTSTRAPS,
    macro_metrics,
    paired_macro_auprc_delta_ci,
    representation_domain_metrics,
    threshold_metrics,
)
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
    select_correction_components,
)
from trace_preprocessing.quality import compare_images, decide_quality
from trace_preprocessing.reference import reference_from_dict


REFERENCE_NAME = "rfmid_model_source_reference"
STRENGTH = 0.25
COMPONENTS = (
    "colour_balance",
    "saturation",
    "brightness",
    "contrast",
    "gamma",
    "illumination",
)


def main() -> None:
    output_dir = WORKSPACE / "outputs" / "photometric_component_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = reference_from_dict(
        json.loads(
            (
                WORKSPACE
                / "outputs"
                / "photometric_reference"
                / f"{REFERENCE_NAME}.json"
            ).read_text(encoding="utf-8")
        )
    )
    with np.load(
        WORKSPACE
        / "data"
        / "features"
        / "rfmid_validation_dinov2_vits14_geometry_v1.npz",
        allow_pickle=False,
    ) as payload:
        record_ids = payload["record_ids"].astype(str)
        baseline_embeddings = payload["embeddings"].astype(np.float32)
        labels = payload["labels"].astype(np.uint8)
        class_order = payload["class_order"].astype(str).tolist()
    with np.load(
        WORKSPACE / "outputs" / "rfmid_baseline" / "baseline_predictions.npz",
        allow_pickle=False,
    ) as payload:
        if not np.array_equal(record_ids, payload["record_ids"].astype(str)):
            raise RuntimeError("Baseline record order mismatch")
        baseline_probabilities = payload["probabilities"].astype(np.float32)
        domains = payload["proxy_domains"].astype(str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for component ablation")
    encoder = load_dinov2_vits14(WORKSPACE, device=device)
    head, bundle = load_rfmid_bundle(
        WORKSPACE
        / "models"
        / "rfmid_dinov2_linear_v1"
        / "rfmid_dinov2_linear_v1.pt",
        device=device,
    )
    if bundle["class_order"] != class_order:
        raise RuntimeError("Class order mismatch")
    config = PreprocessingConfig(apply_photometric_correction=False)
    geometry_pipeline = TracePreprocessor(config)
    component_embeddings = {
        component: np.zeros_like(baseline_embeddings) for component in COMPONENTS
    }
    pending_tensors: list[torch.Tensor] = []
    pending_locations: list[tuple[str, int]] = []
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
            raise RuntimeError("Non-finite component-ablation embedding")
        for embedding, (component, index) in zip(output_array, pending_locations):
            component_embeddings[component][index] = embedding
        pending_tensors.clear()
        pending_locations.clear()

    loader = ManifestImageLoader(WORKSPACE, allow_test=False)
    if loader.record_ids(split="validation", dataset="RFMiD") != record_ids.tolist():
        raise RuntimeError("Loader record order mismatch")
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
            full = scale_corrections(
                derive_corrections(
                    geometry.stats_before,
                    reference,
                    config.photometric_bounds,
                ),
                STRENGTH,
            )
            for component in COMPONENTS:
                corrections = select_correction_components(full, {component})
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
                        "component": component,
                        "attempted_status": status.value,
                        "fallback_to_geometry": fallback,
                        "reasons": "|".join(reasons),
                        "warnings": "|".join(warnings),
                        **comparison.to_dict(),
                        **corrections.to_dict(),
                    }
                )
                if fallback:
                    component_embeddings[component][record_index] = baseline_embeddings[
                        record_index
                    ]
                else:
                    pending_tensors.append(prepare_dinov2_tensor(attempted))
                    pending_locations.append((component, record_index))
                    if len(pending_tensors) >= 8:
                        flush_batch()
            completed = record_index + 1
            if completed % 50 == 0 or completed == len(record_ids):
                print(
                    f"component_ablation_progress {completed}/{len(record_ids)}",
                    flush=True,
                )
        flush_batch()
    finally:
        loader.close()

    safety = pd.DataFrame(safety_rows)
    safety.to_csv(output_dir / "component_safety_records.csv", index=False)
    supported = np.asarray(
        [
            bundle["support_status"][name]
            != "unsupported_no_training_positives"
            for name in class_order
        ]
    )
    positives = labels.sum(axis=0)
    robust = supported & (positives >= 5) & (positives < len(labels))
    baseline_auprc, baseline_auroc = macro_metrics(
        labels, baseline_probabilities, robust
    )
    feature_mean = torch.as_tensor(bundle["feature_mean"]).cpu().numpy()
    feature_scale = torch.as_tensor(bundle["feature_scale"]).cpu().numpy()
    baseline_domain = representation_domain_metrics(
        baseline_embeddings, domains, feature_mean, feature_scale
    )
    summaries: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    for component in COMPONENTS:
        embeddings = component_embeddings[component]
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
        selected_safety = safety[safety["component"] == component]
        fallback_count = int(selected_safety["fallback_to_geometry"].sum())
        summaries.append(
            {
                "component": component,
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
                    labels, baseline_probabilities, probabilities, robust
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
        )
        for class_index in np.flatnonzero(robust):
            truth = labels[:, class_index]
            threshold = float(bundle["thresholds"][class_index])
            baseline_positive = (
                baseline_probabilities[:, class_index]
                >= threshold - DECISION_TOLERANCE
            )
            candidate_positive = (
                probabilities[:, class_index]
                >= threshold - DECISION_TOLERANCE
            )
            per_class_rows.append(
                {
                    "component": component,
                    "label": class_order[class_index],
                    "validation_positive": int(truth.sum()),
                    "auprc_delta": float(
                        average_precision_score(truth, probabilities[:, class_index])
                        - average_precision_score(
                            truth, baseline_probabilities[:, class_index]
                        )
                    ),
                    "auroc_delta": float(
                        roc_auc_score(truth, probabilities[:, class_index])
                        - roc_auc_score(
                            truth, baseline_probabilities[:, class_index]
                        )
                    ),
                    "sensitivity_delta": float(
                        candidate_positive[truth == 1].mean()
                        - baseline_positive[truth == 1].mean()
                    ),
                    "specificity_delta": float(
                        (~candidate_positive[truth == 0]).mean()
                        - (~baseline_positive[truth == 0]).mean()
                    ),
                }
            )
        np.savez_compressed(
            output_dir / f"{component}.npz",
            record_ids=record_ids,
            embeddings=embeddings.astype(np.float32),
            probabilities=probabilities.astype(np.float32),
            labels=labels,
            class_order=np.asarray(class_order, dtype=np.str_),
            proxy_domains=domains.astype(np.str_),
            split=np.asarray(["validation"], dtype=np.str_),
        )
    pd.DataFrame(per_class_rows).to_csv(
        output_dir / "component_per_class_deltas.csv", index=False
    )

    passing = [
        item
        for item in summaries
        if item["fallback_fraction"] <= 0.02
        and item["macro_auprc_delta"] >= 0.0
        and item["domain_probe_balanced_accuracy_delta"] < 0.0
    ]
    decision = (
        {
            "status": "component_candidate_found",
            "components": [item["component"] for item in passing],
            "next_step": "evaluate only passing components in a train-tuned combination",
        }
        if passing
        else {
            "status": "no_component_candidate_found",
            "next_step": (
                "The six photometric operations do not solve the representation-domain shift alone; "
                "test train-only synthetic-shift consistency or representation-level adaptation."
            ),
        }
    )
    result = {
        "status": "PASS",
        "scope": "single-component RFMiD validation ablation",
        "reference": REFERENCE_NAME,
        "profile_id": reference.profile_id,
        "strength": STRENGTH,
        "baseline": {
            "macro_auprc_robust_21_classes": baseline_auprc,
            "macro_auroc_robust_21_classes": baseline_auroc,
            "threshold_metrics": threshold_metrics(
                labels, baseline_probabilities, bundle["thresholds"]
            ),
            "representation_domain_metrics": baseline_domain,
        },
        "components": summaries,
        "decision": decision,
        "bootstrap_replicates": BOOTSTRAPS,
        "validation_reuse_limit": (
            "Diagnostic ablation on reused validation data; not an independent performance claim."
        ),
        "proxy_disclaimer": "native-resolution groups are not verified camera identities",
        "test_records_accessed": 0,
    }
    (output_dir / "component_ablation_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
