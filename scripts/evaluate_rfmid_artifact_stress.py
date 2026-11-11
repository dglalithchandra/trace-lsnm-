"""Evaluate isolated glare, resolution and field-of-view stress features."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts.evaluate_rfmid_camera_stress import paired_metrics
from trace_models import load_rfmid_bundle, probabilities_from_embedding
from trace_models.rfmid_dinov2 import DECISION_TOLERANCE


OUTPUT = WORKSPACE / "outputs" / "rfmid_artifact_stress_v2"


def main() -> None:
    feature_path = (
        WORKSPACE
        / "data/features/rfmid_validation_dinov2_artifact_stress_v2.npz"
    )
    checkpoint = (
        WORKSPACE
        / "models/rfmid_dinov2_linear_v1/rfmid_dinov2_linear_v1.pt"
    )
    with np.load(feature_path, allow_pickle=False) as payload:
        record_ids = payload["record_ids"].astype(str)
        clean_embeddings = payload["clean_embeddings"].astype(np.float32)
        artifact_embeddings = payload["artifact_embeddings"].astype(np.float32)
        labels = payload["labels"].astype(np.uint8)
        class_order = payload["class_order"].astype(str).tolist()
        styles = payload["styles"].astype(str)
        severities = payload["severities"].astype(str)
        safety_status = payload["safety_status"].astype(str)

    model, bundle = load_rfmid_bundle(checkpoint)
    if class_order != bundle["class_order"]:
        raise RuntimeError("artifact_stress_class_order_mismatch")

    def score(embeddings: np.ndarray) -> np.ndarray:
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        probabilities = probabilities_from_embedding(
            model, torch.from_numpy(flat), bundle
        ).numpy()
        return probabilities.reshape(*embeddings.shape[:-1], len(class_order))

    clean_probabilities = score(clean_embeddings)
    artifact_probabilities = score(artifact_embeddings)
    supported = np.asarray(
        [
            bundle["support_status"][name] != "unsupported_no_training_positives"
            for name in class_order
        ]
    )
    base_positive_counts = labels.sum(axis=0)
    base_robust = (
        supported
        & (base_positive_counts >= 5)
        & (base_positive_counts < len(labels))
    )

    style_rows = []
    severity_rows = []
    style_details = {}
    for style_index, style in enumerate(styles):
        details = paired_metrics(
            labels,
            clean_probabilities,
            artifact_probabilities[:, style_index],
            clean_embeddings,
            artifact_embeddings[:, style_index],
            bundle["thresholds"],
            base_robust,
        )
        style_details[style] = details
        interval = details["paired_macro_auprc_delta_ci"]
        style_rows.append(
            {
                "stress_style": style,
                "records": len(labels),
                "macro_auprc_delta": details["macro_auprc_delta"],
                "delta_ci95_lower": interval["paired_delta_ci95_lower"],
                "delta_ci95_upper": interval["paired_delta_ci95_upper"],
                "clean_micro_sensitivity": details["clean_threshold_metrics"][
                    "micro_sensitivity"
                ],
                "stress_micro_sensitivity": details["stress_threshold_metrics"][
                    "micro_sensitivity"
                ],
                "decision_flip_fraction": details[
                    "threshold_decision_flip_fraction"
                ],
                "positive_to_negative_flip_count": details[
                    "positive_to_negative_flip_count"
                ],
                "embedding_cosine_similarity_mean": details[
                    "embedding_cosine_similarity_mean"
                ],
            }
        )
        for severity in ("mild", "moderate", "severe"):
            selected = severities[:, style_index] == severity
            counts = labels[selected].sum(axis=0)
            robust = supported & (counts >= 5) & (counts < int(selected.sum()))
            block = paired_metrics(
                labels[selected],
                clean_probabilities[selected],
                artifact_probabilities[selected, style_index],
                clean_embeddings[selected],
                artifact_embeddings[selected, style_index],
                bundle["thresholds"],
                robust,
            )
            interval = block["paired_macro_auprc_delta_ci"]
            severity_rows.append(
                {
                    "stress_style": style,
                    "severity": severity,
                    "records": int(selected.sum()),
                    "evaluated_classes": block["evaluated_classes"],
                    "macro_auprc_delta": block["macro_auprc_delta"],
                    "delta_ci95_lower": interval["paired_delta_ci95_lower"],
                    "delta_ci95_upper": interval["paired_delta_ci95_upper"],
                    "clean_micro_sensitivity": block["clean_threshold_metrics"][
                        "micro_sensitivity"
                    ],
                    "stress_micro_sensitivity": block["stress_threshold_metrics"][
                        "micro_sensitivity"
                    ],
                    "decision_flip_fraction": block[
                        "threshold_decision_flip_fraction"
                    ],
                    "embedding_cosine_similarity_mean": block[
                        "embedding_cosine_similarity_mean"
                    ],
                }
            )

    thresholded = np.asarray([value is not None for value in bundle["thresholds"]])
    threshold_values = np.asarray(
        [float(value) if value is not None else 0.5 for value in bundle["thresholds"]]
    )
    clean_decisions = (
        clean_probabilities[:, thresholded]
        >= threshold_values[thresholded][None, :] - DECISION_TOLERANCE
    )
    truth = labels[:, thresholded].astype(bool)
    quality_rows = []
    for style_index, style in enumerate(styles):
        stressed_decisions = (
            artifact_probabilities[:, style_index, thresholded]
            >= threshold_values[thresholded][None, :] - DECISION_TOLERANCE
        )
        harmful = np.any(clean_decisions & ~stressed_decisions & truth, axis=1)
        flagged = safety_status[:, style_index] != "accept"
        tp = int(np.sum(harmful & flagged))
        fn = int(np.sum(harmful & ~flagged))
        fp = int(np.sum(~harmful & flagged))
        tn = int(np.sum(~harmful & ~flagged))
        quality_rows.append(
            {
                "stress_style": style,
                "views_with_positive_to_negative_flip": int(harmful.sum()),
                "quality_flagged_views": int(flagged.sum()),
                "harmful_view_detection_sensitivity": tp / (tp + fn)
                if tp + fn
                else None,
                "quality_flag_precision_for_harmful_view": tp / (tp + fp)
                if tp + fp
                else None,
                "quality_specificity_for_nonharmful_view": tn / (tn + fp)
                if tn + fp
                else None,
                "harmful_views_missed": fn,
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(style_rows).to_csv(OUTPUT / "style_metrics.csv", index=False)
    pd.DataFrame(severity_rows).to_csv(OUTPUT / "severity_metrics.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(
        OUTPUT / "quality_detection_metrics.csv", index=False
    )
    result = {
        "status": "PASS",
        "protocol": "isolated paired artifact stress; frozen encoder and disease head",
        "by_style": style_details,
        "quality_detection": quality_rows,
        "decision": "characterisation_only",
        "interpretation_rule": (
            "A correction may be proposed only after a separate experiment; this benchmark "
            "does not select or tune corrections."
        ),
        "external_datasets_accessed": 0,
        "protected_test_records_accessed": 0,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
