"""Evaluate frozen RFMiD predictions under paired bounded camera stresses.

This is a development stress test, not a simulation claim for any real camera.
The RFMiD final test split and external Paraguay labels are never loaded.
"""

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
    macro_metrics,
    paired_macro_auprc_delta_ci,
    threshold_metrics,
)
from trace_models import load_rfmid_bundle, probabilities_from_embedding
from trace_models.rfmid_dinov2 import DECISION_TOLERANCE


OUTPUT = WORKSPACE / "outputs" / "rfmid_camera_stress_v1"


def stress_style(raw: str) -> str:
    values = json.loads(raw)
    if float(values["vignette_strength"]) > 0:
        return "illumination_falloff"
    if float(values["blur_sigma"]) > 0:
        return "optics_sensor_compression"
    if (
        float(values["red_gain"]) != 1.0
        or float(values["blue_gain"]) != 1.0
        or float(values["saturation_gain"]) != 1.0
    ):
        return "colour_exposure"
    return "tone_contrast"


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def paired_metrics(
    labels: np.ndarray,
    clean_probabilities: np.ndarray,
    stress_probabilities: np.ndarray,
    clean_embeddings: np.ndarray,
    stress_embeddings: np.ndarray,
    thresholds: list[float | None],
    robust: np.ndarray,
) -> dict[str, object]:
    clean_auprc, clean_auroc = macro_metrics(labels, clean_probabilities, robust)
    stress_auprc, stress_auroc = macro_metrics(labels, stress_probabilities, robust)
    selected = np.asarray([value is not None for value in thresholds])
    threshold_values = np.asarray(
        [float(value) if value is not None else 0.5 for value in thresholds]
    )
    clean_decisions = (
        clean_probabilities[:, selected]
        >= threshold_values[selected][None, :] - DECISION_TOLERANCE
    )
    stress_decisions = (
        stress_probabilities[:, selected]
        >= threshold_values[selected][None, :] - DECISION_TOLERANCE
    )
    truth = labels[:, selected].astype(bool)
    positive_to_negative = clean_decisions & ~stress_decisions & truth
    negative_to_positive = ~clean_decisions & stress_decisions & ~truth
    return {
        "records": int(len(labels)),
        "evaluated_classes": int(robust.sum()),
        "clean_macro_auprc": clean_auprc,
        "stress_macro_auprc": stress_auprc,
        "macro_auprc_delta": stress_auprc - clean_auprc,
        "clean_macro_auroc": clean_auroc,
        "stress_macro_auroc": stress_auroc,
        "macro_auroc_delta": stress_auroc - clean_auroc,
        "paired_macro_auprc_delta_ci": paired_macro_auprc_delta_ci(
            labels, clean_probabilities, stress_probabilities, robust
        ),
        "clean_threshold_metrics": threshold_metrics(
            labels, clean_probabilities, thresholds
        ),
        "stress_threshold_metrics": threshold_metrics(
            labels, stress_probabilities, thresholds
        ),
        "threshold_decision_flip_fraction": float(
            np.mean(clean_decisions != stress_decisions)
        ),
        "positive_to_negative_flip_count": int(positive_to_negative.sum()),
        "negative_to_positive_flip_count": int(negative_to_positive.sum()),
        "mean_absolute_probability_change": float(
            np.mean(np.abs(stress_probabilities - clean_probabilities))
        ),
        "embedding_cosine_similarity_mean": float(
            cosine_similarity(clean_embeddings, stress_embeddings).mean()
        ),
        "embedding_cosine_similarity_p05": float(
            np.quantile(cosine_similarity(clean_embeddings, stress_embeddings), 0.05)
        ),
    }


def main() -> None:
    pair_path = (
        WORKSPACE
        / "data"
        / "features"
        / "rfmid_validation_dinov2_camera_stress_v1.npz"
    )
    checkpoint = (
        WORKSPACE
        / "models"
        / "rfmid_dinov2_linear_v1"
        / "rfmid_dinov2_linear_v1.pt"
    )
    with np.load(pair_path, allow_pickle=False) as payload:
        record_ids = payload["record_ids"].astype(str)
        clean_embeddings = payload["clean_embeddings"].astype(np.float32)
        stress_embeddings = payload["stressed_embeddings"].astype(np.float32)
        labels = payload["labels"].astype(np.uint8)
        class_order = payload["class_order"].astype(str).tolist()
        fallback = payload["fallback_to_clean"].astype(bool)
        parameters = payload["stress_parameters"].astype(str)
        safety_status = payload["safety_status"].astype(str)
        structural_similarity = payload["structural_similarity"].astype(np.float64)
        mean_absolute_change = payload["mean_absolute_change"].astype(np.float64)

    model, bundle = load_rfmid_bundle(checkpoint)
    if class_order != bundle["class_order"]:
        raise RuntimeError("camera_stress_class_order_mismatch")

    def score(embeddings: np.ndarray) -> np.ndarray:
        return probabilities_from_embedding(
            model, torch.from_numpy(embeddings), bundle
        ).numpy()

    clean_probabilities = score(clean_embeddings)
    stress_probabilities = score(stress_embeddings)
    styles = np.asarray([stress_style(raw) for raw in parameters], dtype=np.str_)
    valid = ~fallback
    positive_counts = labels[valid].sum(axis=0)
    robust = (
        (positive_counts >= 5)
        & (positive_counts < int(valid.sum()))
        & np.asarray(
            [
                bundle["support_status"][name]
                != "unsupported_no_training_positives"
                for name in class_order
            ]
        )
    )

    overall = paired_metrics(
        labels[valid],
        clean_probabilities[valid],
        stress_probabilities[valid],
        clean_embeddings[valid],
        stress_embeddings[valid],
        bundle["thresholds"],
        robust,
    )

    style_rows: list[dict[str, object]] = []
    style_details: dict[str, object] = {}
    for style in sorted(set(styles)):
        selected = valid & (styles == style)
        style_positive_counts = labels[selected].sum(axis=0)
        style_robust = robust & (style_positive_counts >= 5) & (
            style_positive_counts < int(selected.sum())
        )
        details = paired_metrics(
            labels[selected],
            clean_probabilities[selected],
            stress_probabilities[selected],
            clean_embeddings[selected],
            stress_embeddings[selected],
            bundle["thresholds"],
            style_robust,
        )
        style_details[style] = details
        interval = details["paired_macro_auprc_delta_ci"]
        style_rows.append(
            {
                "stress_style": style,
                "records": details["records"],
                "evaluated_classes": details["evaluated_classes"],
                "macro_auprc_delta": details["macro_auprc_delta"],
                "delta_ci95_lower": interval["paired_delta_ci95_lower"],
                "delta_ci95_upper": interval["paired_delta_ci95_upper"],
                "micro_sensitivity_clean": details["clean_threshold_metrics"][
                    "micro_sensitivity"
                ],
                "micro_sensitivity_stress": details["stress_threshold_metrics"][
                    "micro_sensitivity"
                ],
                "decision_flip_fraction": details[
                    "threshold_decision_flip_fraction"
                ],
                "mean_absolute_probability_change": details[
                    "mean_absolute_probability_change"
                ],
                "embedding_cosine_similarity_mean": details[
                    "embedding_cosine_similarity_mean"
                ],
            }
        )

    per_class_rows = []
    for index, name in enumerate(class_order):
        y = labels[valid, index]
        if y.sum() < 5 or y.sum() == len(y):
            continue
        clean_ap = float(average_precision_score(y, clean_probabilities[valid, index]))
        stress_ap = float(average_precision_score(y, stress_probabilities[valid, index]))
        clean_auc = float(roc_auc_score(y, clean_probabilities[valid, index]))
        stress_auc = float(roc_auc_score(y, stress_probabilities[valid, index]))
        threshold = bundle["thresholds"][index]
        flip_fraction = None
        if threshold is not None:
            clean_decision = clean_probabilities[valid, index] >= float(threshold) - DECISION_TOLERANCE
            stress_decision = stress_probabilities[valid, index] >= float(threshold) - DECISION_TOLERANCE
            flip_fraction = float(np.mean(clean_decision != stress_decision))
        per_class_rows.append(
            {
                "class_name": name,
                "positives": int(y.sum()),
                "clean_auprc": clean_ap,
                "stress_auprc": stress_ap,
                "auprc_delta": stress_ap - clean_ap,
                "clean_auroc": clean_auc,
                "stress_auroc": stress_auc,
                "auroc_delta": stress_auc - clean_auc,
                "mean_absolute_probability_change": float(
                    np.mean(
                        np.abs(
                            stress_probabilities[valid, index]
                            - clean_probabilities[valid, index]
                        )
                    )
                ),
                "threshold_decision_flip_fraction": flip_fraction,
            }
        )

    per_class_rows.sort(key=lambda row: float(row["auprc_delta"]))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(style_rows).to_csv(OUTPUT / "stress_style_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(OUTPUT / "per_class_metrics.csv", index=False)
    np.savez_compressed(
        OUTPUT / "paired_predictions.npz",
        record_ids=record_ids,
        labels=labels,
        class_order=np.asarray(class_order, dtype=np.str_),
        styles=styles,
        fallback=fallback.astype(np.uint8),
        clean_probabilities=clean_probabilities.astype(np.float32),
        stress_probabilities=stress_probabilities.astype(np.float32),
    )
    result = {
        "status": "PASS",
        "protocol": "paired bounded synthetic camera stress on RFMiD validation",
        "overall_nonfallback": overall,
        "by_stress_style": style_details,
        "records": int(len(record_ids)),
        "nonfallback_records": int(valid.sum()),
        "fallback_to_clean_records": int(fallback.sum()),
        "safety_status_counts": {
            name: int((safety_status == name).sum())
            for name in sorted(set(safety_status))
        },
        "image_change": {
            "structural_similarity_p05": float(np.quantile(structural_similarity, 0.05)),
            "mean_absolute_change_mean": float(mean_absolute_change.mean()),
        },
        "decision": (
            "characterisation_only; do not select a correction from this experiment"
        ),
        "missing_stresses_for_v2": [
            "glare",
            "resolution_downsampling",
            "field_of_view_or_missing_retina",
        ],
        "limitations": [
            "Synthetic stresses are not verified simulations of named cameras.",
            "Each image receives one mild style, not a complete factorial sweep.",
            "RFMiD validation was previously used for disease-head development.",
            "Results do not establish unseen-camera or clinical generalisation.",
        ],
        "external_datasets_accessed": 0,
        "protected_test_records_accessed": 0,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
