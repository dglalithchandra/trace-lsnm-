"""Measure the frozen RFMiD baseline before photometric harmonisation.

Only saved RFMiD validation features are read. Native-resolution groups are
used as unverified acquisition-domain proxies, never as asserted camera IDs.
"""

from __future__ import annotations

import json
import sys
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


SEED = 20261108


def disease_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_order: list[str],
    supported: np.ndarray,
    *,
    minimum_positives: int,
) -> dict[str, object]:
    positives = labels.sum(axis=0)
    evaluable = (
        supported
        & (positives >= minimum_positives)
        & ((len(labels) - positives) >= minimum_positives)
    )
    indices = np.flatnonzero(evaluable)
    if not len(indices):
        return {
            "records": int(len(labels)),
            "minimum_positives_and_negatives": minimum_positives,
            "evaluable_classes": 0,
            "class_names": [],
            "macro_auroc": None,
            "macro_auprc": None,
        }
    return {
        "records": int(len(labels)),
        "minimum_positives_and_negatives": minimum_positives,
        "evaluable_classes": int(len(indices)),
        "class_names": [class_order[index] for index in indices],
        "macro_auroc": float(
            np.mean(
                [
                    roc_auc_score(labels[:, index], probabilities[:, index])
                    for index in indices
                ]
            )
        ),
        "macro_auprc": float(
            np.mean(
                [
                    average_precision_score(
                        labels[:, index], probabilities[:, index]
                    )
                    for index in indices
                ]
            )
        ),
    }


def threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float | None],
) -> dict[str, object]:
    selected = np.asarray([value is not None for value in thresholds])
    if not selected.any():
        return {"thresholded_classes": 0}
    threshold_values = np.asarray(
        [float(value) if value is not None else 0.5 for value in thresholds]
    )
    truth = labels[:, selected].astype(bool).ravel()
    predicted = (
        probabilities[:, selected]
        >= threshold_values[selected][None, :] - DECISION_TOLERANCE
    ).ravel()
    true_positive = int(np.sum(truth & predicted))
    false_negative = int(np.sum(truth & ~predicted))
    true_negative = int(np.sum(~truth & ~predicted))
    false_positive = int(np.sum(~truth & predicted))
    return {
        "thresholded_classes": int(selected.sum()),
        "positive_label_instances": int(truth.sum()),
        "negative_label_instances": int((~truth).sum()),
        "micro_sensitivity": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        ),
        "micro_specificity": (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive
            else None
        ),
    }


def main() -> None:
    output_dir = WORKSPACE / "outputs" / "rfmid_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = (
        WORKSPACE
        / "data"
        / "features"
        / "rfmid_validation_dinov2_vits14_geometry_v1.npz"
    )
    with np.load(feature_path, allow_pickle=False) as payload:
        record_ids = payload["record_ids"].astype(str)
        embeddings = payload["embeddings"].astype(np.float32)
        labels = payload["labels"].astype(np.uint8)
        class_order = payload["class_order"].astype(str).tolist()
        stored_split = str(payload["split"][0])
    if stored_split != "validation":
        raise RuntimeError(f"Expected validation features, found {stored_split}")

    head, bundle = load_rfmid_bundle(
        WORKSPACE
        / "models"
        / "rfmid_dinov2_linear_v1"
        / "rfmid_dinov2_linear_v1.pt"
    )
    if class_order != bundle["class_order"]:
        raise RuntimeError("Feature and model class orders differ")
    probabilities = (
        probabilities_from_embedding(head, torch.from_numpy(embeddings), bundle)
        .detach()
        .cpu()
        .numpy()
    )

    audit = pd.read_csv(
        WORKSPACE / "outputs" / "geometry_audit" / "geometry_audit_records.csv",
        usecols=["record_id", "effective_split", "dataset", "camera_domain"],
    )
    audit = audit[
        (audit["effective_split"] == "validation") & (audit["dataset"] == "RFMiD")
    ]
    domain_by_id = dict(zip(audit["record_id"].astype(str), audit["camera_domain"]))
    missing = [record_id for record_id in record_ids if record_id not in domain_by_id]
    if missing:
        raise RuntimeError(f"Missing proxy domains for {len(missing)} RFMiD records")
    domains = np.asarray([domain_by_id[record_id] for record_id in record_ids])

    supported = np.asarray(
        [
            bundle["support_status"][name]
            != "unsupported_no_training_positives"
            for name in class_order
        ]
    )
    overall = disease_metrics(
        labels,
        probabilities,
        class_order,
        supported,
        minimum_positives=5,
    )
    overall["threshold_metrics"] = threshold_metrics(
        labels, probabilities, bundle["thresholds"]
    )

    by_domain: dict[str, object] = {}
    for domain in sorted(set(domains)):
        included = domains == domain
        metrics = disease_metrics(
            labels[included],
            probabilities[included],
            class_order,
            supported,
            minimum_positives=5,
        )
        metrics["threshold_metrics"] = threshold_metrics(
            labels[included], probabilities[included], bundle["thresholds"]
        )
        by_domain[domain] = metrics

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        ),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    predicted_domains = cross_val_predict(probe, embeddings, domains, cv=folds)
    domain_probe = {
        "role": "unverified native-resolution-domain proxy, not camera identity",
        "classes": sorted(set(domains)),
        "records_per_class": {
            name: int(np.sum(domains == name)) for name in sorted(set(domains))
        },
        "chance_balanced_accuracy": float(1.0 / len(set(domains))),
        "five_fold_balanced_accuracy": float(
            balanced_accuracy_score(domains, predicted_domains)
        ),
        "five_fold_macro_f1": float(
            f1_score(domains, predicted_domains, average="macro")
        ),
    }

    mean = torch.as_tensor(bundle["feature_mean"]).numpy()
    scale = torch.as_tensor(bundle["feature_scale"]).numpy()
    standardized = (embeddings - mean) / scale
    centroids = {
        domain: standardized[domains == domain].mean(axis=0)
        for domain in sorted(set(domains))
    }
    pairwise_centroid_distance: dict[str, float] = {}
    names = sorted(centroids)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pairwise_centroid_distance[f"{left}__vs__{right}"] = float(
                np.linalg.norm(centroids[left] - centroids[right])
                / np.sqrt(embeddings.shape[1])
            )

    np.savez_compressed(
        output_dir / "baseline_predictions.npz",
        record_ids=record_ids,
        probabilities=probabilities.astype(np.float32),
        labels=labels,
        class_order=np.asarray(class_order, dtype=np.str_),
        proxy_domains=domains.astype(np.str_),
        split=np.asarray(["validation"], dtype=np.str_),
    )
    summary = {
        "status": "PASS",
        "scope": "RFMiD validation geometry-only frozen-model baseline",
        "records": int(len(record_ids)),
        "overall": overall,
        "by_unverified_resolution_domain": by_domain,
        "domain_probe": domain_probe,
        "pairwise_standardized_centroid_distance": pairwise_centroid_distance,
        "important_limit": (
            "Native resolution is an acquisition-domain proxy and is not verified camera metadata."
        ),
        "model_scores_are_not_clinical_certainty": True,
        "test_records_accessed": 0,
    }
    (output_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
