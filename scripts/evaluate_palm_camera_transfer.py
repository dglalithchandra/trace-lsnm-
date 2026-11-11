"""Development-only PALM camera-transfer baseline for pathologic myopia."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


WORKSPACE = Path(__file__).resolve().parents[1]
C_CANDIDATES = (0.01, 0.1, 1.0, 10.0)
BOOTSTRAP_REPLICATES = 5000
SEED = 20261113


def load_split(path: Path) -> dict[str, np.ndarray | list[str]]:
    with np.load(path, allow_pickle=False) as bundle:
        return {
            "record_ids": bundle["record_ids"].astype(str),
            "embeddings": bundle["embeddings"].astype(np.float32),
            "camera_labels": bundle["camera_labels"].astype(np.uint8),
            "pathology_labels": bundle["pathology_labels"].astype(np.uint8),
            "camera_order": bundle["camera_order"].astype(str).tolist(),
        }


def make_model(c_value: float) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=SEED,
        ),
    )


def threshold_at_specificity(
    labels: np.ndarray, scores: np.ndarray, minimum_specificity: float = 0.90
) -> float:
    negatives = scores[labels == 0]
    if len(negatives) == 0:
        raise ValueError("threshold_requires_negative_examples")
    # The finite-sample empirical threshold is fitted only on out-of-fold
    # training predictions. Higher values are more conservative.
    rank = int(np.ceil(minimum_specificity * len(negatives))) - 1
    rank = min(max(rank, 0), len(negatives) - 1)
    return float(np.nextafter(np.sort(negatives)[rank], np.inf))


def point_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(np.uint8)
    positives = labels == 1
    negatives = labels == 0
    return {
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
        "sensitivity": float(predictions[positives].mean()),
        "specificity": float((predictions[negatives] == 0).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def stratified_bootstrap_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    distributions = {
        name: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
        for name in ("auprc", "auroc", "sensitivity", "specificity", "balanced_accuracy")
    }
    for replicate in range(BOOTSTRAP_REPLICATES):
        query = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        metrics = point_metrics(labels[query], scores[query], threshold)
        for name, value in metrics.items():
            distributions[name][replicate] = value
    return {
        name: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for name, values in distributions.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessing",
        choices=("geometry", "resize_only"),
        default="geometry",
    )
    args = parser.parse_args()
    suffix = "geometry_v1" if args.preprocessing == "geometry" else "resize_only_v1"
    output_name = (
        "palm_camera_transfer_v1"
        if args.preprocessing == "geometry"
        else "palm_camera_transfer_resize_only_v1"
    )
    output_dir = WORKSPACE / "outputs" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = WORKSPACE / "data" / "features"
    train = load_split(feature_dir / f"palm_train_dinov2_vits14_{suffix}.npz")
    validation = load_split(
        feature_dir / f"palm_validation_dinov2_vits14_{suffix}.npz"
    )
    if train["camera_order"] != validation["camera_order"]:
        raise RuntimeError("PALM camera order differs between feature splits")
    camera_order = train["camera_order"]
    rng = np.random.default_rng(SEED)

    train_groups: dict[str, np.ndarray] = {
        "pooled": np.arange(len(train["pathology_labels"]), dtype=np.int64)
    }
    for camera_index, camera_name in enumerate(camera_order):
        train_groups[camera_name] = np.flatnonzero(
            train["camera_labels"] == camera_index
        )

    tuning_rows: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []
    saved_scores: dict[str, np.ndarray] = {}
    source_thresholds: dict[str, float] = {}
    for source_name, source_indices in train_groups.items():
        x_train = train["embeddings"][source_indices]
        y_train = train["pathology_labels"][source_indices]
        class_counts = np.bincount(y_train, minlength=2)
        if class_counts.min() < 5:
            raise RuntimeError(f"Insufficient source classes for {source_name}: {class_counts}")
        folds = int(min(5, class_counts.min()))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
        candidate_scores: list[tuple[float, float]] = []
        for c_value in C_CANDIDATES:
            scores = cross_val_score(
                make_model(c_value),
                x_train,
                y_train,
                cv=splitter,
                scoring="average_precision",
                n_jobs=1,
            )
            mean_score = float(scores.mean())
            candidate_scores.append((c_value, mean_score))
            tuning_rows.append(
                {
                    "training_camera": source_name,
                    "c": c_value,
                    "folds": folds,
                    "mean_cv_auprc": mean_score,
                    "cv_auprc_std": float(scores.std(ddof=1)),
                    "training_records": len(source_indices),
                    "training_positive": int(class_counts[1]),
                    "training_negative": int(class_counts[0]),
                }
            )
        selected_c = max(candidate_scores, key=lambda item: (item[1], -item[0]))[0]
        oof_scores = cross_val_predict(
            make_model(selected_c),
            x_train,
            y_train,
            cv=splitter,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold = threshold_at_specificity(y_train, oof_scores)
        source_thresholds[source_name] = threshold
        model = make_model(selected_c)
        model.fit(x_train, y_train)

        for target_index, target_name in enumerate(camera_order):
            target_indices = np.flatnonzero(
                validation["camera_labels"] == target_index
            )
            y_target = validation["pathology_labels"][target_indices]
            scores = model.predict_proba(validation["embeddings"][target_indices])[:, 1]
            metrics = point_metrics(y_target, scores, threshold)
            intervals = stratified_bootstrap_metrics(
                y_target, scores, threshold, rng=rng
            )
            key = f"{source_name}|{target_name}"
            saved_scores[key] = scores.astype(np.float32)
            row: dict[str, object] = {
                "training_camera": source_name,
                "validation_camera": target_name,
                "same_camera": source_name == target_name,
                "selected_c": selected_c,
                "threshold_from_oof_training": threshold,
                "validation_records": len(target_indices),
                "validation_positive": int(y_target.sum()),
                "validation_negative": int((1 - y_target).sum()),
                "validation_prevalence": float(y_target.mean()),
            }
            for metric_name, metric_value in metrics.items():
                row[metric_name] = metric_value
                row[f"{metric_name}_ci95_lower"] = intervals[metric_name][0]
                row[f"{metric_name}_ci95_upper"] = intervals[metric_name][1]
            evaluation_rows.append(row)

    tuning = pd.DataFrame(tuning_rows)
    evaluation = pd.DataFrame(evaluation_rows)
    tuning.to_csv(output_dir / "source_only_hyperparameter_tuning.csv", index=False)
    evaluation.to_csv(output_dir / "camera_transfer_metrics.csv", index=False)

    # Estimate the within-source versus cross-camera AUROC gap without treating
    # the two camera samples as paired observations.
    gap_rows: list[dict[str, object]] = []
    for source_index, source_name in enumerate(camera_order):
        target_name = camera_order[1 - source_index]
        source_val_indices = np.flatnonzero(
            validation["camera_labels"] == source_index
        )
        target_val_indices = np.flatnonzero(
            validation["camera_labels"] == (1 - source_index)
        )
        y_source = validation["pathology_labels"][source_val_indices]
        y_target = validation["pathology_labels"][target_val_indices]
        source_scores = saved_scores[f"{source_name}|{source_name}"]
        target_scores = saved_scores[f"{source_name}|{target_name}"]
        source_positive = np.flatnonzero(y_source == 1)
        source_negative = np.flatnonzero(y_source == 0)
        target_positive = np.flatnonzero(y_target == 1)
        target_negative = np.flatnonzero(y_target == 0)
        distribution = np.empty(BOOTSTRAP_REPLICATES)
        for replicate in range(BOOTSTRAP_REPLICATES):
            source_query = np.concatenate(
                [
                    rng.choice(source_positive, len(source_positive), replace=True),
                    rng.choice(source_negative, len(source_negative), replace=True),
                ]
            )
            target_query = np.concatenate(
                [
                    rng.choice(target_positive, len(target_positive), replace=True),
                    rng.choice(target_negative, len(target_negative), replace=True),
                ]
            )
            distribution[replicate] = roc_auc_score(
                y_target[target_query], target_scores[target_query]
            ) - roc_auc_score(y_source[source_query], source_scores[source_query])
        gap_rows.append(
            {
                "training_camera": source_name,
                "same_camera_validation": source_name,
                "cross_camera_validation": target_name,
                "cross_minus_same_auroc": float(
                    evaluation.loc[
                        (evaluation.training_camera == source_name)
                        & (evaluation.validation_camera == target_name),
                        "auroc",
                    ].iloc[0]
                    - evaluation.loc[
                        (evaluation.training_camera == source_name)
                        & (evaluation.validation_camera == source_name),
                        "auroc",
                    ].iloc[0]
                ),
                "gap_ci95_lower": float(np.quantile(distribution, 0.025)),
                "gap_ci95_upper": float(np.quantile(distribution, 0.975)),
                "ci_excludes_zero": bool(
                    np.quantile(distribution, 0.025) > 0
                    or np.quantile(distribution, 0.975) < 0
                ),
            }
        )
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(output_dir / "cross_camera_auroc_gaps.csv", index=False)

    def safe_key(value: str) -> str:
        return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")

    score_payload: dict[str, np.ndarray] = {}
    for target_index, target_name in enumerate(camera_order):
        target_indices = np.flatnonzero(validation["camera_labels"] == target_index)
        target_key = safe_key(target_name)
        score_payload[f"labels__{target_key}"] = validation["pathology_labels"][target_indices]
        score_payload[f"record_ids__{target_key}"] = validation["record_ids"][target_indices]
    for combined_key, scores in saved_scores.items():
        source_name, target_name = combined_key.split("|", maxsplit=1)
        score_payload[
            f"scores__{safe_key(source_name)}__{safe_key(target_name)}"
        ] = scores
        score_payload[f"threshold__{safe_key(source_name)}"] = np.asarray(
            [source_thresholds[source_name]], dtype=np.float64
        )
    np.savez_compressed(output_dir / "camera_transfer_scores.npz", **score_payload)

    summary = {
        "status": "PASS",
        "dataset": "PALM",
        "task": "pathologic_myopia",
        "encoder": "frozen DINOv2-S/14",
        "preprocessing": (
            "TRACE geometry-only at 448 px"
            if args.preprocessing == "geometry"
            else "direct square resize-only control at 448 px"
        ),
        "camera_order": camera_order,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "threshold_policy": "90% empirical specificity on source-only out-of-fold training predictions",
        "limitations": [
            "Canon CR-2 has very few negative validation images, so intervals are wide.",
            "Camera and pathology prevalence are associated in PALM.",
            "Patient identifiers were unavailable in the released manifest used here.",
            "Two known cameras cannot support a universal-camera claim.",
        ],
        "test_records_accessed": 0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nMetrics:\n" + evaluation.to_string(index=False))
    print("\nTransfer gaps:\n" + gaps.to_string(index=False))


if __name__ == "__main__":
    main()
