"""Evaluate source-only calibration, OOD flags and abstention under PALM shift."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neighbors import NearestNeighbors


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from evaluate_palm_camera_transfer import (
    C_CANDIDATES,
    load_split,
    make_model,
    threshold_at_specificity,
)
from trace_trust import (
    expected_calibration_error,
    finite_conformal_quantile,
    mondrian_prediction_sets,
    risk_coverage_auc,
    selective_metrics,
)


OUTPUT_DIR = WORKSPACE / "outputs" / "palm_trust_layer_v1"
SEED = 20261116
CONFORMAL_ALPHA = 0.10
SOURCE_COVERAGES = (0.90, 0.80, 0.70)
OOD_BOOTSTRAP_REPLICATES = 2000


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def fit_platt(oof_probabilities: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(C=1_000.0, solver="lbfgs", random_state=SEED)
    calibrator.fit(logit(oof_probabilities).reshape(-1, 1), labels)
    return calibrator


def apply_platt(calibrator: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(logit(probabilities).reshape(-1, 1))[:, 1]


def knn_distances(
    source_features: np.ndarray, query_features: np.ndarray, *, self_query: bool
) -> np.ndarray:
    neighbors = min(6 if self_query else 5, len(source_features))
    model = NearestNeighbors(n_neighbors=neighbors, metric="cosine", n_jobs=1)
    model.fit(source_features)
    distances = model.kneighbors(query_features, return_distance=True)[0]
    if self_query:
        if distances.shape[1] < 2:
            raise RuntimeError("knn_loo_requires_multiple_source_records")
        distances = distances[:, 1:]
    return distances.mean(axis=1)


def squared_mahalanobis(model: LedoitWolf, features: np.ndarray) -> np.ndarray:
    centered = features - model.location_
    return np.einsum("ij,jk,ik->i", centered, model.precision_, centered)


def oof_mahalanobis(features: np.ndarray) -> np.ndarray:
    folds = min(5, len(features))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    scores = np.empty(len(features), dtype=np.float64)
    for fit_indices, held_indices in splitter.split(features):
        model = LedoitWolf().fit(features[fit_indices])
        scores[held_indices] = squared_mahalanobis(model, features[held_indices])
    return scores


def bootstrap_ood_auc(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[float, float]:
    distribution = np.empty(OOD_BOOTSTRAP_REPLICATES)
    for replicate in range(OOD_BOOTSTRAP_REPLICATES):
        sampled_id = rng.choice(id_scores, len(id_scores), replace=True)
        sampled_ood = rng.choice(ood_scores, len(ood_scores), replace=True)
        labels = np.concatenate(
            [np.zeros(len(sampled_id), dtype=np.uint8), np.ones(len(sampled_ood), dtype=np.uint8)]
        )
        scores = np.concatenate([sampled_id, sampled_ood])
        distribution[replicate] = roc_auc_score(labels, scores)
    return (
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    )


def conformal_metrics(labels: np.ndarray, sets: np.ndarray) -> dict[str, float]:
    covered = sets[np.arange(len(labels)), labels]
    sizes = sets.sum(axis=1)
    positive = labels == 1
    negative = labels == 0
    return {
        "marginal_coverage": float(covered.mean()),
        "negative_coverage": float(covered[negative].mean()),
        "positive_coverage": float(covered[positive].mean()),
        "mean_set_size": float(sizes.mean()),
        "singleton_fraction": float((sizes == 1).mean()),
        "ambiguous_fraction": float((sizes == 2).mean()),
        "empty_fraction": float((sizes == 0).mean()),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_dir = WORKSPACE / "data" / "features"
    train = load_split(feature_dir / "palm_train_dinov2_vits14_geometry_v1.npz")
    validation = load_split(
        feature_dir / "palm_validation_dinov2_vits14_geometry_v1.npz"
    )
    if train["camera_order"] != validation["camera_order"]:
        raise RuntimeError("PALM camera order differs between feature splits")
    camera_order = train["camera_order"]
    train_features_all = l2_normalize(train["embeddings"])
    validation_features_all = l2_normalize(validation["embeddings"])
    rng = np.random.default_rng(SEED)

    source_groups: dict[str, np.ndarray] = {
        "pooled": np.arange(len(train["pathology_labels"]), dtype=np.int64)
    }
    for camera_index, camera_name in enumerate(camera_order):
        source_groups[camera_name] = np.flatnonzero(
            train["camera_labels"] == camera_index
        )

    tuning_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    conformal_rows: list[dict[str, object]] = []
    ood_rows: list[dict[str, object]] = []
    selective_rows: list[dict[str, object]] = []
    aurc_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for source_name, source_indices in source_groups.items():
        x_source = train_features_all[source_indices]
        y_source = train["pathology_labels"][source_indices]
        class_counts = np.bincount(y_source, minlength=2)
        folds = int(min(5, class_counts.min()))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
        candidate_results: list[tuple[float, float]] = []
        for c_value in C_CANDIDATES:
            scores = cross_val_score(
                make_model(c_value),
                x_source,
                y_source,
                cv=splitter,
                scoring="average_precision",
                n_jobs=1,
            )
            mean_score = float(scores.mean())
            candidate_results.append((c_value, mean_score))
            tuning_rows.append(
                {
                    "training_camera": source_name,
                    "c": c_value,
                    "mean_cv_auprc": mean_score,
                    "cv_auprc_std": float(scores.std(ddof=1)),
                    "folds": folds,
                }
            )
        selected_c = max(candidate_results, key=lambda item: (item[1], -item[0]))[0]
        base_model = make_model(selected_c)
        oof_raw = cross_val_predict(
            base_model,
            x_source,
            y_source,
            cv=splitter,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        calibrator = fit_platt(oof_raw, y_source)
        oof_calibrated = apply_platt(calibrator, oof_raw)
        decision_threshold = threshold_at_specificity(y_source, oof_calibrated)
        confidence_source = np.abs(oof_calibrated - decision_threshold)

        q0 = finite_conformal_quantile(
            oof_calibrated[y_source == 0], CONFORMAL_ALPHA
        )
        q1 = finite_conformal_quantile(
            1.0 - oof_calibrated[y_source == 1], CONFORMAL_ALPHA
        )

        base_model.fit(x_source, y_source)
        validation_raw_all = base_model.predict_proba(validation_features_all)[:, 1]
        validation_calibrated_all = apply_platt(calibrator, validation_raw_all)

        source_knn = knn_distances(x_source, x_source, self_query=True)
        validation_knn_all = knn_distances(
            x_source, validation_features_all, self_query=False
        )
        source_mahal = oof_mahalanobis(x_source)
        mahal_model = LedoitWolf().fit(x_source)
        validation_mahal_all = squared_mahalanobis(
            mahal_model, validation_features_all
        )

        confidence_thresholds = {
            coverage: float(np.quantile(confidence_source, 1.0 - coverage))
            for coverage in SOURCE_COVERAGES
        }
        knn_thresholds = {
            coverage: float(np.quantile(source_knn, coverage))
            for coverage in SOURCE_COVERAGES
        }
        mahal_thresholds = {
            coverage: float(np.quantile(source_mahal, coverage))
            for coverage in SOURCE_COVERAGES
        }

        for target_index, target_name in enumerate(camera_order):
            target_indices = np.flatnonzero(
                validation["camera_labels"] == target_index
            )
            y_target = validation["pathology_labels"][target_indices]
            raw_target = validation_raw_all[target_indices]
            calibrated_target = validation_calibrated_all[target_indices]
            predictions = (calibrated_target >= decision_threshold).astype(np.uint8)
            confidence = np.abs(calibrated_target - decision_threshold)
            knn_target = validation_knn_all[target_indices]
            mahal_target = validation_mahal_all[target_indices]

            for local_index, validation_index in enumerate(target_indices):
                row: dict[str, object] = {
                    "training_camera": source_name,
                    "validation_camera": target_name,
                    "record_id": validation["record_ids"][validation_index],
                    "label": int(y_target[local_index]),
                    "prediction": int(predictions[local_index]),
                    "raw_probability": float(raw_target[local_index]),
                    "calibrated_probability": float(calibrated_target[local_index]),
                    "confidence_margin": float(confidence[local_index]),
                    "knn_distance": float(knn_target[local_index]),
                    "mahalanobis_distance": float(mahal_target[local_index]),
                }
                for coverage in SOURCE_COVERAGES:
                    key = int(round(coverage * 100))
                    row[f"confidence_threshold_source_{key}"] = confidence_thresholds[coverage]
                    row[f"knn_threshold_source_{key}"] = knn_thresholds[coverage]
                    row[f"mahalanobis_threshold_source_{key}"] = mahal_thresholds[coverage]
                prediction_rows.append(row)

            for calibration_name, probabilities in (
                ("raw_logistic", raw_target),
                ("source_oof_platt", calibrated_target),
            ):
                calibration_rows.append(
                    {
                        "training_camera": source_name,
                        "validation_camera": target_name,
                        "calibration": calibration_name,
                        "records": len(y_target),
                        "prevalence": float(y_target.mean()),
                        "brier": float(brier_score_loss(y_target, probabilities)),
                        "log_loss": float(log_loss(y_target, probabilities, labels=[0, 1])),
                        "ece_10": expected_calibration_error(
                            y_target, probabilities, bins=10
                        ),
                    }
                )

            prediction_sets = mondrian_prediction_sets(
                calibrated_target, q0=q0, q1=q1
            )
            conformal_rows.append(
                {
                    "training_camera": source_name,
                    "validation_camera": target_name,
                    "method": "class_conditional_cross_conformal_diagnostic",
                    "alpha": CONFORMAL_ALPHA,
                    "q0": q0,
                    "q1": q1,
                    **conformal_metrics(y_target, prediction_sets),
                }
            )

            baseline = selective_metrics(
                y_target, predictions, np.ones(len(y_target), dtype=bool)
            )
            selective_rows.append(
                {
                    "training_camera": source_name,
                    "validation_camera": target_name,
                    "selection_method": "none",
                    "source_target_coverage": 1.0,
                    **baseline,
                }
            )
            for coverage in SOURCE_COVERAGES:
                acceptors = {
                    "confidence": confidence >= confidence_thresholds[coverage],
                    "knn_ood": knn_target <= knn_thresholds[coverage],
                    "mahalanobis_ood": mahal_target <= mahal_thresholds[coverage],
                    "confidence_and_knn": (
                        (confidence >= confidence_thresholds[coverage])
                        & (knn_target <= knn_thresholds[coverage])
                    ),
                }
                for method, accepted in acceptors.items():
                    selective_rows.append(
                        {
                            "training_camera": source_name,
                            "validation_camera": target_name,
                            "selection_method": method,
                            "source_target_coverage": coverage,
                            **selective_metrics(y_target, predictions, accepted),
                        }
                    )

            for method, ranking_confidence in (
                ("confidence", confidence),
                ("knn_ood", -knn_target),
                ("mahalanobis_ood", -mahal_target),
            ):
                aurc_rows.append(
                    {
                        "training_camera": source_name,
                        "validation_camera": target_name,
                        "selection_method": method,
                        "aurc": risk_coverage_auc(
                            y_target, predictions, ranking_confidence
                        ),
                    }
                )

        if source_name != "pooled":
            source_index = camera_order.index(source_name)
            target_index = 1 - source_index
            source_validation_indices = np.flatnonzero(
                validation["camera_labels"] == source_index
            )
            target_validation_indices = np.flatnonzero(
                validation["camera_labels"] == target_index
            )
            for method, source_scores, target_scores, train_scores in (
                (
                    "knn_cosine_distance",
                    validation_knn_all[source_validation_indices],
                    validation_knn_all[target_validation_indices],
                    source_knn,
                ),
                (
                    "ledoit_wolf_mahalanobis",
                    validation_mahal_all[source_validation_indices],
                    validation_mahal_all[target_validation_indices],
                    source_mahal,
                ),
            ):
                labels = np.concatenate(
                    [
                        np.zeros(len(source_scores), dtype=np.uint8),
                        np.ones(len(target_scores), dtype=np.uint8),
                    ]
                )
                scores = np.concatenate([source_scores, target_scores])
                lower, upper = bootstrap_ood_auc(
                    source_scores, target_scores, rng=rng
                )
                threshold95 = float(np.quantile(train_scores, 0.95))
                ood_rows.append(
                    {
                        "training_camera": source_name,
                        "other_camera": camera_order[target_index],
                        "method": method,
                        "camera_shift_auroc": float(roc_auc_score(labels, scores)),
                        "camera_shift_auroc_ci95_lower": lower,
                        "camera_shift_auroc_ci95_upper": upper,
                        "source_validation_flag_rate_at_train_p95": float(
                            (source_scores > threshold95).mean()
                        ),
                        "other_camera_flag_rate_at_train_p95": float(
                            (target_scores > threshold95).mean()
                        ),
                    }
                )

    pd.DataFrame(tuning_rows).to_csv(
        OUTPUT_DIR / "source_only_tuning.csv", index=False
    )
    pd.DataFrame(calibration_rows).to_csv(
        OUTPUT_DIR / "calibration_metrics.csv", index=False
    )
    pd.DataFrame(conformal_rows).to_csv(
        OUTPUT_DIR / "conformal_diagnostics.csv", index=False
    )
    pd.DataFrame(ood_rows).to_csv(
        OUTPUT_DIR / "ood_camera_shift_metrics.csv", index=False
    )
    pd.DataFrame(selective_rows).to_csv(
        OUTPUT_DIR / "selective_risk_coverage.csv", index=False
    )
    pd.DataFrame(aurc_rows).to_csv(
        OUTPUT_DIR / "aurc_metrics.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        OUTPUT_DIR / "validation_trust_scores.csv", index=False
    )

    summary = {
        "status": "PASS",
        "dataset": "PALM",
        "task": "pathologic_myopia",
        "encoder": "frozen DINOv2-S/14",
        "preprocessing": "TRACE geometry-only at 448 px",
        "calibration": "Platt scaling fitted on source-only out-of-fold probabilities",
        "ood_methods": ["5-NN cosine distance", "Ledoit-Wolf Mahalanobis"],
        "conformal_status": (
            "class-conditional cross-conformal diagnostic only; ordinary "
            "exchangeability guarantee is not claimed under camera shift"
        ),
        "source_target_coverages": list(SOURCE_COVERAGES),
        "limitations": [
            "Canon CR-2 has only 50 training and 56 validation records.",
            "Camera and pathology prevalence are associated.",
            "Patient identifiers are unavailable in the local PALM manifest.",
            "OOD here means the other known PALM camera, not arbitrary clinical OOD.",
            "Validation labels are used only for evaluation, not threshold fitting.",
        ],
        "test_records_accessed": 0,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nOOD:\n" + pd.DataFrame(ood_rows).to_string(index=False))
    print("\nConformal:\n" + pd.DataFrame(conformal_rows).to_string(index=False))
    print("\nAURC:\n" + pd.DataFrame(aurc_rows).to_string(index=False))


if __name__ == "__main__":
    main()
