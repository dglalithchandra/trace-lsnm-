"""Train-only artifact QC fitting followed by frozen validation evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_models import load_rfmid_bundle, probabilities_from_embedding
from trace_models.rfmid_dinov2 import DECISION_TOLERANCE


SEED = 20261111
C_CANDIDATES = (0.01, 0.1, 1.0, 10.0)
WARN_TARGET_RECALL = 0.95
REJECT_TARGET_RECALL = 0.80
MODEL_DIR = WORKSPACE / "models" / "artifact_qc_v1"
OUTPUT_DIR = WORKSPACE / "outputs" / "artifact_qc_v1"


def load_feature_split(split: str) -> dict[str, np.ndarray]:
    path = (
        WORKSPACE / "data/features" / f"rfmid_{split}_artifact_qc_features_v1.npz"
    )
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["split"][0]) != split:
            raise RuntimeError(f"artifact_qc_split_mismatch:{split}")
        return {key: payload[key] for key in payload.files}


def make_model(c_value: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=3000,
            random_state=SEED,
        ),
    )


def threshold_for_recall(
    labels: np.ndarray, scores: np.ndarray, target_recall: float
) -> float:
    positive_scores = scores[labels == 1]
    candidates = np.unique(scores)
    valid = [
        float(value)
        for value in candidates
        if float(np.mean(positive_scores >= value)) >= target_recall
    ]
    if not valid:
        return float(np.min(scores) - 1e-12)
    return max(valid)


def wilson(successes: int, total: int) -> dict[str, float]:
    if total <= 0:
        return {"lower_95": float("nan"), "upper_95": float("nan")}
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z
        * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return {"lower_95": float(center - half), "upper_95": float(center + half)}


def status_from_scores(
    score_matrix: np.ndarray, detectors: dict[str, dict[str, object]]
) -> np.ndarray:
    names = list(detectors)
    warning = np.zeros(score_matrix.shape[:-1], dtype=bool)
    reject = np.zeros(score_matrix.shape[:-1], dtype=bool)
    for index, name in enumerate(names):
        warning |= score_matrix[..., index] >= float(detectors[name]["warn_threshold"])
        reject |= score_matrix[..., index] >= float(detectors[name]["reject_threshold"])
    return np.where(reject, "reject", np.where(warning, "warn", "accept"))


def micro_disease_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float | None],
    accepted: np.ndarray,
) -> dict[str, float | int | None]:
    selected = np.asarray([value is not None for value in thresholds])
    values = np.asarray(
        [float(value) if value is not None else 0.5 for value in thresholds]
    )
    truth = labels[accepted][:, selected].astype(bool).ravel()
    predicted = (
        probabilities[accepted][:, selected]
        >= values[selected][None, :] - DECISION_TOLERANCE
    ).ravel()
    if not len(truth):
        return {
            "records": 0,
            "coverage": 0.0,
            "micro_sensitivity": None,
            "micro_specificity": None,
        }
    tp = int(np.sum(truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    tn = int(np.sum(~truth & ~predicted))
    fp = int(np.sum(~truth & predicted))
    return {
        "records": int(accepted.sum()),
        "coverage": float(accepted.mean()),
        "micro_sensitivity": tp / (tp + fn) if tp + fn else None,
        "micro_specificity": tn / (tn + fp) if tn + fp else None,
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Training is deliberately loaded and fully processed before validation.
    train = load_feature_split("train")
    view_names = train["view_names"].astype(str).tolist()
    feature_names = train["feature_names"].astype(str).tolist()
    record_ids = train["record_ids"].astype(str)
    clean_index = view_names.index("clean")
    artifact_names = [name for name in view_names if name != "clean"]
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    detectors: dict[str, dict[str, object]] = {}
    selection_rows = []

    for artifact_name in artifact_names:
        artifact_index = view_names.index(artifact_name)
        x = np.concatenate(
            [train["features"][:, clean_index], train["features"][:, artifact_index]],
            axis=0,
        ).astype(np.float32)
        y = np.concatenate(
            [np.zeros(len(record_ids), dtype=np.uint8), np.ones(len(record_ids), dtype=np.uint8)]
        )
        groups = np.concatenate([record_ids, record_ids])
        best = None
        for c_value in C_CANDIDATES:
            probabilities = cross_val_predict(
                make_model(c_value),
                x,
                y,
                groups=groups,
                cv=folds,
                method="predict_proba",
                n_jobs=-1,
            )[:, 1]
            auprc = float(average_precision_score(y, probabilities))
            auroc = float(roc_auc_score(y, probabilities))
            selection_rows.append(
                {
                    "artifact": artifact_name,
                    "c_value": c_value,
                    "oof_auprc": auprc,
                    "oof_auroc": auroc,
                }
            )
            candidate = (auprc, -c_value, c_value, probabilities)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        assert best is not None
        selected_c = float(best[2])
        oof = best[3]
        warn_threshold = threshold_for_recall(y, oof, WARN_TARGET_RECALL)
        reject_threshold = threshold_for_recall(y, oof, REJECT_TARGET_RECALL)
        if reject_threshold < warn_threshold:
            raise RuntimeError("artifact_qc_threshold_order_failure")
        final_model = make_model(selected_c).fit(x, y)
        detectors[artifact_name] = {
            "model": final_model,
            "selected_c": selected_c,
            "warn_threshold": warn_threshold,
            "reject_threshold": reject_threshold,
            "train_oof_auprc": float(average_precision_score(y, oof)),
            "train_oof_auroc": float(roc_auc_score(y, oof)),
            "train_warn_recall": float(np.mean(oof[y == 1] >= warn_threshold)),
            "train_warn_clean_flag_rate": float(np.mean(oof[y == 0] >= warn_threshold)),
            "train_reject_recall": float(np.mean(oof[y == 1] >= reject_threshold)),
            "train_reject_clean_rate": float(np.mean(oof[y == 0] >= reject_threshold)),
        }

    # Freeze serializable train-derived objects before validation is loaded.
    serializable = {
        name: {
            key: value
            for key, value in details.items()
            if key != "model"
        }
        for name, details in detectors.items()
    }
    joblib.dump(
        {
            "detectors": {name: details["model"] for name, details in detectors.items()},
            "metadata": serializable,
            "feature_names": feature_names,
            "view_names": view_names,
            "training_record_ids": record_ids.tolist(),
            "research_only": True,
        },
        MODEL_DIR / "artifact_qc_v1.joblib",
    )
    (MODEL_DIR / "train_frozen_metadata.json").write_text(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_VALIDATION",
                "detectors": serializable,
                "feature_names": feature_names,
                "warn_target_recall": WARN_TARGET_RECALL,
                "reject_target_recall": REJECT_TARGET_RECALL,
                "validation_loaded": False,
                "external_datasets_accessed": 0,
                "protected_test_records_accessed": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(selection_rows).to_csv(
        MODEL_DIR / "train_model_selection.csv", index=False
    )

    # Frozen evaluation begins here.
    validation = load_feature_split("validation")
    if validation["feature_names"].astype(str).tolist() != feature_names:
        raise RuntimeError("artifact_qc_feature_order_mismatch")
    if validation["view_names"].astype(str).tolist() != view_names:
        raise RuntimeError("artifact_qc_view_order_mismatch")
    count = len(validation["record_ids"])
    score_tensor = np.zeros((count, len(view_names), len(artifact_names)), dtype=np.float64)
    for detector_index, name in enumerate(artifact_names):
        model = detectors[name]["model"]
        for view_index in range(len(view_names)):
            score_tensor[:, view_index, detector_index] = model.predict_proba(
                validation["features"][:, view_index]
            )[:, 1]
    statuses = status_from_scores(score_tensor, detectors)

    clean_status = statuses[:, clean_index]
    clean_accepts = int(np.sum(clean_status == "accept"))
    clean_metrics = {
        "records": count,
        "accept_rate": clean_accepts / count,
        "accept_rate_ci95": wilson(clean_accepts, count),
        "warn_rate": float(np.mean(clean_status == "warn")),
        "reject_rate": float(np.mean(clean_status == "reject")),
    }

    # Recompute frozen disease probabilities only for validation harm analysis.
    stress_path = (
        WORKSPACE / "data/features/rfmid_validation_dinov2_artifact_stress_v2.npz"
    )
    with np.load(stress_path, allow_pickle=False) as payload:
        stress_ids = payload["record_ids"].astype(str)
        clean_embeddings = payload["clean_embeddings"].astype(np.float32)
        artifact_embeddings = payload["artifact_embeddings"].astype(np.float32)
        stress_styles = payload["styles"].astype(str).tolist()
        labels = payload["labels"].astype(np.uint8)
    if stress_ids.tolist() != validation["record_ids"].astype(str).tolist():
        raise RuntimeError("artifact_qc_stress_record_order_mismatch")
    if stress_styles != artifact_names:
        raise RuntimeError("artifact_qc_stress_style_order_mismatch")
    head, bundle = load_rfmid_bundle(
        WORKSPACE / "models/rfmid_dinov2_linear_v1/rfmid_dinov2_linear_v1.pt"
    )
    clean_probabilities = probabilities_from_embedding(
        head, torch.from_numpy(clean_embeddings), bundle
    ).numpy()
    flat_artifact = artifact_embeddings.reshape(-1, artifact_embeddings.shape[-1])
    artifact_probabilities = probabilities_from_embedding(
        head, torch.from_numpy(flat_artifact), bundle
    ).numpy().reshape(count, len(artifact_names), -1)
    thresholded = np.asarray([value is not None for value in bundle["thresholds"]])
    threshold_values = np.asarray(
        [float(value) if value is not None else 0.5 for value in bundle["thresholds"]]
    )
    clean_decisions = (
        clean_probabilities[:, thresholded]
        >= threshold_values[thresholded][None, :] - DECISION_TOLERANCE
    )
    truth = labels[:, thresholded].astype(bool)

    artifact_rows = []
    severity_rows = []
    for style_index, style in enumerate(artifact_names):
        view_index = view_names.index(style)
        status = statuses[:, view_index]
        flagged = status != "accept"
        stressed_decisions = (
            artifact_probabilities[:, style_index, thresholded]
            >= threshold_values[thresholded][None, :] - DECISION_TOLERANCE
        )
        harmful = np.any(clean_decisions & ~stressed_decisions & truth, axis=1)
        harmful_detected = int(np.sum(harmful & flagged))
        harmful_total = int(harmful.sum())
        accepted = ~flagged
        all_records = np.ones(count, dtype=bool)
        unfiltered_disease = micro_disease_metrics(
            labels,
            artifact_probabilities[:, style_index],
            bundle["thresholds"],
            all_records,
        )
        accepted_disease = micro_disease_metrics(
            labels,
            artifact_probabilities[:, style_index],
            bundle["thresholds"],
            accepted,
        )
        row = {
            "artifact": style,
            "records": count,
            "flag_rate": float(flagged.mean()),
            "reject_rate": float(np.mean(status == "reject")),
            "artifact_detection_recall_ci95": wilson(int(flagged.sum()), count),
            "harmful_views": harmful_total,
            "harmful_detected": harmful_detected,
            "harmful_detection_recall": harmful_detected / harmful_total
            if harmful_total
            else None,
            "harmful_detection_recall_ci95": wilson(harmful_detected, harmful_total),
            "harmful_missed": int(np.sum(harmful & ~flagged)),
            "unfiltered_micro_sensitivity": unfiltered_disease["micro_sensitivity"],
            "accepted_micro_sensitivity": accepted_disease["micro_sensitivity"],
            "accepted_micro_specificity": accepted_disease["micro_specificity"],
            "accepted_coverage": accepted_disease["coverage"],
        }
        artifact_rows.append(row)
        style_severity = validation["severities"][:, view_index].astype(str)
        for severity in ("mild", "moderate", "severe"):
            selected = style_severity == severity
            severity_rows.append(
                {
                    "artifact": style,
                    "severity": severity,
                    "records": int(selected.sum()),
                    "flag_rate": float(flagged[selected].mean()),
                    "reject_rate": float(np.mean(status[selected] == "reject")),
                    "harmful_views": int(harmful[selected].sum()),
                    "harmful_detected": int(np.sum(harmful[selected] & flagged[selected])),
                }
            )

    pass_clean = clean_metrics["accept_rate"] >= 0.80
    pass_harm = all(
        row["harmful_detection_recall"] is not None
        and float(row["harmful_detection_recall"]) >= 0.80
        for row in artifact_rows
    )
    pass_sensitivity = all(
        row["accepted_micro_sensitivity"] is not None
        and row["unfiltered_micro_sensitivity"] is not None
        and float(row["accepted_micro_sensitivity"])
        >= float(row["unfiltered_micro_sensitivity"])
        for row in artifact_rows
    )
    selected = pass_clean and pass_harm and pass_sensitivity
    decision = "selected_for_research_qc_v1" if selected else "not_selected"

    pd.DataFrame(artifact_rows).to_csv(
        OUTPUT_DIR / "validation_artifact_metrics.csv", index=False
    )
    pd.DataFrame(severity_rows).to_csv(
        OUTPUT_DIR / "validation_severity_metrics.csv", index=False
    )
    np.savez_compressed(
        OUTPUT_DIR / "validation_scores.npz",
        record_ids=validation["record_ids"].astype(np.str_),
        view_names=np.asarray(view_names, dtype=np.str_),
        detector_names=np.asarray(artifact_names, dtype=np.str_),
        scores=score_tensor.astype(np.float32),
        statuses=statuses.astype(np.str_),
    )
    result = {
        "status": "PASS",
        "decision": decision,
        "decision_rule": {
            "clean_accept_rate_at_least": 0.80,
            "each_harmful_detection_recall_at_least": 0.80,
            "accepted_micro_sensitivity_must_not_decrease": True,
        },
        "train_only_frozen_detectors": serializable,
        "validation_clean": clean_metrics,
        "validation_artifacts": artifact_rows,
        "decision_checks": {
            "clean_acceptance_pass": pass_clean,
            "harmful_detection_pass": pass_harm,
            "accepted_sensitivity_pass": pass_sensitivity,
        },
        "limitations": [
            "Detectors are trained on controlled synthetic artifacts, not labelled real camera failures.",
            "RFMiD validation was previously used for disease-head development.",
            "Warn/reject targets are engineering choices, not clinically validated limits.",
            "A fresh real external dataset is still required after selection.",
        ],
        "external_datasets_accessed": 0,
        "protected_test_records_accessed": 0,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
