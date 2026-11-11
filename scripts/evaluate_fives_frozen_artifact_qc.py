"""Fresh real-image proxy validation of frozen TRACE artifact QC candidates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.qc_features import extract_qc_features, qc_feature_names


SEED = 20261111
BOOTSTRAPS = 2_000
FEATURE_PATH = WORKSPACE / "data/features/fives_artifact_qc_features_v1.npz"
OUTPUT = WORKSPACE / "outputs/fives_frozen_artifact_qc_v1"


def wilson(successes: int, total: int) -> dict[str, float]:
    if total <= 0:
        return {"lower_95": float("nan"), "upper_95": float("nan")}
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"lower_95": float(center - half), "upper_95": float(center + half)}


def extract_features(frame: pd.DataFrame) -> None:
    if FEATURE_PATH.exists():
        with np.load(FEATURE_PATH, allow_pickle=False) as payload:
            if payload["record_ids"].astype(str).tolist() != frame["record_id"].tolist():
                raise RuntimeError("existing_fives_qc_feature_order_mismatch")
        return
    partial = FEATURE_PATH.with_suffix(".partial.npz")
    names = qc_feature_names()
    features = np.zeros((len(frame), len(names)), dtype=np.float32)
    start = 0
    if partial.exists():
        with np.load(partial, allow_pickle=False) as payload:
            start = int(payload["completed"][0])
            if payload["record_ids"].astype(str).tolist() != frame["record_id"].tolist()[:start]:
                raise RuntimeError("partial_fives_qc_feature_order_mismatch")
            features[:start] = payload["features"]
        print(f"fives_qc_resume completed={start}", flush=True)
    pipeline = TracePreprocessor(PreprocessingConfig(apply_photometric_correction=False))
    for index in range(start, len(frame)):
        row = frame.iloc[index]
        with Image.open(WORKSPACE / row["image_path"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        result = pipeline.process(rgb, image_id=str(row["record_id"]))
        if (
            result.status == ProcessingStatus.REJECT
            or result.geometry_only_image is None
            or result.retinal_mask is None
        ):
            raise RuntimeError(f"fives_qc_geometry_rejected:{row['record_id']}")
        features[index] = extract_qc_features(
            result.geometry_only_image, result.retinal_mask
        )
        completed = index + 1
        if completed % 100 == 0 or completed == len(frame):
            np.savez_compressed(
                partial,
                completed=np.asarray([completed], dtype=np.int32),
                record_ids=np.asarray(frame["record_id"].tolist()[:completed], dtype=np.str_),
                features=features[:completed],
                feature_names=np.asarray(names, dtype=np.str_),
            )
            print(f"fives_qc_progress {completed}/{len(frame)}", flush=True)
    np.savez_compressed(
        FEATURE_PATH,
        record_ids=np.asarray(frame["record_id"].tolist(), dtype=np.str_),
        features=features,
        feature_names=np.asarray(names, dtype=np.str_),
        quality_blur_good=frame["quality_blur_good"].to_numpy(dtype=np.uint8),
        quality_illumination_colour_good=frame[
            "quality_illumination_colour_good"
        ].to_numpy(dtype=np.uint8),
        quality_low_contrast_good=frame["quality_low_contrast_good"].to_numpy(dtype=np.uint8),
    )
    partial.unlink(missing_ok=True)


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = scores >= threshold
    positive = labels == 1
    negative = ~positive
    sensitivity = int(np.sum(predicted & positive)) / int(positive.sum())
    specificity = int(np.sum(~predicted & negative)) / int(negative.sum())
    return {
        "records": int(len(labels)),
        "positive_poor_quality": int(positive.sum()),
        "negative_good_quality": int(negative.sum()),
        "prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "flag_rate": float(predicted.mean()),
    }


def bootstrap(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(SEED)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    values = {name: [] for name in ("auroc", "auprc", "sensitivity", "specificity")}
    for _ in range(BOOTSTRAPS):
        query = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        block = classification_metrics(labels[query], scores[query], threshold)
        for name in values:
            values[name].append(float(block[name]))
    return {
        name: {
            "lower_95": float(np.quantile(samples, 0.025)),
            "upper_95": float(np.quantile(samples, 0.975)),
        }
        for name, samples in values.items()
    }


def main() -> None:
    manifest = pd.read_csv(WORKSPACE / "data/manifests/fives_manifest.csv")
    frame = manifest[manifest["is_primary_analysis_record"].astype(bool)].copy()
    frame = frame.sort_values("record_id").reset_index(drop=True)
    if len(frame) != 794:
        raise RuntimeError(f"unexpected_fives_primary_count:{len(frame)}")
    extract_features(frame)

    bundle = joblib.load(WORKSPACE / "models/artifact_qc_v1/artifact_qc_v1.joblib")
    metadata = bundle["metadata"]
    if bundle["feature_names"] != list(qc_feature_names()):
        raise RuntimeError("fives_qc_feature_order_mismatch")
    with np.load(FEATURE_PATH, allow_pickle=False) as payload:
        features = payload["features"].astype(np.float32)
        blur_poor = 1 - payload["quality_blur_good"].astype(np.uint8)
        illumination_poor = 1 - payload["quality_illumination_colour_good"].astype(np.uint8)
        contrast_poor = 1 - payload["quality_low_contrast_good"].astype(np.uint8)

    resolution_model = bundle["detectors"]["resolution_loss"]
    field_model = bundle["detectors"]["narrow_field_of_view"]
    resolution_scores = resolution_model.predict_proba(features)[:, 1]
    field_scores = field_model.predict_proba(features)[:, 1]
    resolution_warn = float(metadata["resolution_loss"]["warn_threshold"])
    resolution_reject = float(metadata["resolution_loss"]["reject_threshold"])
    field_warn = float(metadata["narrow_field_of_view"]["warn_threshold"])
    field_reject = float(metadata["narrow_field_of_view"]["reject_threshold"])

    primary = classification_metrics(blur_poor, resolution_scores, resolution_warn)
    primary_ci = bootstrap(blur_poor, resolution_scores, resolution_warn)
    reject_metrics = classification_metrics(blur_poor, resolution_scores, resolution_reject)
    combined_warn = (resolution_scores >= resolution_warn) | (field_scores >= field_warn)
    combined_reject = (resolution_scores >= resolution_reject) | (field_scores >= field_reject)
    good_blur = blur_poor == 0

    proxy_rows = []
    for name, labels in (
        ("blur_poor", blur_poor),
        ("illumination_colour_poor", illumination_poor),
        ("low_contrast_poor", contrast_poor),
    ):
        proxy_rows.append(
            {
                "proxy_label": name,
                "positives": int(labels.sum()),
                "resolution_detector_auroc": float(roc_auc_score(labels, resolution_scores)),
                "resolution_detector_auprc": float(average_precision_score(labels, resolution_scores)),
                "field_detector_auroc": float(roc_auc_score(labels, field_scores)),
                "field_detector_auprc": float(average_precision_score(labels, field_scores)),
            }
        )

    pass_auroc = primary_ci["auroc"]["lower_95"] >= 0.75
    pass_sensitivity = primary_ci["sensitivity"]["lower_95"] >= 0.80
    pass_specificity = primary_ci["specificity"]["lower_95"] >= 0.80
    selected = pass_auroc and pass_sensitivity and pass_specificity
    result = {
        "status": "PASS",
        "protocol": "frozen no-retuning FIVES real-image proxy validation",
        "quality_label_source": "published automatic quality algorithms; not expert grading",
        "primary_target": "FIVES quality_blur_good == 0",
        "resolution_warn_metrics": primary,
        "resolution_warn_bootstrap_ci95": primary_ci,
        "resolution_reject_metrics": reject_metrics,
        "partial_module_external_behavior": {
            "combined_warn_rate_all": float(combined_warn.mean()),
            "combined_reject_rate_all": float(combined_reject.mean()),
            "combined_accept_rate_on_blur_good": float((~combined_warn[good_blur]).mean()),
            "field_warn_rate_all_without_ground_truth": float((field_scores >= field_warn).mean()),
        },
        "predeclared_resolution_gate": {
            "auroc_lower_ci_at_least_0_75": pass_auroc,
            "sensitivity_lower_ci_at_least_0_80": pass_sensitivity,
            "specificity_lower_ci_at_least_0_80": pass_specificity,
        },
        "resolution_detector_decision": (
            "selected_as_real_proxy_candidate" if selected else "not_selected"
        ),
        "field_detector_decision": "not_selectable_no_matching_fives_ground_truth",
        "limitations": [
            "FIVES quality labels are automatic-algorithm outputs, not clinician labels.",
            "Patient identifiers and camera metadata are unavailable.",
            "The dataset is not a new-camera-labelled generalisation test.",
            "Image-level bootstrap intervals do not account for unknown patient clustering.",
        ],
        "public_fives_records_accessed": len(frame),
        "protected_trace_test_records_accessed": 0,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(proxy_rows).to_csv(OUTPUT / "proxy_label_metrics.csv", index=False)
    pd.DataFrame(
        {
            "record_id": frame["record_id"],
            "blur_poor": blur_poor,
            "resolution_score": resolution_scores,
            "field_score": field_scores,
            "combined_warn": combined_warn.astype(np.uint8),
            "combined_reject": combined_reject.astype(np.uint8),
        }
    ).to_csv(OUTPUT / "per_record_scores.csv", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
