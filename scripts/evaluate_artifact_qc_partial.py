"""Post-hoc detector-level audit of the frozen Experiment 4B QC models.

This does not create new models or thresholds. The detector eligibility rule
uses train OOF metrics only, but it was specified after seeing the failed
combined validation result, so all partial-module findings are exploratory.
"""

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

from scripts.train_evaluate_artifact_qc import micro_disease_metrics, wilson
from trace_models import load_rfmid_bundle, probabilities_from_embedding
from trace_models.rfmid_dinov2 import DECISION_TOLERANCE


OUTPUT = WORKSPACE / "outputs" / "artifact_qc_v1"


def main() -> None:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    eligibility = {}
    for name, metrics in summary["train_only_frozen_detectors"].items():
        checks = {
            "train_oof_auroc_at_least_0_90": metrics["train_oof_auroc"] >= 0.90,
            "train_warn_clean_flag_rate_at_most_0_10": metrics[
                "train_warn_clean_flag_rate"
            ]
            <= 0.10,
            "train_reject_clean_rate_at_most_0_02": metrics[
                "train_reject_clean_rate"
            ]
            <= 0.02,
        }
        eligibility[name] = {"eligible": all(checks.values()), "checks": checks}
    selected_names = [name for name, block in eligibility.items() if block["eligible"]]

    with np.load(OUTPUT / "validation_scores.npz", allow_pickle=False) as payload:
        record_ids = payload["record_ids"].astype(str)
        view_names = payload["view_names"].astype(str).tolist()
        detector_names = payload["detector_names"].astype(str).tolist()
        scores = payload["scores"].astype(np.float64)
    warning = np.zeros(scores.shape[:2], dtype=bool)
    reject = np.zeros(scores.shape[:2], dtype=bool)
    for name in selected_names:
        detector_index = detector_names.index(name)
        metadata = summary["train_only_frozen_detectors"][name]
        warning |= scores[:, :, detector_index] >= float(metadata["warn_threshold"])
        reject |= scores[:, :, detector_index] >= float(metadata["reject_threshold"])
    statuses = np.where(reject, "reject", np.where(warning, "warn", "accept"))

    stress_path = (
        WORKSPACE / "data/features/rfmid_validation_dinov2_artifact_stress_v2.npz"
    )
    with np.load(stress_path, allow_pickle=False) as payload:
        if payload["record_ids"].astype(str).tolist() != record_ids.tolist():
            raise RuntimeError("partial_qc_record_order_mismatch")
        clean_embeddings = payload["clean_embeddings"].astype(np.float32)
        artifact_embeddings = payload["artifact_embeddings"].astype(np.float32)
        styles = payload["styles"].astype(str).tolist()
        labels = payload["labels"].astype(np.uint8)
    if styles != detector_names:
        raise RuntimeError("partial_qc_style_order_mismatch")
    head, bundle = load_rfmid_bundle(
        WORKSPACE / "models/rfmid_dinov2_linear_v1/rfmid_dinov2_linear_v1.pt"
    )
    clean_probabilities = probabilities_from_embedding(
        head, torch.from_numpy(clean_embeddings), bundle
    ).numpy()
    flat = artifact_embeddings.reshape(-1, artifact_embeddings.shape[-1])
    artifact_probabilities = probabilities_from_embedding(
        head, torch.from_numpy(flat), bundle
    ).numpy().reshape(len(record_ids), len(styles), -1)
    thresholded = np.asarray([value is not None for value in bundle["thresholds"]])
    threshold_values = np.asarray(
        [float(value) if value is not None else 0.5 for value in bundle["thresholds"]]
    )
    truth = labels[:, thresholded].astype(bool)
    clean_decisions = (
        clean_probabilities[:, thresholded]
        >= threshold_values[thresholded][None, :] - DECISION_TOLERANCE
    )

    clean_status = statuses[:, view_names.index("clean")]
    clean_accept = int(np.sum(clean_status == "accept"))
    clean = {
        "records": len(record_ids),
        "accept_rate": clean_accept / len(record_ids),
        "accept_rate_ci95": wilson(clean_accept, len(record_ids)),
        "warn_rate": float(np.mean(clean_status == "warn")),
        "reject_rate": float(np.mean(clean_status == "reject")),
    }
    rows = []
    for style_index, style in enumerate(styles):
        view_status = statuses[:, view_names.index(style)]
        flagged = view_status != "accept"
        stressed_decisions = (
            artifact_probabilities[:, style_index, thresholded]
            >= threshold_values[thresholded][None, :] - DECISION_TOLERANCE
        )
        harmful = np.any(clean_decisions & ~stressed_decisions & truth, axis=1)
        detected = int(np.sum(harmful & flagged))
        harmful_count = int(harmful.sum())
        accepted_metrics = micro_disease_metrics(
            labels,
            artifact_probabilities[:, style_index],
            bundle["thresholds"],
            ~flagged,
        )
        rows.append(
            {
                "artifact": style,
                "covered_by_selected_detector": style in selected_names,
                "flag_rate": float(flagged.mean()),
                "reject_rate": float(np.mean(view_status == "reject")),
                "harmful_views": harmful_count,
                "harmful_detected": detected,
                "harmful_detection_recall": detected / harmful_count
                if harmful_count
                else None,
                "harmful_detection_recall_ci95": wilson(detected, harmful_count),
                "accepted_coverage": accepted_metrics["coverage"],
                "accepted_micro_sensitivity": accepted_metrics["micro_sensitivity"],
                "accepted_micro_specificity": accepted_metrics["micro_specificity"],
            }
        )

    result = {
        "status": "PASS",
        "analysis_type": "post_hoc_exploratory_partial_module_audit",
        "selected_by_train_only_eligibility_rule": selected_names,
        "rejected_by_train_only_eligibility_rule": [
            name for name in detector_names if name not in selected_names
        ],
        "eligibility": eligibility,
        "validation_clean": clean,
        "validation_artifacts": rows,
        "decision": (
            "partial_candidate_requires_fresh_validation; full artifact QC remains not_selected"
        ),
        "why_not_confirmatory": (
            "The eligibility rule was specified after the combined validation failure was observed."
        ),
        "external_datasets_accessed": 0,
        "protected_test_records_accessed": 0,
    }
    pd.DataFrame(rows).to_csv(OUTPUT / "partial_validation_metrics.csv", index=False)
    (OUTPUT / "partial_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
