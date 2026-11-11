"""Refresh generated threshold metrics after standardizing float tolerance."""

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

from scripts.evaluate_photometric_candidates_with_model import threshold_metrics
from trace_models.rfmid_dinov2 import DECISION_TOLERANCE


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def update_per_class(
    *,
    output_path: Path,
    candidates: dict[str, np.ndarray],
    labels: np.ndarray,
    baseline: np.ndarray,
    class_order: list[str],
    thresholds: list[float | None],
) -> None:
    rows: list[dict[str, object]] = []
    for candidate_name, probabilities in candidates.items():
        for index, label in enumerate(class_order):
            threshold = thresholds[index]
            positive_count = int(labels[:, index].sum())
            if threshold is None or positive_count < 5:
                continue
            truth = labels[:, index].astype(bool)
            baseline_positive = (
                baseline[:, index] >= float(threshold) - DECISION_TOLERANCE
            )
            candidate_positive = (
                probabilities[:, index]
                >= float(threshold) - DECISION_TOLERANCE
            )
            rows.append(
                {
                    "candidate": candidate_name,
                    "label": label,
                    "validation_positive": positive_count,
                    "baseline_sensitivity": float(
                        baseline_positive[truth].mean()
                    ),
                    "candidate_sensitivity": float(
                        candidate_positive[truth].mean()
                    ),
                    "sensitivity_delta": float(
                        candidate_positive[truth].mean()
                        - baseline_positive[truth].mean()
                    ),
                    "baseline_specificity": float(
                        (~baseline_positive[~truth]).mean()
                    ),
                    "candidate_specificity": float(
                        (~candidate_positive[~truth]).mean()
                    ),
                    "specificity_delta": float(
                        (~candidate_positive[~truth]).mean()
                        - (~baseline_positive[~truth]).mean()
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    bundle = torch.load(
        WORKSPACE
        / "models"
        / "rfmid_dinov2_linear_v1"
        / "rfmid_dinov2_linear_v1.pt",
        map_location="cpu",
        weights_only=False,
    )
    thresholds = bundle["thresholds"]
    baseline_payload = load_npz(
        WORKSPACE / "outputs" / "rfmid_baseline" / "baseline_predictions.npz"
    )
    labels = baseline_payload["labels"]
    baseline = baseline_payload["probabilities"]
    class_order = baseline_payload["class_order"].astype(str).tolist()

    model_dir = WORKSPACE / "outputs" / "photometric_model_evaluation"
    model_summary_path = model_dir / "model_evaluation_summary.json"
    model_summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
    model_summary["baseline"]["threshold_metrics"] = threshold_metrics(
        labels, baseline, thresholds
    )
    model_candidates: dict[str, np.ndarray] = {}
    for candidate in model_summary["candidates"]:
        safe_name = (
            f"{candidate['reference']}_strength_{candidate['strength']:.2f}"
            .replace(".", "p")
        )
        payload = load_npz(model_dir / f"{safe_name}.npz")
        probabilities = payload["probabilities"]
        candidate["threshold_metrics"] = threshold_metrics(
            labels, probabilities, thresholds
        )
        model_candidates[safe_name] = probabilities
    model_summary["threshold_decision_tolerance"] = DECISION_TOLERANCE
    model_summary_path.write_text(
        json.dumps(model_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    update_per_class(
        output_path=model_dir / "per_class_threshold_deltas.csv",
        candidates=model_candidates,
        labels=labels,
        baseline=baseline,
        class_order=class_order,
        thresholds=thresholds,
    )

    component_dir = WORKSPACE / "outputs" / "photometric_component_ablation"
    component_summary_path = component_dir / "component_ablation_summary.json"
    component_summary = json.loads(
        component_summary_path.read_text(encoding="utf-8")
    )
    component_summary["baseline"]["threshold_metrics"] = threshold_metrics(
        labels, baseline, thresholds
    )
    component_candidates: dict[str, np.ndarray] = {}
    for candidate in component_summary["components"]:
        payload = load_npz(component_dir / f"{candidate['component']}.npz")
        probabilities = payload["probabilities"]
        candidate["threshold_metrics"] = threshold_metrics(
            labels, probabilities, thresholds
        )
        component_candidates[candidate["component"]] = probabilities
    component_summary["threshold_decision_tolerance"] = DECISION_TOLERANCE
    component_summary_path.write_text(
        json.dumps(component_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    update_per_class(
        output_path=component_dir / "component_per_class_threshold_deltas.csv",
        candidates=component_candidates,
        labels=labels,
        baseline=baseline,
        class_order=class_order,
        thresholds=thresholds,
    )

    report = {
        "status": "PASS",
        "threshold_decision_tolerance": DECISION_TOLERANCE,
        "updated": [
            str(model_summary_path.relative_to(WORKSPACE)).replace("\\", "/"),
            str(component_summary_path.relative_to(WORKSPACE)).replace("\\", "/"),
        ],
        "test_records_accessed": 0,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
