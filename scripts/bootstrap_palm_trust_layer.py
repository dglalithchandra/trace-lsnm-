"""Paired uncertainty for PALM calibration and selective-risk diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKSPACE / "outputs" / "palm_trust_layer_v1"
BOOTSTRAP_REPLICATES = 3000
SEED = 20261117
COVERAGES = (90, 80, 70)


def acceptance(frame: pd.DataFrame, method: str, coverage: int) -> np.ndarray:
    if method == "confidence":
        return frame["confidence_margin"].to_numpy() >= frame[
            f"confidence_threshold_source_{coverage}"
        ].iloc[0]
    if method == "knn_ood":
        return frame["knn_distance"].to_numpy() <= frame[
            f"knn_threshold_source_{coverage}"
        ].iloc[0]
    if method == "mahalanobis_ood":
        return frame["mahalanobis_distance"].to_numpy() <= frame[
            f"mahalanobis_threshold_source_{coverage}"
        ].iloc[0]
    if method == "confidence_and_knn":
        return acceptance(frame, "confidence", coverage) & acceptance(
            frame, "knn_ood", coverage
        )
    raise ValueError(f"Unknown selection method: {method}")


def main() -> None:
    frame = pd.read_csv(OUTPUT_DIR / "validation_trust_scores.csv")
    rng = np.random.default_rng(SEED)
    risk_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []

    for (source, target), group in frame.groupby(
        ["training_camera", "validation_camera"], sort=True
    ):
        group = group.reset_index(drop=True)
        labels = group["label"].to_numpy(dtype=np.uint8)
        predictions = group["prediction"].to_numpy(dtype=np.uint8)
        errors = predictions != labels
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)

        raw_probability = group["raw_probability"].to_numpy(dtype=np.float64)
        calibrated_probability = group["calibrated_probability"].to_numpy(
            dtype=np.float64
        )
        brier_improvement = np.empty(BOOTSTRAP_REPLICATES)
        bootstrap_queries: list[np.ndarray] = []
        for replicate in range(BOOTSTRAP_REPLICATES):
            query = np.concatenate(
                [
                    rng.choice(positive, len(positive), replace=True),
                    rng.choice(negative, len(negative), replace=True),
                ]
            )
            bootstrap_queries.append(query)
            brier_improvement[replicate] = brier_score_loss(
                labels[query], raw_probability[query]
            ) - brier_score_loss(labels[query], calibrated_probability[query])
        calibration_rows.append(
            {
                "training_camera": source,
                "validation_camera": target,
                "raw_brier": float(brier_score_loss(labels, raw_probability)),
                "platt_brier": float(
                    brier_score_loss(labels, calibrated_probability)
                ),
                "raw_minus_platt_brier": float(
                    brier_score_loss(labels, raw_probability)
                    - brier_score_loss(labels, calibrated_probability)
                ),
                "improvement_ci95_lower": float(
                    np.quantile(brier_improvement, 0.025)
                ),
                "improvement_ci95_upper": float(
                    np.quantile(brier_improvement, 0.975)
                ),
                "ci_supports_platt_improvement": bool(
                    np.quantile(brier_improvement, 0.025) > 0
                ),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )

        baseline_risk = float(errors.mean())
        for method in (
            "confidence",
            "knn_ood",
            "mahalanobis_ood",
            "confidence_and_knn",
        ):
            for source_coverage in COVERAGES:
                accepted = acceptance(group, method, source_coverage)
                point_risk = (
                    float(errors[accepted].mean()) if accepted.any() else float("nan")
                )
                improvements: list[float] = []
                sampled_coverages: list[float] = []
                for query in bootstrap_queries:
                    sampled_accepted = accepted[query]
                    sampled_coverages.append(float(sampled_accepted.mean()))
                    if not sampled_accepted.any():
                        continue
                    baseline_sample = float(errors[query].mean())
                    selective_sample = float(
                        errors[query][sampled_accepted].mean()
                    )
                    improvements.append(baseline_sample - selective_sample)
                improvement_array = np.asarray(improvements)
                coverage_array = np.asarray(sampled_coverages)
                lower = float(np.quantile(improvement_array, 0.025))
                upper = float(np.quantile(improvement_array, 0.975))
                risk_rows.append(
                    {
                        "training_camera": source,
                        "validation_camera": target,
                        "selection_method": method,
                        "source_target_coverage": source_coverage / 100,
                        "observed_coverage": float(accepted.mean()),
                        "coverage_ci95_lower": float(
                            np.quantile(coverage_array, 0.025)
                        ),
                        "coverage_ci95_upper": float(
                            np.quantile(coverage_array, 0.975)
                        ),
                        "baseline_risk": baseline_risk,
                        "selective_risk": point_risk,
                        "baseline_minus_selective_risk": baseline_risk - point_risk,
                        "risk_improvement_ci95_lower": lower,
                        "risk_improvement_ci95_upper": upper,
                        "ci_supports_risk_improvement": bool(lower > 0),
                        "usable_coverage_at_least_half": bool(accepted.mean() >= 0.5),
                        "bootstrap_replicates": len(improvement_array),
                    }
                )

    risk_results = pd.DataFrame(risk_rows)
    calibration_results = pd.DataFrame(calibration_rows)
    risk_results.to_csv(
        OUTPUT_DIR / "selective_risk_paired_bootstrap.csv", index=False
    )
    calibration_results.to_csv(
        OUTPUT_DIR / "calibration_brier_bootstrap.csv", index=False
    )
    selected = risk_results[
        risk_results["ci_supports_risk_improvement"]
        & risk_results["usable_coverage_at_least_half"]
    ]
    summary = {
        "status": "PASS",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "risk_comparison": "baseline risk minus selective risk on paired stratified image resamples",
        "selection_rule": (
            "Retain only methods with a positive 95% risk-improvement interval "
            "and at least 50% observed validation coverage."
        ),
        "methods_meeting_selection_rule": int(len(selected)),
        "test_records_accessed": 0,
    }
    (OUTPUT_DIR / "bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nSelected rows:\n" + selected.to_string(index=False))
    print("\nCalibration:\n" + calibration_results.to_string(index=False))


if __name__ == "__main__":
    main()
