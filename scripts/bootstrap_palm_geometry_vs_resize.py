"""Paired PALM comparison of TRACE geometry against direct resize-only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_palm_camera_transfer import (
    BOOTSTRAP_REPLICATES,
    point_metrics,
)


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKSPACE / "outputs" / "palm_camera_transfer_v1"
SEED = 20261114


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name] for name in bundle.files}


def main() -> None:
    geometry = load(OUTPUT_DIR / "camera_transfer_scores.npz")
    resize = load(
        WORKSPACE
        / "outputs"
        / "palm_camera_transfer_resize_only_v1"
        / "camera_transfer_scores.npz"
    )
    if set(geometry) != set(resize):
        raise RuntimeError("Geometry and resize score bundles have different fields")
    for name in geometry:
        if name.startswith(("labels__", "record_ids__")) and not np.array_equal(
            geometry[name], resize[name]
        ):
            raise RuntimeError(f"Validation alignment mismatch: {name}")

    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    score_keys = sorted(name for name in geometry if name.startswith("scores__"))
    for score_key in score_keys:
        _, source, target = score_key.split("__")
        labels = geometry[f"labels__{target}"].astype(np.uint8)
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        geometry_scores = geometry[score_key]
        resize_scores = resize[score_key]
        geometry_threshold = float(geometry[f"threshold__{source}"][0])
        resize_threshold = float(resize[f"threshold__{source}"][0])
        metric_names = (
            "auprc",
            "auroc",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
        )
        distributions = {
            metric: np.empty(BOOTSTRAP_REPLICATES) for metric in metric_names
        }
        geometry_point = point_metrics(labels, geometry_scores, geometry_threshold)
        resize_point = point_metrics(labels, resize_scores, resize_threshold)
        for replicate in range(BOOTSTRAP_REPLICATES):
            query = np.concatenate(
                [
                    rng.choice(positive, len(positive), replace=True),
                    rng.choice(negative, len(negative), replace=True),
                ]
            )
            geometry_metrics = point_metrics(
                labels[query], geometry_scores[query], geometry_threshold
            )
            resize_metrics = point_metrics(
                labels[query], resize_scores[query], resize_threshold
            )
            for metric in metric_names:
                distributions[metric][replicate] = (
                    geometry_metrics[metric] - resize_metrics[metric]
                )
        for metric in metric_names:
            lower = float(np.quantile(distributions[metric], 0.025))
            upper = float(np.quantile(distributions[metric], 0.975))
            rows.append(
                {
                    "training_camera_key": source,
                    "validation_camera_key": target,
                    "metric": metric,
                    "geometry_value": geometry_point[metric],
                    "resize_only_value": resize_point[metric],
                    "geometry_minus_resize": (
                        geometry_point[metric] - resize_point[metric]
                    ),
                    "difference_ci95_lower": lower,
                    "difference_ci95_upper": upper,
                    "ci_excludes_zero": bool(lower > 0 or upper < 0),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
        print(f"ablation_complete source={source} target={target}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "geometry_vs_resize_paired_bootstrap.csv", index=False)
    summary = {
        "status": "PASS",
        "comparison": "TRACE geometry minus direct square resize-only",
        "paired_on": "identical PALM validation images with class-stratified resampling",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "interpretation_rule": (
            "Geometry earns a disease-transfer claim only for metrics whose paired "
            "95% interval excludes zero without an important opposing metric loss."
        ),
        "limitations": [
            "This is a two-known-camera development ablation, not unseen-camera evidence.",
            "Canon CR-2 validation contains only seven negatives.",
            "Each preprocessing pipeline selected its own C and threshold using training data only.",
        ],
        "test_records_accessed": 0,
    }
    (OUTPUT_DIR / "geometry_vs_resize_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n" + results.to_string(index=False))


if __name__ == "__main__":
    main()
