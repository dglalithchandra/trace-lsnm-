"""Paired label × support-draw bootstrap for TRACE SSL K=3 development deltas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/trace_ssl_v1_k3_dev/episode_metrics.csv"
OUTPUT = ROOT / "outputs/trace_ssl_v1_k3_dev/paired_bootstrap.json"
COMPARISONS = (("full_field_epoch5", "plain_crop_matched_epoch5"), ("full_field_epoch5", "frozen_dinov2_s"), ("plain_crop_matched_epoch5", "frozen_dinov2_s"), ("full_field_epoch5", "frozen_dinov2_l"), ("plain_crop_matched_epoch5", "frozen_dinov2_l"))


def main() -> None:
    rows = pd.read_csv(INPUT)
    paired = rows.pivot(index=["label", "replicate", "episode_id"], columns="model", values="auprc")
    if paired.shape != (550, 5) or paired.isna().any().any():
        raise RuntimeError("Unpaired or incomplete episode results")
    labels = sorted(paired.index.get_level_values("label").unique())
    if len(labels) != 11:
        raise RuntimeError("Novel-label set changed")
    rng = np.random.default_rng(20261111)
    outputs = []
    for newer, older in COMPARISONS:
        difference = np.stack([(paired.loc[label, newer] - paired.loc[label, older]).to_numpy() for label in labels])
        if difference.shape != (11, 50):
            raise RuntimeError("Support replicate shape changed")
        draws = np.empty(5000, dtype=np.float64)
        for number in range(len(draws)):
            label_sample = rng.integers(0, 11, size=11)
            replicate_sample = rng.integers(0, 50, size=(11, 50))
            draws[number] = difference[label_sample[:, None], replicate_sample].mean()
        outputs.append({"newer": newer, "older": older, "observed_macro_auprc_difference": float(difference.mean()), "hierarchical_bootstrap_95_interval": np.quantile(draws, (0.025, 0.975)).tolist(), "bootstrap_fraction_above_zero": float((draws > 0).mean()), "labels_improved": int((difference.mean(axis=1) > 0).sum())})
    summary = {"status": "DEVELOPMENT_ONLY_PAIRED_BOOTSTRAP", "resamples": 5000, "unit": "labels_then_support_draws;fixed_validation_queries_not_resampled", "limitations": ["Does not account for query or patient sampling uncertainty", "Only one training seed", "Prior development use of validation episodes"], "comparisons": outputs, "protected_test_records_accessed": 0}
    OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
