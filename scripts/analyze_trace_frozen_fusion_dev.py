"""Paired development-only comparison for the fixed frozen fusion screen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def paired_interval(rows: pd.DataFrame, left: str, right: str) -> tuple[float, float, float]:
    paired = rows.loc[rows.model.isin([left, right])].pivot(
        index=["label", "episode_id"], columns="model", values="auprc"
    )
    if paired.isna().any().any() or len(paired) != 550:
        raise RuntimeError("Missing or unpaired episode rows")
    by_label = [
        (group[left] - group[right]).to_numpy()
        for _, group in paired.groupby(level="label", sort=True)
    ]
    point = float(np.mean([values.mean() for values in by_label]))
    rng = np.random.default_rng(20261112)
    draws = np.empty(5000, dtype=np.float64)
    for step in range(len(draws)):
        label_indices = rng.integers(0, len(by_label), size=len(by_label))
        draws[step] = np.mean([
            by_label[index][rng.integers(0, len(by_label[index]), size=len(by_label[index]))].mean()
            for index in label_indices
        ])
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high)


def main() -> None:
    output = ROOT / "outputs/trace_frozen_fusion_dev"
    fusion = pd.read_csv(output / "episode_metrics.csv")
    existing = pd.read_csv(ROOT / "outputs/trace_ssl_v1_k3_dev/episode_metrics.csv")
    if set(fusion.model) != {"equal_frozen_fusion"}:
        raise RuntimeError("Unexpected fusion model rows")
    rows = pd.concat([fusion, existing], ignore_index=True)
    references = ["frozen_dinov2_l", "frozen_retfound", "plain_crop_matched_epoch5", "full_field_epoch5"]
    comparisons = {}
    for reference in references:
        point, low, high = paired_interval(rows, "equal_frozen_fusion", reference)
        comparisons[reference] = {"delta_macro_auprc": point, "ci95_fixed_queries": [low, high]}
    by_label = rows.loc[rows.model.isin(["equal_frozen_fusion", "frozen_dinov2_l", "frozen_retfound"])].groupby(
        ["label", "model"], as_index=False
    ).auprc.mean().pivot(index="label", columns="model", values="auprc")
    by_label["fusion_minus_dinov2_l"] = by_label["equal_frozen_fusion"] - by_label["frozen_dinov2_l"]
    by_label.to_csv(output / "paired_per_label.csv")
    result = {
        "status": "DEVELOPMENT_ONLY",
        "comparisons": comparisons,
        "labels_fusion_above_dinov2_l": int((by_label.fusion_minus_dinov2_l > 0).sum()),
        "labels_total": len(by_label),
        "limitation": "Bootstraps labels and support draws, holding reused validation queries fixed; no protected test is used.",
    }
    (output / "paired_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
