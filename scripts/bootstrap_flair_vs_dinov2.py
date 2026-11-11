"""Paired hierarchical comparison of base-selected FLAIR and DINOv2 heads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKSPACE / "outputs" / "encoder_comparison_v1"
BOOTSTRAP_REPLICATES = 2000
SEED = 20261111
PRIMARY_HEAD = "logistic"


def load_scores(directory: Path) -> dict[str, np.ndarray | list[str]]:
    with np.load(directory / "novel_episode_query_scores.npz", allow_pickle=False) as p:
        return {
            "labels": p["labels"].astype(str),
            "k": p["k"].astype(int),
            "replicates": p["replicates"].astype(int),
            "protocols": p["support_protocols"].astype(str),
            "heads": p["heads"].astype(str),
            "scores": p["scores"].astype(np.float32),
            "validation_labels": p["validation_labels"].astype(np.uint8),
            "class_order": p["class_order"].astype(str).tolist(),
            "validation_record_ids": p["validation_record_ids"].astype(str),
        }


def score_grid(
    bundle: dict[str, np.ndarray | list[str]],
    protocol: str,
    k: int,
    label: str,
) -> np.ndarray:
    selected = (
        (bundle["protocols"] == protocol)
        & (bundle["k"] == k)
        & (bundle["labels"] == label)
        & (bundle["heads"] == PRIMARY_HEAD)
    )
    replicates = bundle["replicates"][selected]
    order = np.argsort(replicates)
    if not np.array_equal(replicates[order], np.arange(50)):
        raise RuntimeError(f"Incomplete score grid: {protocol}, K={k}, {label}")
    return bundle["scores"][selected][order]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dino = load_scores(WORKSPACE / "outputs" / "dinov2_fewshot_v1")
    flair = load_scores(WORKSPACE / "outputs" / "flair_fewshot_v1")
    for field in ("class_order", "validation_record_ids"):
        left = dino[field]
        right = flair[field]
        if isinstance(left, list):
            equal = left == right
        else:
            equal = np.array_equal(left, right)
        if not equal:
            raise RuntimeError(f"DINOv2/FLAIR mismatch in {field}")
    if not np.array_equal(dino["validation_labels"], flair["validation_labels"]):
        raise RuntimeError("DINOv2/FLAIR validation labels differ")

    labels = sorted(set(dino["labels"].tolist()))
    class_index = {label: i for i, label in enumerate(dino["class_order"])}
    protocols_and_k = sorted(
        set(zip(dino["protocols"], dino["k"], strict=True))
    )
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for protocol, k in protocols_and_k:
        dino_grids = {label: score_grid(dino, protocol, k, label) for label in labels}
        flair_grids = {label: score_grid(flair, protocol, k, label) for label in labels}
        dino_distribution = np.empty(BOOTSTRAP_REPLICATES)
        flair_distribution = np.empty(BOOTSTRAP_REPLICATES)
        for bootstrap_index in range(BOOTSTRAP_REPLICATES):
            sampled_labels = rng.choice(labels, size=len(labels), replace=True)
            dino_values: list[float] = []
            flair_values: list[float] = []
            for label in sampled_labels:
                y = dino["validation_labels"][:, class_index[label]]
                positive = np.flatnonzero(y == 1)
                negative = np.flatnonzero(y == 0)
                query = np.concatenate(
                    [
                        rng.choice(positive, size=len(positive), replace=True),
                        rng.choice(negative, size=len(negative), replace=True),
                    ]
                )
                support_replicate = int(rng.integers(0, 50))
                sampled_y = y[query]
                dino_values.append(
                    average_precision_score(
                        sampled_y, dino_grids[label][support_replicate, query]
                    )
                )
                flair_values.append(
                    average_precision_score(
                        sampled_y, flair_grids[label][support_replicate, query]
                    )
                )
            dino_distribution[bootstrap_index] = np.mean(dino_values)
            flair_distribution[bootstrap_index] = np.mean(flair_values)
        difference = flair_distribution - dino_distribution
        rows.append(
            {
                "support_protocol": protocol,
                "k": k,
                "head": PRIMARY_HEAD,
                "dinov2_macro_auprc_mean": float(dino_distribution.mean()),
                "flair_macro_auprc_mean": float(flair_distribution.mean()),
                "flair_minus_dinov2_mean": float(difference.mean()),
                "difference_ci95_lower": float(np.quantile(difference, 0.025)),
                "difference_ci95_upper": float(np.quantile(difference, 0.975)),
                "ci_excludes_zero": bool(
                    np.quantile(difference, 0.025) > 0
                    or np.quantile(difference, 0.975) < 0
                ),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )
        print(f"comparison_complete protocol={protocol} k={k}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "flair_vs_dinov2_bootstrap.csv", index=False)
    summary = {
        "status": "PASS",
        "comparison": "FLAIR minus DINOv2-S with each encoder's base-selected logistic head",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "paired_on": [
            "novel label",
            "support episode",
            "validation query resample",
        ],
        "contamination_warning": (
            "FLAIR's official pretraining assembly lists RFMiD. This comparison "
            "cannot establish clean external generalisation or isolate the benefit "
            "of contrastive pretraining. FLAIR is also larger than DINOv2-S."
        ),
        "test_records_accessed": 0,
    }
    (OUTPUT_DIR / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n" + results.to_string(index=False))


if __name__ == "__main__":
    main()

