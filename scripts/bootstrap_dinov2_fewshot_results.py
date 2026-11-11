"""Paired hierarchical bootstrap over novel labels and validation queries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKSPACE / "outputs" / "dinov2_fewshot_v1"
BOOTSTRAP_REPLICATES = 2000
SEED = 20261111
HEADS = ("prototype", "knn", "logistic")


def interval(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(values.mean()),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def main() -> None:
    with np.load(OUTPUT_DIR / "novel_episode_query_scores.npz", allow_pickle=False) as payload:
        episode_labels = payload["labels"].astype(str)
        episode_k = payload["k"].astype(int)
        episode_replicates = payload["replicates"].astype(int)
        episode_protocols = payload["support_protocols"].astype(str)
        episode_heads = payload["heads"].astype(str)
        episode_scores = payload["scores"].astype(np.float32)
        validation_labels = payload["validation_labels"].astype(np.uint8)
        class_order = payload["class_order"].astype(str).tolist()

    labels = sorted(set(episode_labels.tolist()))
    class_index = {label: index for index, label in enumerate(class_order)}
    rng = np.random.default_rng(SEED)
    head_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []

    protocols_and_k = sorted(set(zip(episode_protocols, episode_k, strict=True)))
    for support_protocol, k in protocols_and_k:
        score_lookup: dict[tuple[str, str, int], np.ndarray] = {}
        for label in labels:
            for head in HEADS:
                selected = (
                    (episode_protocols == support_protocol)
                    & (episode_k == k)
                    & (episode_labels == label)
                    & (episode_heads == head)
                )
                selected_replicates = episode_replicates[selected]
                order = np.argsort(selected_replicates)
                if not np.array_equal(selected_replicates[order], np.arange(50)):
                    raise RuntimeError(
                        f"Incomplete support grid for {support_protocol}, K={k}, "
                        f"{label}, {head}"
                    )
                score_lookup[(label, head, k)] = episode_scores[selected][order]
        distributions = {
            head: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
            for head in HEADS
        }
        for replicate in range(BOOTSTRAP_REPLICATES):
            sampled_labels = rng.choice(labels, size=len(labels), replace=True)
            macro_values = {head: [] for head in HEADS}
            for label in sampled_labels:
                y = validation_labels[:, class_index[label]]
                positive = np.flatnonzero(y == 1)
                negative = np.flatnonzero(y == 0)
                sampled_query = np.concatenate(
                    [
                        rng.choice(positive, size=len(positive), replace=True),
                        rng.choice(negative, size=len(negative), replace=True),
                    ]
                )
                sampled_y = y[sampled_query]
                support_replicate = int(rng.integers(0, 50))
                for head in HEADS:
                    sampled_scores = score_lookup[(label, head, k)][
                        support_replicate, sampled_query
                    ]
                    macro_values[head].append(
                        average_precision_score(sampled_y, sampled_scores)
                    )
            for head in HEADS:
                distributions[head][replicate] = float(np.mean(macro_values[head]))

        for head in HEADS:
            mean, lower, upper = interval(distributions[head])
            head_rows.append(
                {
                    "support_protocol": support_protocol,
                    "k": k,
                    "head": head,
                    "bootstrap_macro_auprc_mean": mean,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
        for left, right in (
            ("logistic", "prototype"),
            ("logistic", "knn"),
            ("knn", "prototype"),
        ):
            differences = distributions[left] - distributions[right]
            mean, lower, upper = interval(differences)
            difference_rows.append(
                {
                    "support_protocol": support_protocol,
                    "k": k,
                    "comparison": f"{left}_minus_{right}",
                    "paired_macro_auprc_difference_mean": mean,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "ci_excludes_zero": bool(lower > 0 or upper < 0),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )

    head_frame = pd.DataFrame(head_rows)
    difference_frame = pd.DataFrame(difference_rows)
    head_frame.to_csv(OUTPUT_DIR / "hierarchical_bootstrap_heads.csv", index=False)
    difference_frame.to_csv(
        OUTPUT_DIR / "hierarchical_bootstrap_differences.csv", index=False
    )
    summary = {
        "status": "PASS",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED,
        "hierarchy": [
            "sample novel labels with replacement",
            "sample one of the 50 paired support draws for each sampled label",
            "within each sampled label, resample positive and negative query images separately",
        ],
        "paired": (
            "Every compared head uses the same sampled labels and query indices."
        ),
        "support_draw_handling": (
            "One support draw is sampled within each label and shared across heads. "
            "Episode rows are not bootstrapped as independent test sets."
        ),
        "test_records_accessed": 0,
    }
    (OUTPUT_DIR / "bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nHead intervals:\n" + head_frame.to_string(index=False))
    print("\nPaired differences:\n" + difference_frame.to_string(index=False))


if __name__ == "__main__":
    main()

