"""Paired hierarchical comparison of FLAIR zero-shot, cache and logistic heads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKSPACE / "outputs" / "flair_cache_v1"
BOOTSTRAP_REPLICATES = 2000
SEED = 20261111


def main() -> None:
    with np.load(OUTPUT_DIR / "novel_cache_query_scores.npz", allow_pickle=False) as p:
        cache_labels = p["labels"].astype(str)
        cache_k = p["k"].astype(int)
        cache_replicates = p["replicates"].astype(int)
        cache_protocols = p["support_protocols"].astype(str)
        cache_scores = p["scores"].astype(np.float32)
        validation_labels = p["validation_labels"].astype(np.uint8)
        class_order = p["class_order"].astype(str).tolist()
        validation_ids = p["validation_record_ids"].astype(str)
    with np.load(
        WORKSPACE / "outputs" / "flair_fewshot_v1" / "novel_episode_query_scores.npz",
        allow_pickle=False,
    ) as p:
        logistic_labels = p["labels"].astype(str)
        logistic_k = p["k"].astype(int)
        logistic_replicates = p["replicates"].astype(int)
        logistic_protocols = p["support_protocols"].astype(str)
        logistic_heads = p["heads"].astype(str)
        logistic_scores = p["scores"].astype(np.float32)
        if not np.array_equal(p["validation_record_ids"].astype(str), validation_ids):
            raise RuntimeError("FLAIR cache/logistic validation IDs differ")
        if not np.array_equal(p["validation_labels"], validation_labels):
            raise RuntimeError("FLAIR cache/logistic validation labels differ")
    with np.load(
        WORKSPACE / "outputs" / "flair_zero_shot_v1" / "selected_zero_shot_scores.npz",
        allow_pickle=False,
    ) as p:
        zero_labels = p["labels"].astype(str).tolist()
        zero_scores = p["scores"].astype(np.float32)

    labels = sorted(set(cache_labels.tolist()))
    class_index = {label: index for index, label in enumerate(class_order)}
    zero_index = {label: index for index, label in enumerate(zero_labels)}
    protocols_and_k = sorted(set(zip(cache_protocols, cache_k, strict=True)))
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []

    for protocol, k in protocols_and_k:
        cache_grid: dict[str, np.ndarray] = {}
        logistic_grid: dict[str, np.ndarray] = {}
        for label in labels:
            selected_cache = (
                (cache_protocols == protocol)
                & (cache_k == k)
                & (cache_labels == label)
            )
            order_cache = np.argsort(cache_replicates[selected_cache])
            cache_grid[label] = cache_scores[selected_cache][order_cache]
            selected_logistic = (
                (logistic_protocols == protocol)
                & (logistic_k == k)
                & (logistic_labels == label)
                & (logistic_heads == "logistic")
            )
            order_logistic = np.argsort(logistic_replicates[selected_logistic])
            logistic_grid[label] = logistic_scores[selected_logistic][order_logistic]
            if cache_grid[label].shape[0] != 50 or logistic_grid[label].shape[0] != 50:
                raise RuntimeError(f"Incomplete score grid: {protocol}, K={k}, {label}")

        distributions = {
            name: np.empty(BOOTSTRAP_REPLICATES)
            for name in ("zero_shot", "cache", "logistic")
        }
        for bootstrap_index in range(BOOTSTRAP_REPLICATES):
            sampled_labels = rng.choice(labels, size=len(labels), replace=True)
            values = {name: [] for name in distributions}
            for label in sampled_labels:
                y = validation_labels[:, class_index[label]]
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
                values["zero_shot"].append(
                    average_precision_score(
                        sampled_y, zero_scores[query, zero_index[label]]
                    )
                )
                values["cache"].append(
                    average_precision_score(
                        sampled_y, cache_grid[label][support_replicate, query]
                    )
                )
                values["logistic"].append(
                    average_precision_score(
                        sampled_y, logistic_grid[label][support_replicate, query]
                    )
                )
            for name in distributions:
                distributions[name][bootstrap_index] = np.mean(values[name])

        for left, right in (
            ("cache", "zero_shot"),
            ("cache", "logistic"),
            ("logistic", "zero_shot"),
        ):
            difference = distributions[left] - distributions[right]
            lower = float(np.quantile(difference, 0.025))
            upper = float(np.quantile(difference, 0.975))
            rows.append(
                {
                    "support_protocol": protocol,
                    "k": k,
                    "comparison": f"{left}_minus_{right}",
                    "paired_macro_auprc_difference_mean": float(difference.mean()),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "ci_excludes_zero": bool(lower > 0 or upper < 0),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
        print(f"cache_bootstrap_complete protocol={protocol} k={k}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "cache_paired_bootstrap.csv", index=False)
    summary = {
        "status": "PASS",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "paired_on": ["label", "support draw", "query resample"],
        "contamination_tag": "possibly_seen_rfmid_listed_in_pretraining_assembly",
        "test_records_accessed": 0,
    }
    (OUTPUT_DIR / "cache_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n" + results.to_string(index=False))


if __name__ == "__main__":
    main()

