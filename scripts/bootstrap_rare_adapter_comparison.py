"""Paired hierarchical bootstrap: TRACE rare adapter versus frozen DINOv2-S.

The comparison is paired on novel label, support episode, and validation query
resample.  It never reads the protected RFMiD test split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


WORKSPACE = Path(__file__).resolve().parents[1]
BASELINE_DIR = WORKSPACE / "outputs" / "dinov2_fewshot_v1"
ADAPTER_DIR = WORKSPACE / "outputs" / "trace_rare_adapter_fewshot_v1"
OUTPUT_DIR = ADAPTER_DIR
BOOTSTRAP_REPLICATES = 5000
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
    bundle: dict[str, np.ndarray | list[str]], protocol: str, k: int, label: str
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


def point_macro_auprc(
    grids: dict[str, np.ndarray],
    labels: list[str],
    validation_labels: np.ndarray,
    class_index: dict[str, int],
) -> float:
    values: list[float] = []
    for label in labels:
        y = validation_labels[:, class_index[label]]
        values.extend(
            average_precision_score(y, episode_scores) for episode_scores in grids[label]
        )
    return float(np.mean(values))


def main() -> None:
    baseline = load_scores(BASELINE_DIR)
    adapter = load_scores(ADAPTER_DIR)
    if adapter["class_order"] != baseline["class_order"]:
        raise RuntimeError("Class-order mismatch")
    if not np.array_equal(adapter["validation_record_ids"], baseline["validation_record_ids"]):
        raise RuntimeError("Validation-record mismatch")
    if not np.array_equal(adapter["validation_labels"], baseline["validation_labels"]):
        raise RuntimeError("Validation-label mismatch")
    for field in ("labels", "k", "replicates", "protocols", "heads"):
        if not np.array_equal(adapter[field], baseline[field]):
            raise RuntimeError(f"Episode-grid mismatch: {field}")

    labels = sorted(set(baseline["labels"].tolist()))
    class_index = {label: i for i, label in enumerate(baseline["class_order"])}
    protocol_k = sorted(set(zip(baseline["protocols"], baseline["k"], strict=True)))
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []

    for protocol, k in protocol_k:
        baseline_grids = {
            label: score_grid(baseline, protocol, int(k), label) for label in labels
        }
        adapter_grids = {
            label: score_grid(adapter, protocol, int(k), label) for label in labels
        }
        baseline_point = point_macro_auprc(
            baseline_grids, labels, baseline["validation_labels"], class_index
        )
        adapter_point = point_macro_auprc(
            adapter_grids, labels, baseline["validation_labels"], class_index
        )

        differences = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
        for bootstrap_index in range(BOOTSTRAP_REPLICATES):
            sampled_labels = rng.choice(labels, size=len(labels), replace=True)
            label_differences: list[float] = []
            for label in sampled_labels:
                y = baseline["validation_labels"][:, class_index[label]]
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
                adapter_ap = average_precision_score(
                    sampled_y, adapter_grids[label][support_replicate, query]
                )
                baseline_ap = average_precision_score(
                    sampled_y, baseline_grids[label][support_replicate, query]
                )
                label_differences.append(adapter_ap - baseline_ap)
            differences[bootstrap_index] = np.mean(label_differences)

        lower = float(np.quantile(differences, 0.025))
        upper = float(np.quantile(differences, 0.975))
        rows.append(
            {
                "support_protocol": str(protocol),
                "k": int(k),
                "head": PRIMARY_HEAD,
                "baseline_macro_auprc": baseline_point,
                "adapter_macro_auprc": adapter_point,
                "adapter_minus_baseline": adapter_point - baseline_point,
                "bootstrap_difference_mean": float(differences.mean()),
                "difference_ci95_lower": lower,
                "difference_ci95_upper": upper,
                "adapter_significantly_better": bool(lower > 0),
                "adapter_significantly_worse": bool(upper < 0),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )
        print(f"comparison_complete protocol={protocol} k={k}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "paired_vs_dinov2_bootstrap.csv", index=False)

    primary = results[
        (results["support_protocol"] == "abundant_negatives")
        & (results["k"].isin([3, 5]))
    ]
    selected = bool(
        (primary["adapter_minus_baseline"] > 0).any()
        and (primary["adapter_significantly_better"]).any()
        and not (primary["adapter_significantly_worse"]).any()
    )
    decision = "SELECT_ADAPTER" if selected else "REJECT_ADAPTER_KEEP_DINOV2_S"
    summary = {
        "status": "PASS",
        "decision": decision,
        "selection_gate": (
            "Select only with a positive point estimate and paired 95% CI above zero "
            "at abundant-negative K=3 or K=5, without significant harm at either."
        ),
        "primary_head": PRIMARY_HEAD,
        "head_selection": "locked independently using base labels only",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "paired_on": ["novel label", "support episode", "validation query resample"],
        "novel_labels_used_for_adapter_training": [],
        "protected_test_records_accessed": 0,
        "limitations": [
            "This is development-validation evidence, not a protected-test estimate.",
            "Eleven novel labels and 50 support episodes per K still yield wide uncertainty.",
            "A rejected adapter checkpoint is retained for reproducibility, not deployment.",
        ],
    }
    (OUTPUT_DIR / "paired_vs_dinov2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n" + results.to_string(index=False))


if __name__ == "__main__":
    main()
