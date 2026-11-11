"""Paired hierarchical comparison of DINOv2-S, DINOv2-L and FLAIR."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKSPACE / "outputs" / "encoder_comparison_v1"
BOOTSTRAP_REPLICATES = 2000
SEED = 20261112
PRIMARY_HEAD = "logistic"
MODEL_DIRECTORIES = {
    "dinov2_s": WORKSPACE / "outputs" / "dinov2_fewshot_v1",
    "dinov2_l": WORKSPACE / "outputs" / "dinov2_l_fewshot_v1",
    "flair": WORKSPACE / "outputs" / "flair_fewshot_v1",
}
COMPARISONS = (
    ("dinov2_l", "dinov2_s"),
    ("flair", "dinov2_l"),
    ("flair", "dinov2_s"),
)


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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundles = {
        name: load_scores(directory) for name, directory in MODEL_DIRECTORIES.items()
    }
    reference = bundles["dinov2_s"]
    for name, bundle in bundles.items():
        if bundle["class_order"] != reference["class_order"]:
            raise RuntimeError(f"Class-order mismatch for {name}")
        if not np.array_equal(
            bundle["validation_record_ids"], reference["validation_record_ids"]
        ):
            raise RuntimeError(f"Validation-ID mismatch for {name}")
        if not np.array_equal(
            bundle["validation_labels"], reference["validation_labels"]
        ):
            raise RuntimeError(f"Validation-label mismatch for {name}")

    labels = sorted(set(reference["labels"].tolist()))
    class_index = {label: i for i, label in enumerate(reference["class_order"])}
    protocols_and_k = sorted(
        set(zip(reference["protocols"], reference["k"], strict=True))
    )
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for protocol, k in protocols_and_k:
        grids = {
            model: {
                label: score_grid(bundle, protocol, k, label) for label in labels
            }
            for model, bundle in bundles.items()
        }
        distributions = {
            model: np.empty(BOOTSTRAP_REPLICATES) for model in bundles
        }
        for bootstrap_index in range(BOOTSTRAP_REPLICATES):
            sampled_labels = rng.choice(labels, size=len(labels), replace=True)
            values = {model: [] for model in bundles}
            for label in sampled_labels:
                y = reference["validation_labels"][:, class_index[label]]
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
                for model in bundles:
                    values[model].append(
                        average_precision_score(
                            sampled_y,
                            grids[model][label][support_replicate, query],
                        )
                    )
            for model in bundles:
                distributions[model][bootstrap_index] = np.mean(values[model])

        for left, right in COMPARISONS:
            difference = distributions[left] - distributions[right]
            lower = float(np.quantile(difference, 0.025))
            upper = float(np.quantile(difference, 0.975))
            rows.append(
                {
                    "support_protocol": protocol,
                    "k": int(k),
                    "head": PRIMARY_HEAD,
                    "left_encoder": left,
                    "right_encoder": right,
                    "left_macro_auprc_mean": float(distributions[left].mean()),
                    "right_macro_auprc_mean": float(distributions[right].mean()),
                    "left_minus_right_mean": float(difference.mean()),
                    "difference_ci95_lower": lower,
                    "difference_ci95_upper": upper,
                    "ci_excludes_zero": bool(lower > 0 or upper < 0),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
        print(f"comparison_complete protocol={protocol} k={k}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "three_encoder_pairwise_bootstrap.csv", index=False)
    summary = {
        "status": "PASS",
        "encoders": list(bundles),
        "comparisons": [list(pair) for pair in COMPARISONS],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "paired_on": ["novel label", "support episode", "validation query resample"],
        "limitations": [
            "FLAIR lists RFMiD in its pretraining assembly and is tagged possibly seen.",
            "DINOv2-L is size-matched to RETFound, not exactly to FLAIR.",
            "These are development-validation estimates, not protected-test results.",
        ],
        "test_records_accessed": 0,
    }
    (OUTPUT_DIR / "three_encoder_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n" + results.to_string(index=False))


if __name__ == "__main__":
    main()
