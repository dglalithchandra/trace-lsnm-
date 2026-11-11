"""Paired hierarchical RETFound comparisons against generic DINOv2 controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKSPACE / "outputs" / "encoder_comparison_v1"
BOOTSTRAP_REPLICATES = 2000
SEED = 20261115
MODEL_DIRECTORIES = {
    "retfound": WORKSPACE / "outputs" / "retfound_fewshot_v1",
    "dinov2_l": WORKSPACE / "outputs" / "dinov2_l_fewshot_v1",
    "dinov2_s": WORKSPACE / "outputs" / "dinov2_fewshot_v1",
}
COMPARISONS = (("retfound", "dinov2_l"), ("retfound", "dinov2_s"))


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


def load_selected_heads(directory: Path) -> dict[tuple[str, int], str]:
    lock = json.loads(
        (directory / "locked_head_hyperparameters.json").read_text(encoding="utf-8")
    )
    return {
        (protocol, int(k_key.removeprefix("k"))): settings[
            "selected_head_on_base_labels"
        ]
        for protocol, by_k in lock["locked"].items()
        for k_key, settings in by_k.items()
    }


def score_grid(
    bundle: dict[str, np.ndarray | list[str]],
    protocol: str,
    k: int,
    label: str,
    head: str,
) -> np.ndarray:
    selected = (
        (bundle["protocols"] == protocol)
        & (bundle["k"] == k)
        & (bundle["labels"] == label)
        & (bundle["heads"] == head)
    )
    replicates = bundle["replicates"][selected]
    order = np.argsort(replicates)
    if not np.array_equal(replicates[order], np.arange(50)):
        raise RuntimeError(f"Incomplete score grid: {protocol}, K={k}, {label}, {head}")
    return bundle["scores"][selected][order]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundles = {
        name: load_scores(directory) for name, directory in MODEL_DIRECTORIES.items()
    }
    selected_heads = {
        name: load_selected_heads(directory)
        for name, directory in MODEL_DIRECTORIES.items()
    }
    reference = bundles["retfound"]
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
    for comparison_mode in ("base_selected_head", "fixed_logistic"):
        for protocol, k in protocols_and_k:
            heads = {
                model: (
                    selected_heads[model][(protocol, int(k))]
                    if comparison_mode == "base_selected_head"
                    else "logistic"
                )
                for model in bundles
            }
            grids = {
                model: {
                    label: score_grid(bundle, protocol, k, label, heads[model])
                    for label in labels
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
                            rng.choice(positive, len(positive), replace=True),
                            rng.choice(negative, len(negative), replace=True),
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
                        "comparison_mode": comparison_mode,
                        "support_protocol": protocol,
                        "k": int(k),
                        "left_encoder": left,
                        "left_head": heads[left],
                        "right_encoder": right,
                        "right_head": heads[right],
                        "left_macro_auprc_mean": float(distributions[left].mean()),
                        "right_macro_auprc_mean": float(distributions[right].mean()),
                        "left_minus_right_mean": float(difference.mean()),
                        "difference_ci95_lower": lower,
                        "difference_ci95_upper": upper,
                        "ci_excludes_zero": bool(lower > 0 or upper < 0),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    }
                )
            print(
                f"comparison_complete mode={comparison_mode} protocol={protocol} k={k}",
                flush=True,
            )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "retfound_pairwise_bootstrap.csv", index=False)
    summary = {
        "status": "PASS",
        "primary_comparison": "RETFound-DINOv2 MEH versus size-matched DINOv2-L",
        "secondary_comparison": "RETFound-DINOv2 MEH versus DINOv2-S",
        "comparison_modes": ["base_selected_head", "fixed_logistic"],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "paired_on": ["novel label", "support episode", "validation query resample"],
        "model_size_control": (
            "RETFound and DINOv2-L share the same 304,368,640-parameter "
            "ViT-L/14 architecture and 448-pixel evaluation resolution."
        ),
        "pretraining_disclosure": (
            "The official RETFound model card declares pretraining on part of "
            "AlzEye. RFMiD is not listed; absence of overlap is not proven."
        ),
        "test_records_accessed": 0,
    }
    (OUTPUT_DIR / "retfound_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n" + results.to_string(index=False))


if __name__ == "__main__":
    main()
