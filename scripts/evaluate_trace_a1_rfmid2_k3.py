"""Exploratory three-seed TRACE A1 check on RFMiD 2.0 validation queries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dinov2_fewshot_benchmark import parse_indices, score_head
from trace_models.dinov2 import sha256_file
from trace_protocols import l2_normalize, load_feature_split


LABELS = ("CRS", "MS", "RT")
SEEDS = (20261111, 20261112, 20261113)


def paired_query_interval(
    truth: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(truth == 1)
    negative = np.flatnonzero(truth == 0)
    draws = np.empty(5000, dtype=np.float64)
    for index in range(len(draws)):
        sample = np.concatenate(
            (
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            )
        )
        draws[index] = average_precision_score(truth[sample], candidate[sample]) - average_precision_score(
            truth[sample], baseline[sample]
        )
    return np.quantile(draws, (0.025, 0.975)).tolist()


def main() -> None:
    features = {
        "frozen_dinov2_s": (
            "rfmid_train_dinov2_vits14_geometry_v1.npz",
            "rfmid2_validation_dinov2_s_geometry448_v1.npz",
        ),
        **{
            f"trace_a1_seed_{seed}": (
                f"rfmid_train_trace_source_balance_a1_seed{seed}.npz",
                f"rfmid2_validation_trace_a1_seed{seed}.npz",
            )
            for seed in SEEDS
        },
    }
    episode_path = ROOT / "data/protocols/trace_fewshot_v1/episode_plan.csv"
    episodes = pd.read_csv(episode_path)
    episodes = episodes.loc[episodes.label.isin(LABELS) & episodes.k.eq(3)]
    if len(episodes) != 150 or episodes.label.value_counts().to_dict() != {
        label: 50 for label in LABELS
    }:
        raise RuntimeError("Frozen RFMiD 2.0 K=3 support plan changed")

    predictions: dict[tuple[str, str], np.ndarray] = {}
    truths: dict[str, np.ndarray] = {}
    per_label: list[dict[str, object]] = []
    feature_hashes: dict[str, dict[str, str]] = {}
    reference_train: tuple[np.ndarray, np.ndarray] | None = None
    reference_query: tuple[np.ndarray, np.ndarray] | None = None

    for model, (train_name, query_name) in features.items():
        train_path = ROOT / "data/features" / train_name
        query_path = ROOT / "data/features" / query_name
        train = load_feature_split(train_path, "train")
        query = load_feature_split(query_path, "validation")
        if len(train.record_ids) != 1917 or len(query.record_ids) != 177:
            raise RuntimeError(f"Unexpected feature count for {model}")
        train_pair = (train.record_ids, train.labels)
        query_pair = (query.record_ids, query.labels)
        if reference_train is None:
            reference_train = tuple(item.copy() for item in train_pair)
            reference_query = tuple(item.copy() for item in query_pair)
        elif not all(
            np.array_equal(left, right)
            for left, right in zip(reference_train + reference_query, train_pair + query_pair, strict=True)
        ):
            raise RuntimeError(f"Unpaired support/query records for {model}")

        train_x = l2_normalize(train.embeddings)
        query_x = l2_normalize(query.embeddings)
        train_index = {record_id: index for index, record_id in enumerate(train.record_ids)}
        for label in LABELS:
            class_index = train.class_order.index(label)
            truth = query.labels[:, query.class_order.index(label)]
            truths[label] = truth
            negatives = np.flatnonzero(train.labels[:, class_index] == 0)
            score_draws = []
            for episode in episodes.loc[episodes.label.eq(label)].itertuples(index=False):
                positives, _ = parse_indices(episode, train_index)
                score_draws.append(
                    score_head(
                        "logistic",
                        train_x,
                        query_x,
                        positives,
                        negatives,
                        1.0,
                        int(episode.seed),
                    )
                )
            mean_score = np.mean(score_draws, axis=0)
            predictions[(model, label)] = mean_score
            per_label.append(
                {
                    "model": model,
                    "label": label,
                    "query_positives": int(truth.sum()),
                    "query_negatives": int(len(truth) - truth.sum()),
                    "auprc": float(average_precision_score(truth, mean_score)),
                    "auroc": float(roc_auc_score(truth, mean_score)),
                }
            )
            print(f"{model} {label}: AUPRC {per_label[-1]['auprc']:.4f}", flush=True)
        feature_hashes[model] = {"train": sha256_file(train_path), "query": sha256_file(query_path)}

    comparisons = []
    for seed in SEEDS:
        model = f"trace_a1_seed_{seed}"
        for label in LABELS:
            candidate = predictions[(model, label)]
            baseline = predictions[("frozen_dinov2_s", label)]
            truth = truths[label]
            delta = float(
                average_precision_score(truth, candidate) - average_precision_score(truth, baseline)
            )
            interval = paired_query_interval(truth, candidate, baseline, seed + LABELS.index(label))
            comparisons.append(
                {
                    "seed": seed,
                    "label": label,
                    "delta_auprc": delta,
                    "query_bootstrap_low": interval[0],
                    "query_bootstrap_high": interval[1],
                }
            )

    per_label_frame = pd.DataFrame(per_label)
    comparison_frame = pd.DataFrame(comparisons)
    macro = per_label_frame.groupby("model", as_index=False).agg(
        macro_auprc=("auprc", "mean"), macro_auroc=("auroc", "mean")
    )
    seed_deltas = comparison_frame.groupby("seed").delta_auprc.mean()
    output = ROOT / "outputs/trace_source_balance_v1/a1_rfmid2_k3_dev"
    output.mkdir(parents=True, exist_ok=True)
    per_label_frame.to_csv(output / "per_label.csv", index=False)
    comparison_frame.to_csv(output / "paired_comparisons.csv", index=False)
    macro.to_csv(output / "macro.csv", index=False)
    summary = {
        "status": "EXPLORATORY_DEVELOPMENT_ONLY",
        "query_cohort": "RFMiD_2.0_validation_177",
        "labels": list(LABELS),
        "query_positives": {label: int(truths[label].sum()) for label in LABELS},
        "support": "RFMiD_v1_train_fixed_50_draws_K3_abundant_negatives",
        "macro_results": macro.to_dict("records"),
        "a1_minus_frozen_macro_auprc_delta_by_seed": {
            str(seed): float(seed_deltas.loc[seed]) for seed in SEEDS
        },
        "mean_seed_macro_auprc_delta": float(seed_deltas.mean()),
        "positive_macro_delta_seeds": int((seed_deltas > 0).sum()),
        "feature_sha256": feature_hashes,
        "episode_plan_sha256": sha256_file(episode_path),
        "protected_test_members_opened": 0,
        "limitations": [
            "Only three mapped labels with 6-8 positive validation queries each.",
            "RFMiD 2.0 validation has already been used for development.",
            "Patient identifiers and verified per-image device metadata are unavailable.",
            "Intervals are conditional on fixed support ensembles and resample query images within class.",
            "This analysis cannot establish camera generalization or fresh external confirmation.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(macro.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
