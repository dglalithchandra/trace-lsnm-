"""Paired K=3/K=5 development evaluation for TRACE source-balance-v1."""

from __future__ import annotations

import argparse
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


def paired_hierarchical_interval(
    rows: pd.DataFrame, left: str, right: str, k: int
) -> dict[str, object]:
    subset = rows.loc[rows.k.eq(k) & rows.model.isin([left, right])]
    paired = subset.pivot(index=["label", "episode_id"], columns="model", values="auprc")
    labels = sorted(subset.label.unique())
    expected = len(labels) * 50
    if paired.isna().any().any() or len(paired) != expected:
        raise RuntimeError(f"Missing or unpaired K={k} episode rows")
    by_label = [
        (group[left] - group[right]).to_numpy()
        for _, group in paired.groupby(level="label", sort=True)
    ]
    point = float(np.mean([values.mean() for values in by_label]))
    rng = np.random.default_rng(20261112 + k)
    draws = np.empty(5000, dtype=np.float64)
    for step in range(len(draws)):
        label_indices = rng.integers(0, len(by_label), size=len(by_label))
        draws[step] = np.mean([
            by_label[index][rng.integers(0, len(by_label[index]), size=len(by_label[index]))].mean()
            for index in label_indices
        ])
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "delta_macro_auprc": point,
        "ci95_labels_then_support_episodes": [float(low), float(high)],
        "labels": len(labels),
        "episodes_per_label": 50,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-tag", required=True)
    args = parser.parse_args()
    feature_dir = ROOT / "data/features"
    paths = {
        "a0_frozen_dinov2_s": {
            split: feature_dir / f"rfmid_{split}_dinov2_vits14_geometry_v1.npz"
            for split in ("train", "validation")
        },
        "a1_core_ssl": {
            split: feature_dir / f"rfmid_{split}_trace_source_balance_{args.candidate_tag}.npz"
            for split in ("train", "validation")
        },
    }
    episode_path = ROOT / "data/protocols/trace_fewshot_v1/episode_plan.csv"
    episodes = pd.read_csv(episode_path)
    episodes = episodes.loc[episodes.k.isin([3, 5])].copy()
    counts = episodes.groupby("k").size().to_dict()
    if counts != {3: 550, 5: 550} or episodes.label.nunique() != 11:
        raise RuntimeError("Frozen K=3/K=5 episode plan changed")

    rows: list[dict[str, object]] = []
    reference: tuple[np.ndarray, ...] | None = None
    feature_hashes: dict[str, dict[str, str]] = {}
    for model, model_paths in paths.items():
        train = load_feature_split(model_paths["train"], "train")
        query = load_feature_split(model_paths["validation"], "validation")
        if len(train.record_ids) != 1917 or len(query.record_ids) != 639:
            raise RuntimeError("Unexpected RFMiD development record count")
        if train.class_order != query.class_order:
            raise RuntimeError("Train and validation class orders differ")
        pairing = (train.record_ids, train.labels, query.record_ids, query.labels)
        if reference is None:
            reference = tuple(item.copy() for item in pairing)
        elif not all(np.array_equal(a, b) for a, b in zip(reference, pairing, strict=True)):
            raise RuntimeError("Candidate and baseline are not paired by IDs and labels")

        train_x = l2_normalize(train.embeddings)
        query_x = l2_normalize(query.embeddings)
        train_index = {record_id: index for index, record_id in enumerate(train.record_ids)}
        for number, episode in enumerate(episodes.itertuples(index=False), 1):
            class_index = train.class_order.index(str(episode.label))
            positives, _ = parse_indices(episode, train_index)
            negatives = np.flatnonzero(train.labels[:, class_index] == 0)
            truth = query.labels[:, class_index]
            scores = score_head(
                "logistic", train_x, query_x, positives, negatives, 1.0, int(episode.seed)
            )
            prevalence = float(truth.mean())
            ap = float(average_precision_score(truth, scores))
            rows.append({
                "model": model,
                "episode_id": episode.episode_id,
                "label": str(episode.label),
                "k": int(episode.k),
                "replicate": int(episode.replicate),
                "auprc": ap,
                "auroc": float(roc_auc_score(truth, scores)),
                "query_positives": int(truth.sum()),
                "query_prevalence": prevalence,
                "ap_divided_by_prevalence": ap / prevalence,
            })
            if number % 250 == 0 or number == len(episodes):
                print(f"{model}: {number}/{len(episodes)}", flush=True)
        feature_hashes[model] = {
            split: sha256_file(path) for split, path in model_paths.items()
        }

    output = ROOT / "outputs/trace_source_balance_v1" / args.candidate_tag
    output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output / "episode_metrics.csv", index=False)
    per_label = results.groupby(["model", "k", "label"], as_index=False).agg(
        auprc=("auprc", "mean"),
        auroc=("auroc", "mean"),
        ap_divided_by_prevalence=("ap_divided_by_prevalence", "mean"),
        query_positives=("query_positives", "first"),
        query_prevalence=("query_prevalence", "first"),
    )
    per_label.to_csv(output / "per_label_results.csv", index=False)
    macro = per_label.groupby(["model", "k"], as_index=False).agg(
        macro_auprc=("auprc", "mean"),
        macro_auroc=("auroc", "mean"),
        macro_ap_divided_by_prevalence=("ap_divided_by_prevalence", "mean"),
    )
    macro.to_csv(output / "macro_results.csv", index=False)

    intervals = {
        f"k{k}": paired_hierarchical_interval(results, "a1_core_ssl", "a0_frozen_dinov2_s", k)
        for k in (3, 5)
    }
    paired_labels = per_label.pivot(index=["k", "label"], columns="model", values=["auprc", "auroc"])
    paired_labels.columns = [f"{metric}_{model}" for metric, model in paired_labels.columns]
    paired_labels = paired_labels.reset_index()
    paired_labels["delta_auprc"] = paired_labels.auprc_a1_core_ssl - paired_labels.auprc_a0_frozen_dinov2_s
    paired_labels["delta_auroc"] = paired_labels.auroc_a1_core_ssl - paired_labels.auroc_a0_frozen_dinov2_s
    paired_labels.to_csv(output / "paired_per_label.csv", index=False)

    k3_delta = intervals["k3"]["delta_macro_auprc"]
    k5_delta = intervals["k5"]["delta_macro_auprc"]
    worst_label = float(paired_labels.delta_auprc.min())
    macro_pivot = macro.pivot(index="k", columns="model", values="macro_auroc")
    worst_macro_auroc_delta = float(
        (macro_pivot.a1_core_ssl - macro_pivot.a0_frozen_dinov2_s).min()
    )
    gate = {
        "k3_delta_at_least_0_02": bool(k3_delta >= 0.02),
        "k5_delta_at_least_0_02": bool(k5_delta >= 0.02),
        "macro_auroc_drop_no_more_than_0_01": bool(worst_macro_auroc_delta >= -0.01),
        "no_label_auprc_harmed_more_than_0_10": bool(worst_label >= -0.10),
    }
    gate["advance_to_replication"] = all(gate.values())
    summary = {
        "status": "DEVELOPMENT_ONLY_NOT_PROTECTED_TEST",
        "candidate_tag": args.candidate_tag,
        "head": "logistic",
        "c": 1.0,
        "support_protocol": "abundant_negatives",
        "macro_results": macro.to_dict("records"),
        "paired_intervals": intervals,
        "gate": gate,
        "worst_per_label_delta_auprc": worst_label,
        "worst_macro_auroc_delta": worst_macro_auroc_delta,
        "feature_hashes": feature_hashes,
        "episode_plan_sha256": sha256_file(episode_path),
        "protected_test_records_accessed": 0,
        "limitations": [
            "One training seed.",
            "The same 639 validation images are reused across support episodes.",
            "Patient identifiers are unavailable, so patient independence cannot be verified.",
            "The interval resamples labels and support episodes but holds query images fixed.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(macro.to_string(index=False))
    print(json.dumps({"paired_intervals": intervals, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
