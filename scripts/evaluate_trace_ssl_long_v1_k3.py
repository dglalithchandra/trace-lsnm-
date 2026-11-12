"""Development-only matched K=3 evaluation of the epoch-20 SSL continuations."""

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

from scripts.analyze_trace_frozen_fusion_dev import paired_interval
from scripts.run_dinov2_fewshot_benchmark import parse_indices, score_head
from trace_models.dinov2 import sha256_file
from trace_protocols import l2_normalize, load_feature_split

MODES = ("plain_crop_matched", "full_field")


def main() -> None:
    protocol = ROOT / "data/protocols/trace_fewshot_v1/episode_plan.csv"
    episodes = pd.read_csv(protocol)
    episodes = episodes.loc[episodes.k.eq(3)]
    if len(episodes) != 550 or episodes.label.nunique() != 11:
        raise RuntimeError("Frozen development episode plan changed")
    lock = json.loads((ROOT / "outputs/dinov2_fewshot_v1/locked_head_hyperparameters.json").read_text(encoding="utf-8"))
    choice = lock["locked"]["abundant_negatives"]["k3"]
    head = choice["selected_head_on_base_labels"]
    parameter = choice[head]["parameter"]
    if (head, parameter) != ("logistic", 1.0):
        raise RuntimeError("Fixed base-selected head changed")
    reference = None
    rows = []
    feature_hashes = {}
    for mode in MODES:
        paths = {
            split: ROOT / "data/features" / f"rfmid_{split}_trace_ssl_{mode}_epoch20.npz"
            for split in ("train", "validation")
        }
        train = load_feature_split(paths["train"], "train")
        query = load_feature_split(paths["validation"], "validation")
        if len(train.record_ids) != 1917 or len(query.record_ids) != 639 or train.class_order != query.class_order:
            raise RuntimeError("Unexpected development feature schema")
        pairing = (train.record_ids, train.labels, query.record_ids, query.labels)
        if reference is None:
            reference = tuple(item.copy() for item in pairing)
        elif not all(np.array_equal(a, b) for a, b in zip(reference, pairing, strict=True)):
            raise RuntimeError("Epoch-20 methods are not paired by ID and label")
        train_x, query_x = l2_normalize(train.embeddings), l2_normalize(query.embeddings)
        train_index = {record_id: index for index, record_id in enumerate(train.record_ids)}
        for number, episode in enumerate(episodes.itertuples(index=False), 1):
            label = str(episode.label)
            class_index = train.class_order.index(label)
            positives, _ = parse_indices(episode, train_index)
            negatives = np.flatnonzero(train.labels[:, class_index] == 0)
            truth = query.labels[:, class_index]
            scores = score_head(head, train_x, query_x, positives, negatives, parameter, int(episode.seed))
            rows.append({
                "model": f"{mode}_epoch20", "episode_id": episode.episode_id,
                "label": label, "replicate": int(episode.replicate),
                "auprc": float(average_precision_score(truth, scores)),
                "auroc": float(roc_auc_score(truth, scores)),
                "query_positives": int(truth.sum()),
                "query_prevalence": float(truth.mean()),
            })
            if number % 100 == 0 or number == len(episodes):
                print(f"{mode} epoch 20: {number}/{len(episodes)}", flush=True)
        feature_hashes[mode] = {split: sha256_file(path) for split, path in paths.items()}
    output = ROOT / "outputs/trace_ssl_long_v1_k3_dev"
    output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output / "episode_metrics.csv", index=False)
    per_label = results.groupby(["model", "label"], as_index=False).agg(
        auprc=("auprc", "mean"), auroc=("auroc", "mean"),
        query_positives=("query_positives", "first"), query_prevalence=("query_prevalence", "first"),
    )
    per_label.to_csv(output / "per_label_results.csv", index=False)
    macro = per_label.groupby("model", as_index=False).agg(macro_auprc=("auprc", "mean"), macro_auroc=("auroc", "mean"))
    macro.to_csv(output / "macro_results.csv", index=False)
    older = pd.read_csv(ROOT / "outputs/trace_ssl_v1_k3_dev/episode_metrics.csv")
    both = pd.concat([results, older], ignore_index=True)
    comparisons = {}
    for left, right in (
        ("full_field_epoch20", "plain_crop_matched_epoch20"),
        ("full_field_epoch20", "frozen_dinov2_l"),
        ("plain_crop_matched_epoch20", "frozen_dinov2_l"),
        ("full_field_epoch20", "full_field_epoch5"),
        ("plain_crop_matched_epoch20", "plain_crop_matched_epoch5"),
    ):
        delta, low, high = paired_interval(both, left, right)
        comparisons[f"{left}_minus_{right}"] = {"delta_macro_auprc": delta, "ci95_fixed_queries": [low, high]}
    summary = {
        "status": "DEVELOPMENT_ONLY_NOT_IN_LOCKED_FINAL_TEST",
        "epoch": 20, "head": head, "parameter": parameter,
        "episodes_per_model": len(episodes), "validation_queries": 639,
        "feature_hashes": feature_hashes,
        "episode_plan_sha256": sha256_file(protocol),
        "macro_results": macro.to_dict("records"), "comparisons": comparisons,
        "protected_test_records_accessed": 0,
        "limitation": "One seed, reused validation queries; intervals hold query images fixed and do not model patient dependence.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(macro.to_string(index=False))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
