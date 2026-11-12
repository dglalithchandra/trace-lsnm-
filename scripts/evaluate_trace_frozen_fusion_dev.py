"""Development-only equal-weight frozen feature fusion on fixed RFMiD episodes.

This screens a cheap model-result hypothesis without opening protected test data.
The candidate is deliberately fixed before scoring: equal unit-normalised DINOv2-L
and RETFound-DINOv2 embeddings, concatenated and unit-normalised again.
"""

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


def _load_pair(split: str):
    prefix = "train" if split == "train" else "validation"
    root = ROOT / "data/features"
    left_path = root / f"rfmid_{prefix}_dinov2_vitl14_geometry448_v1.npz"
    right_path = root / f"rfmid_{prefix}_retfound_dinov2_meh_geometry448_v1.npz"
    left = load_feature_split(left_path, split)
    right = load_feature_split(right_path, split)
    if not (
        np.array_equal(left.record_ids, right.record_ids)
        and np.array_equal(left.labels, right.labels)
        and left.class_order == right.class_order
    ):
        raise RuntimeError("Frozen feature files are not paired by record and label")
    fused = l2_normalize(
        np.concatenate([l2_normalize(left.embeddings), l2_normalize(right.embeddings)], axis=1)
    )
    return left, fused, {"dinov2_l": sha256_file(left_path), "retfound": sha256_file(right_path)}


def main() -> None:
    episode_path = ROOT / "data/protocols/trace_fewshot_v1/episode_plan.csv"
    episodes = pd.read_csv(episode_path)
    episodes = episodes.loc[episodes.k.eq(3)]
    if len(episodes) != 550 or episodes.label.nunique() != 11:
        raise RuntimeError("Fixed K=3 development episode plan changed")
    lock_path = ROOT / "outputs/dinov2_fewshot_v1/locked_head_hyperparameters.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    choice = lock["locked"]["abundant_negatives"]["k3"]
    head = choice["selected_head_on_base_labels"]
    parameter = choice[head]["parameter"]
    if (head, parameter) != ("logistic", 1.0):
        raise RuntimeError("Fixed base-selected head changed")

    train, train_x, train_hashes = _load_pair("train")
    query, query_x, query_hashes = _load_pair("validation")
    if not (len(train.record_ids) == 1917 and len(query.record_ids) == 639):
        raise RuntimeError("Unexpected development feature count")
    if train.class_order != query.class_order:
        raise RuntimeError("Train and validation classes differ")
    record_index = {record_id: index for index, record_id in enumerate(train.record_ids)}
    rows = []
    for number, episode in enumerate(episodes.itertuples(index=False), 1):
        class_index = train.class_order.index(str(episode.label))
        positives, _ = parse_indices(episode, record_index)
        negatives = np.flatnonzero(train.labels[:, class_index] == 0)
        truth = query.labels[:, class_index]
        scores = score_head(head, train_x, query_x, positives, negatives, parameter, int(episode.seed))
        rows.append({
            "model": "equal_frozen_fusion",
            "episode_id": episode.episode_id,
            "label": str(episode.label),
            "replicate": int(episode.replicate),
            "auprc": float(average_precision_score(truth, scores)),
            "auroc": float(roc_auc_score(truth, scores)),
            "query_positives": int(truth.sum()),
            "query_prevalence": float(truth.mean()),
        })
        if number % 100 == 0 or number == len(episodes):
            print(f"fusion: {number}/{len(episodes)}", flush=True)
    output = ROOT / "outputs/trace_frozen_fusion_dev"
    output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output / "episode_metrics.csv", index=False)
    per_label = results.groupby("label", as_index=False).agg(
        auprc=("auprc", "mean"), auroc=("auroc", "mean"),
        query_positives=("query_positives", "first"),
        query_prevalence=("query_prevalence", "first"),
    )
    per_label.to_csv(output / "per_label_results.csv", index=False)
    summary = {
        "status": "DEVELOPMENT_ONLY_NOT_IN_LOCKED_FINAL_TEST",
        "candidate": "equal concatenation of independently unit-normalised DINOv2-L and RETFound-DINOv2 frozen features",
        "head": head, "parameter": parameter,
        "macro_auprc": float(per_label.auprc.mean()),
        "macro_auroc": float(per_label.auroc.mean()),
        "episodes": len(episodes), "labels": len(per_label),
        "train_records": len(train.record_ids), "validation_queries": len(query.record_ids),
        "episode_plan_sha256": sha256_file(episode_path),
        "head_lock_sha256": sha256_file(lock_path),
        "feature_hashes": {"train": train_hashes, "validation": query_hashes},
        "protected_test_records_accessed": 0,
        "limitation": "Previously reused validation queries; this screen cannot support a confirmatory improvement claim.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"macro_auprc": summary["macro_auprc"], "macro_auroc": summary["macro_auroc"]}))


if __name__ == "__main__":
    main()
