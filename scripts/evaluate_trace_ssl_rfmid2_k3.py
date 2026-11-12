"""Exploratory cross-release K=3 comparison with identical support and head."""

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
FEATURES = {
    "frozen_dinov2_s": ("rfmid_train_dinov2_vits14_geometry_v1.npz", "rfmid2_validation_dinov2_s_geometry448_v1.npz"),
    "frozen_dinov2_l": ("rfmid_train_dinov2_vitl14_geometry448_v1.npz", "rfmid2_validation_dinov2_l_geometry448_v1.npz"),
    "frozen_retfound": ("rfmid_train_retfound_dinov2_meh_geometry448_v1.npz", "rfmid2_validation_retfound_geometry448_v1.npz"),
    "plain_crop_matched_epoch5": ("rfmid_train_trace_ssl_plain_crop_matched_epoch5.npz", "rfmid2_validation_trace_ssl_plain_crop_matched_epoch5.npz"),
    "full_field_epoch5": ("rfmid_train_trace_ssl_full_field_epoch5.npz", "rfmid2_validation_trace_ssl_full_field_epoch5.npz"),
}


def paired_query_interval(truth: np.ndarray, newer: np.ndarray, older: np.ndarray, seed: int) -> list[float]:
    """Stratified image bootstrap, conditional on the fixed support ensemble."""
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(truth == 1)
    negative = np.flatnonzero(truth == 0)
    draws = np.empty(2000)
    for i in range(len(draws)):
        indices = np.concatenate((rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)))
        draws[i] = average_precision_score(truth[indices], newer[indices]) - average_precision_score(truth[indices], older[indices])
    return np.quantile(draws, (0.025, 0.975)).tolist()


def main() -> None:
    plan_path = ROOT / "data/protocols/trace_fewshot_v1/episode_plan.csv"
    episodes = pd.read_csv(plan_path)
    episodes = episodes[episodes.label.isin(LABELS) & episodes.k.eq(3)]
    if len(episodes) != 150 or episodes.label.value_counts().to_dict() != {label: 50 for label in LABELS}:
        raise RuntimeError("Fixed K=3 support plan changed")
    lock_path = ROOT / "outputs/dinov2_fewshot_v1/locked_head_hyperparameters.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("novel_labels_used_during_tuning") != []:
        raise RuntimeError("Head lock used novel labels")
    choice = lock["locked"]["abundant_negatives"]["k3"]
    head = choice["selected_head_on_base_labels"]
    parameter = choice[head]["parameter"]
    if (head, parameter) != ("logistic", 1.0):
        raise RuntimeError("Frozen common head changed")
    predictions: dict[tuple[str, str], np.ndarray] = {}
    truth_by_label: dict[str, np.ndarray] = {}
    reference_ids = None
    reference_labels = None
    reference_train_ids = None
    reference_train_labels = None
    hashes = {}
    per_episode = []
    per_label = []
    for model_name, (train_name, query_name) in FEATURES.items():
        train_path = ROOT / "data/features" / train_name
        query_path = ROOT / "data/features" / query_name
        train = load_feature_split(train_path, "train")
        query = load_feature_split(query_path, "validation")
        if len(train.record_ids) != 1917 or len(query.record_ids) != 177 or tuple(query.class_order) != LABELS:
            raise RuntimeError(f"Unexpected feature split for {model_name}")
        if reference_train_ids is None:
            reference_train_ids = train.record_ids.copy()
            reference_train_labels = train.labels.copy()
        elif not (np.array_equal(reference_train_ids, train.record_ids) and np.array_equal(reference_train_labels, train.labels)):
            raise RuntimeError("Unpaired support records or labels")
        if reference_ids is None:
            reference_ids = query.record_ids.copy()
            reference_labels = query.labels.copy()
        elif not (np.array_equal(reference_ids, query.record_ids) and np.array_equal(reference_labels, query.labels)):
            raise RuntimeError("Unpaired query records or labels")
        train_x, query_x = l2_normalize(train.embeddings), l2_normalize(query.embeddings)
        train_index = {record_id: index for index, record_id in enumerate(train.record_ids)}
        for label in LABELS:
            class_index = train.class_order.index(label)
            truth = query.labels[:, query.class_order.index(label)]
            if int(truth.sum()) < 5:
                raise RuntimeError(f"Insufficient positives for {label}")
            truth_by_label[label] = truth
            negatives = np.flatnonzero(train.labels[:, class_index] == 0)
            score_draws = []
            for episode in episodes[episodes.label.eq(label)].itertuples(index=False):
                positives, _ = parse_indices(episode, train_index)
                scores = score_head(head, train_x, query_x, positives, negatives, parameter, int(episode.seed))
                score_draws.append(scores)
                per_episode.append({"model": model_name, "label": label, "episode_id": episode.episode_id, "auprc": average_precision_score(truth, scores), "auroc": roc_auc_score(truth, scores)})
            mean_scores = np.mean(score_draws, axis=0)
            predictions[(model_name, label)] = mean_scores
            per_label.append({"model": model_name, "label": label, "query_positives": int(truth.sum()), "query_negatives": int(len(truth) - truth.sum()), "auprc_ensemble": average_precision_score(truth, mean_scores), "auroc_ensemble": roc_auc_score(truth, mean_scores), "auprc_mean_episode": float(np.mean([average_precision_score(truth, score) for score in score_draws]))})
            print(f"{model_name} {label}: {per_label[-1]['auprc_ensemble']:.4f}", flush=True)
        hashes[model_name] = {"train": sha256_file(train_path), "query": sha256_file(query_path)}
    output = ROOT / "outputs/trace_ssl_rfmid2_k3_dev"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_episode).to_csv(output / "per_episode.csv", index=False)
    pd.DataFrame(per_label).to_csv(output / "per_label.csv", index=False)
    comparison_rows = []
    for label in LABELS:
        for newer, older in (("full_field_epoch5", "plain_crop_matched_epoch5"), ("full_field_epoch5", "frozen_dinov2_s"), ("full_field_epoch5", "frozen_dinov2_l"), ("plain_crop_matched_epoch5", "frozen_dinov2_s"), ("plain_crop_matched_epoch5", "frozen_dinov2_l")):
            a, b = predictions[(newer, label)], predictions[(older, label)]
            interval = paired_query_interval(truth_by_label[label], a, b, seed=20261112 + LABELS.index(label))
            comparison_rows.append({"label": label, "newer": newer, "older": older, "auprc_difference": average_precision_score(truth_by_label[label], a) - average_precision_score(truth_by_label[label], b), "query_bootstrap_low": interval[0], "query_bootstrap_high": interval[1]})
    pd.DataFrame(comparison_rows).to_csv(output / "paired_comparisons.csv", index=False)
    summary = {"status": "EXPLORATORY_DEVELOPMENT_ONLY", "description": "RFMiD-v1 training supports and RFMiD-2.0 validation queries; same fixed K=3 logistic C=1 head for all models", "query_records": 177, "labels": list(LABELS), "support_draws_per_label": 50, "query_positives": {label: int(truth_by_label[label].sum()) for label in LABELS}, "feature_sha256": hashes, "episode_plan_sha256": sha256_file(plan_path), "head_lock_sha256": sha256_file(lock_path), "ci_scope": "Stratified image bootstrap of paired 50-draw ensemble predictions, fixed support draws; no patient IDs", "protected_test_members_opened": 0, "warning": "The RFMiD 2.0 validation split is already used for development; no model-selection or camera-generalization claim follows."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(per_label).to_string(index=False))


if __name__ == "__main__":
    main()
