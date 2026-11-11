"""Score fixed RFMiD v1 support episodes on RFMiD 2.0 validation queries."""

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

from trace_models.dinov2 import sha256_file
from trace_protocols import l2_normalize, load_feature_split
from scripts.run_dinov2_fewshot_benchmark import parse_indices, score_head

LABELS = ("CRS", "MS", "RT")
FEATURES = {
    "dinov2_s": ("rfmid_train_dinov2_vits14_geometry_v1.npz", "dinov2_fewshot_v1"),
    "dinov2_l": ("rfmid_train_dinov2_vitl14_geometry448_v1.npz", "dinov2_l_fewshot_v1"),
    "retfound": ("rfmid_train_retfound_dinov2_meh_geometry448_v1.npz", "retfound_fewshot_v1"),
}


def bootstrap_interval(labels: np.ndarray, scores: np.ndarray, seed: int) -> dict[str, list[float]]:
    """Stratified query bootstrap, conditional on the fixed support ensemble."""
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    ap, auc = [], []
    for _ in range(1000):
        sampled = np.concatenate([
            rng.choice(positives, len(positives), replace=True),
            rng.choice(negatives, len(negatives), replace=True),
        ])
        ap.append(average_precision_score(labels[sampled], scores[sampled]))
        auc.append(roc_auc_score(labels[sampled], scores[sampled]))
    return {
        "auprc": np.quantile(ap, [0.025, 0.975]).tolist(),
        "auroc": np.quantile(auc, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=tuple(FEATURES))
    args = parser.parse_args()
    train_file, lock_dir = FEATURES[args.encoder]
    train_path = ROOT / "data/features" / train_file
    query_path = ROOT / "data/features" / f"rfmid2_validation_{args.encoder}_geometry448_v1.npz"
    train = load_feature_split(train_path, "train")
    query = load_feature_split(query_path, "validation")
    if tuple(query.class_order) != LABELS or len(query.record_ids) != 177:
        raise RuntimeError("Unexpected RFMiD 2.0 query feature schema")
    if set(query.record_ids) & set(train.record_ids):
        raise RuntimeError("Support/query record ID overlap")
    mapping = pd.read_csv(ROOT / "data/protocols/trace_bench_v2/rfmid2_v1_label_mapping_draft.csv")
    eligible = set(mapping.loc[mapping.development_query_eligible_5_positive_minimum, "normalized_code"])
    if eligible != set(LABELS):
        raise RuntimeError(f"Eligibility changed: {eligible}")
    plan_path = ROOT / "data/protocols/trace_fewshot_v1/episode_plan.csv"
    episodes = pd.read_csv(plan_path)
    episodes = episodes[episodes.label.isin(LABELS)]
    if len(episodes) != 3 * 3 * 50:
        raise RuntimeError(f"Unexpected support episode count: {len(episodes)}")
    lock_path = ROOT / "outputs" / lock_dir / "locked_head_hyperparameters.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("novel_labels_used_during_tuning") != []:
        raise RuntimeError("Locked head was tuned on novel labels")
    train_features = l2_normalize(train.embeddings)
    query_features = l2_normalize(query.embeddings)
    train_index = {record_id: index for index, record_id in enumerate(train.record_ids)}
    negative_pool = {
        label: np.flatnonzero(train.labels[:, train.class_order.index(label)] == 0)
        for label in LABELS
    }
    score_groups: dict[tuple[str, int], list[np.ndarray]] = {}
    episode_rows = []
    for episode in episodes.itertuples(index=False):
        label = str(episode.label)
        k = int(episode.k)
        positives, _ = parse_indices(episode, train_index)
        setting = lock["locked"]["abundant_negatives"][f"k{k}"]
        head = setting["selected_head_on_base_labels"]
        parameter = setting[head]["parameter"]
        scores = score_head(
            head, train_features, query_features, positives,
            negative_pool[label], parameter, int(episode.seed),
        )
        query_labels = query.labels[:, query.class_order.index(label)]
        score_groups.setdefault((label, k), []).append(scores)
        episode_rows.append({
            "episode_id": episode.episode_id,
            "label": label,
            "k": k,
            "head": head,
            "parameter": parameter,
            "auprc": average_precision_score(query_labels, scores),
            "auroc": roc_auc_score(query_labels, scores),
        })
    output_dir = ROOT / "outputs/rfmid2_development_transfer_v1" / args.encoder
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_rows).to_csv(output_dir / "episode_metrics.csv", index=False)
    results = []
    mean_scores = []
    score_keys = []
    for label in LABELS:
        for k in (1, 3, 5):
            prediction = np.mean(score_groups[(label, k)], axis=0)
            truth = query.labels[:, query.class_order.index(label)]
            prevalence = float(truth.mean())
            ci = bootstrap_interval(truth, prediction, seed=20261111 + LABELS.index(label) * 10 + k)
            results.append({
                "label": label,
                "k": k,
                "query_positives": int(truth.sum()),
                "query_negatives": int(len(truth) - truth.sum()),
                "query_prevalence": prevalence,
                "auprc": float(average_precision_score(truth, prediction)),
                "auprc_normalized": float(average_precision_score(truth, prediction) / prevalence),
                "auprc_ci_low": ci["auprc"][0],
                "auprc_ci_high": ci["auprc"][1],
                "auroc": float(roc_auc_score(truth, prediction)),
                "auroc_ci_low": ci["auroc"][0],
                "auroc_ci_high": ci["auroc"][1],
                "support_replicates": len(score_groups[(label, k)]),
            })
            mean_scores.append(prediction)
            score_keys.append(f"{label}:k{k}")
    frame = pd.DataFrame(results)
    frame.to_csv(output_dir / "per_label_results.csv", index=False)
    np.savez_compressed(
        output_dir / "mean_query_scores.npz",
        keys=np.asarray(score_keys),
        scores=np.stack(mean_scores).astype(np.float32),
        query_record_ids=query.record_ids,
        query_labels=query.labels,
        class_order=np.asarray(query.class_order),
    )
    summary = {
        "status": "EXPLORATORY_DEVELOPMENT_ONLY",
        "encoder": args.encoder,
        "labels": list(LABELS),
        "k_values": [1, 3, 5],
        "query_records": len(query.record_ids),
        "episodes": len(episodes),
        "primary_support_protocol": "abundant_negatives",
        "head_selection": "locked_from_rfmid_v1_base_labels",
        "metric_unit": "ranking of fixed RFMiD 2.0 validation queries",
        "ci_scope": "stratified query bootstrap conditional on 50-support-draw mean scores; does not include source or patient uncertainty",
        "train_feature_sha256": sha256_file(train_path),
        "query_feature_sha256": sha256_file(query_path),
        "episode_plan_sha256": sha256_file(plan_path),
        "locked_head_sha256": sha256_file(lock_path),
        "protected_test_members_opened": 0,
        "macro_by_k": frame.groupby("k")[["auprc", "auroc"]].mean().to_dict("index"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
