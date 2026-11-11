"""Evaluate frozen RFMiD few-shot supports on EDID target-versus-healthy queries."""

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

FEATURES = {
    "dinov2_s": ("rfmid_train_dinov2_vits14_geometry_v1.npz", "dinov2_fewshot_v1"),
    "dinov2_l": ("rfmid_train_dinov2_vitl14_geometry448_v1.npz", "dinov2_l_fewshot_v1"),
    "retfound": ("rfmid_train_retfound_dinov2_meh_geometry448_v1.npz", "retfound_fewshot_v1"),
}
LABELS = ("MS", "RP", "ODE")


def intervals(truth: np.ndarray, scores: np.ndarray, seed: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """Image-level bootstrap, conditional on fixed support ensemble."""
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(truth == 1)
    negative = np.flatnonzero(truth == 0)
    aps, aucs = [], []
    for _ in range(1000):
        chosen = np.concatenate((rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)))
        aps.append(average_precision_score(truth[chosen], scores[chosen]))
        aucs.append(roc_auc_score(truth[chosen], scores[chosen]))
    return (tuple(np.quantile(aps, (0.025, 0.975))), tuple(np.quantile(aucs, (0.025, 0.975))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=tuple(FEATURES))
    args = parser.parse_args()
    train_file, lock_dir = FEATURES[args.encoder]
    train_path = ROOT / "data/features" / train_file
    query_path = ROOT / "data/features" / f"edid_exploratory_{args.encoder}_geometry448_v1.npz"
    schedule_path = ROOT / "data/protocols/trace_bench_v2/edid_support_schedule.csv"
    audit_path = ROOT / "data/protocols/trace_bench_v2/edid_preprocessing_audit.csv"
    lock_path = ROOT / "outputs" / lock_dir / "locked_head_hyperparameters.json"
    train = load_feature_split(train_path, "train")
    query = load_feature_split(query_path, "validation")
    if tuple(query.class_order) != LABELS or len(query.record_ids) != 1333:
        raise RuntimeError("EDID query feature schema changed")
    audit = pd.read_csv(audit_path)
    if not np.array_equal(query.record_ids, audit.record_id.astype(str).to_numpy()):
        raise RuntimeError("Query IDs differ from pre-score preprocessing audit")
    if not np.array_equal(query.labels.sum(axis=0), np.asarray([326, 124, 106])):
        raise RuntimeError("EDID target counts changed")
    if set(query.record_ids) & set(train.record_ids):
        raise RuntimeError("Support/query record ID overlap")
    healthy = audit.class_folder.eq("Healthy").to_numpy()
    if healthy.sum() != 777:
        raise RuntimeError("Healthy control count changed")
    schedule = pd.read_csv(schedule_path)
    if len(schedule) != 326 or set(schedule.label) != set(LABELS):
        raise RuntimeError("Support schedule changed")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("novel_labels_used_during_tuning") != []:
        raise RuntimeError("Locked head used novel labels for tuning")
    train_features, query_features = l2_normalize(train.embeddings), l2_normalize(query.embeddings)
    train_index = {record_id: index for index, record_id in enumerate(train.record_ids)}
    scores_by_label_k: dict[tuple[str, int], list[np.ndarray]] = {}
    episode_metrics = []
    for count, episode in enumerate(schedule.itertuples(index=False), 1):
        label, k = str(episode.label), int(episode.k)
        positive, _ = parse_indices(episode, train_index)
        negative = np.flatnonzero(train.labels[:, train.class_order.index(label)] == 0)
        setting = lock["locked"]["abundant_negatives"][f"k{k}"]
        head = setting["selected_head_on_base_labels"]
        parameter = setting[head]["parameter"]
        predictions = score_head(head, train_features, query_features, positive, negative, parameter, int(episode.seed))
        if len(predictions) != len(query.record_ids) or not np.isfinite(predictions).all():
            raise RuntimeError("Invalid episode scores")
        mask = healthy | (query.labels[:, LABELS.index(label)] == 1)
        truth = query.labels[mask, LABELS.index(label)]
        target_scores = predictions[mask]
        scores_by_label_k.setdefault((label, k), []).append(target_scores)
        episode_metrics.append({"episode_id": episode.episode_id, "label": label, "k": k, "head": head, "parameter": parameter, "auprc": average_precision_score(truth, target_scores), "auroc": roc_auc_score(truth, target_scores)})
        if count % 50 == 0 or count == len(schedule):
            print(f"{args.encoder} episodes {count}/{len(schedule)}", flush=True)
    result_rows = []
    for label in LABELS:
        for k in (1, 3, 5):
            collection = scores_by_label_k.get((label, k))
            if collection is None:
                continue
            mask = healthy | (query.labels[:, LABELS.index(label)] == 1)
            truth = query.labels[mask, LABELS.index(label)]
            scores = np.mean(collection, axis=0)
            ap_ci, auc_ci = intervals(truth, scores, 20261111 + LABELS.index(label) * 10 + k)
            prevalence = float(truth.mean())
            result_rows.append({"encoder": args.encoder, "label": label, "k": k, "support_subsets": len(collection), "query_positives": int(truth.sum()), "healthy_controls": int((truth == 0).sum()), "prevalence": prevalence, "auprc": float(average_precision_score(truth, scores)), "auprc_over_prevalence": float(average_precision_score(truth, scores) / prevalence), "auprc_ci_low": ap_ci[0], "auprc_ci_high": ap_ci[1], "auroc": float(roc_auc_score(truth, scores)), "auroc_ci_low": auc_ci[0], "auroc_ci_high": auc_ci[1]})
    out_dir = ROOT / "outputs/edid_exploratory_transfer_v1" / args.encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_metrics).to_csv(out_dir / "episode_metrics.csv", index=False)
    pd.DataFrame(result_rows).to_csv(out_dir / "per_label_results.csv", index=False)
    summary = {"status": "EXPLORATORY_EXTERNAL_LABEL_EVALUATION", "encoder": args.encoder, "query_definition": "one EDID target folder vs EDID Healthy folder", "support_source": "RFMiD v1 training", "head_selection": "locked_on_RFMiD_v1_base_labels", "ci_scope": "stratified image bootstrap conditional on mean over frozen support subsets; patient/source uncertainty omitted", "train_feature_sha256": sha256_file(train_path), "query_feature_sha256": sha256_file(query_path), "schedule_sha256": sha256_file(schedule_path), "lock_sha256": sha256_file(lock_path), "limitations": ["folder labels not clinically adjudicated", "patient independence unknown", "coarse duplicate candidate screen not exhaustive", "healthy-only controls make task easier than differential diagnosis", "no per-image camera IDs"], "protected_test_images_opened": 0}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(result_rows).to_string(index=False))


if __name__ == "__main__":
    main()
