"""Paired K=3 RFMiD development evaluation with the frozen DINOv2-S head."""

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

FILES = {
    "frozen_dinov2_s": ("rfmid_train_dinov2_vits14_geometry_v1.npz", "rfmid_validation_dinov2_vits14_geometry_v1.npz"),
    "frozen_dinov2_l": ("rfmid_train_dinov2_vitl14_geometry448_v1.npz", "rfmid_validation_dinov2_vitl14_geometry448_v1.npz"),
    "frozen_retfound": ("rfmid_train_retfound_dinov2_meh_geometry448_v1.npz", "rfmid_validation_retfound_dinov2_meh_geometry448_v1.npz"),
    "plain_crop_matched_epoch5": ("rfmid_train_trace_ssl_plain_crop_matched_epoch5.npz", "rfmid_validation_trace_ssl_plain_crop_matched_epoch5.npz"),
    "full_field_epoch5": ("rfmid_train_trace_ssl_full_field_epoch5.npz", "rfmid_validation_trace_ssl_full_field_epoch5.npz"),
}


def main() -> None:
    protocol = ROOT / "data/protocols/trace_fewshot_v1/episode_plan.csv"
    episodes = pd.read_csv(protocol)
    episodes = episodes[episodes.k.eq(3)]
    if len(episodes) != 550 or episodes.label.nunique() != 11:
        raise RuntimeError("Frozen novel-label K=3 episode count changed")
    lock_file = ROOT / "outputs/dinov2_fewshot_v1/locked_head_hyperparameters.json"
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    choice = lock["locked"]["abundant_negatives"]["k3"]
    head = choice["selected_head_on_base_labels"]
    parameter = choice[head]["parameter"]
    if (head, parameter) != ("logistic", 1.0):
        raise RuntimeError("Frozen DINOv2-S K=3 head changed")
    output = ROOT / "outputs/trace_ssl_v1_k3_dev"
    output.mkdir(parents=True, exist_ok=True)
    all_results = []
    reference_ids = None
    reference_labels = None
    file_hashes = {}
    for model_name, (train_name, query_name) in FILES.items():
        train_path = ROOT / "data/features" / train_name
        query_path = ROOT / "data/features" / query_name
        train = load_feature_split(train_path, "train")
        query = load_feature_split(query_path, "validation")
        if train.class_order != query.class_order or len(train.record_ids) != 1917 or len(query.record_ids) != 639:
            raise RuntimeError("Development feature schema changed")
        paired = (train.record_ids, train.labels, query.record_ids, query.labels)
        if reference_ids is None:
            reference_ids = (train.record_ids.copy(), query.record_ids.copy())
            reference_labels = (train.labels.copy(), query.labels.copy())
        elif not (np.array_equal(reference_ids[0], paired[0]) and np.array_equal(reference_ids[1], paired[2]) and np.array_equal(reference_labels[0], paired[1]) and np.array_equal(reference_labels[1], paired[3])):
            raise RuntimeError("Models do not use paired train/query IDs and labels")
        train_x, query_x = l2_normalize(train.embeddings), l2_normalize(query.embeddings)
        train_index = {record_id: index for index, record_id in enumerate(train.record_ids)}
        for number, episode in enumerate(episodes.itertuples(index=False), 1):
            label = str(episode.label)
            class_index = train.class_order.index(label)
            positives, _ = parse_indices(episode, train_index)
            negatives = np.flatnonzero(train.labels[:, class_index] == 0)
            truth = query.labels[:, class_index]
            scores = score_head(head, train_x, query_x, positives, negatives, parameter, int(episode.seed))
            all_results.append({"model": model_name, "episode_id": episode.episode_id, "label": label, "replicate": int(episode.replicate), "auprc": float(average_precision_score(truth, scores)), "auroc": float(roc_auc_score(truth, scores)), "query_positives": int(truth.sum()), "query_prevalence": float(truth.mean())})
            if number % 100 == 0 or number == len(episodes):
                print(f"{model_name}: {number}/{len(episodes)} episodes", flush=True)
        file_hashes[model_name] = {"train": sha256_file(train_path), "validation": sha256_file(query_path)}
    frame = pd.DataFrame(all_results)
    frame.to_csv(output / "episode_metrics.csv", index=False)
    per_label = frame.groupby(["model", "label"], as_index=False).agg(auprc=("auprc", "mean"), auroc=("auroc", "mean"), query_positives=("query_positives", "first"), query_prevalence=("query_prevalence", "first"))
    per_label.to_csv(output / "per_label_results.csv", index=False)
    macro = per_label.groupby("model", as_index=False).agg(macro_auprc=("auprc", "mean"), macro_auroc=("auroc", "mean"))
    macro.to_csv(output / "macro_results.csv", index=False)
    baseline = float(macro.loc[macro.model.eq("frozen_dinov2_s"), "macro_auprc"].iloc[0])
    if abs(baseline - 0.198776) > 0.00001:
        raise RuntimeError(f"Frozen baseline did not reproduce: {baseline}")
    summary = {"status": "DEVELOPMENT_EVALUATION_ONLY", "endpoint": "mean episode AUPRC across 11 novel labels at K=3 with abundant RFMiD training negatives", "head": head, "parameter": parameter, "support_draws_per_label": 50, "query_records": 639, "episode_plan_sha256": sha256_file(protocol), "locked_head_sha256": sha256_file(lock_file), "feature_hashes": file_hashes, "macro_results": macro.to_dict("records"), "test_records_accessed": 0, "warning": "Validation episodes have been used in prior development; no model selection here establishes final-test performance."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
