"""Replay locked scoring on RFMiD validation only; never load test inputs."""

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

from scripts.evaluate_trace_final_v1 import PRIMARY, TRAIN_FILES, primary_bootstrap, score_groups, weighted_ap_for_all_episodes
from scripts.run_dinov2_fewshot_benchmark import parse_indices, score_head
from trace_protocols import l2_normalize, load_feature_split

VALIDATION_FILES = {
    "frozen_dinov2_s": "rfmid_validation_dinov2_vits14_geometry_v1.npz",
    "frozen_dinov2_l": "rfmid_validation_dinov2_vitl14_geometry448_v1.npz",
    "frozen_retfound": "rfmid_validation_retfound_dinov2_meh_geometry448_v1.npz",
    "plain_crop_matched_epoch5": "rfmid_validation_trace_ssl_plain_crop_matched_epoch5.npz",
    "full_field_epoch5": "rfmid_validation_trace_ssl_full_field_epoch5.npz",
}


def main() -> None:
    schedule = pd.read_csv(ROOT / "data/protocols/trace_fewshot_v1/episode_plan.csv")
    episodes = schedule[(schedule.label == "ERM") & (schedule.k == 3) & (schedule.replicate < 2)]
    if len(episodes) != 2:
        raise RuntimeError("Expected two frozen ERM development episodes")
    previous = pd.read_csv(ROOT / "outputs/trace_ssl_v1_k3_dev/episode_metrics.csv")
    reports = []
    for model, train_file in TRAIN_FILES.items():
        train = load_feature_split(ROOT / "data/features" / train_file, "train")
        query = load_feature_split(ROOT / "data/features" / VALIDATION_FILES[model], "validation")
        if train.class_order != query.class_order or len(train.record_ids) != 1917 or len(query.record_ids) != 639:
            raise RuntimeError("Development features changed")
        train_index = {record_id: index for index, record_id in enumerate(train.record_ids)}
        truth = query.labels[:, train.class_order.index("ERM")]
        scores_by_episode = []
        for episode in episodes.itertuples(index=False):
            positives, _ = parse_indices(episode, train_index)
            negatives = np.flatnonzero(train.labels[:, train.class_order.index("ERM")] == 0)
            scores = score_head("logistic", l2_normalize(train.embeddings), l2_normalize(query.embeddings), positives, negatives, 1.0, int(episode.seed))
            expected = previous[(previous.model == model) & (previous.episode_id == episode.episode_id)]
            if len(expected) != 1:
                raise RuntimeError("Prior development episode missing")
            ap = float(average_precision_score(truth, scores))
            auc = float(roc_auc_score(truth, scores))
            if abs(ap - float(expected.auprc.iloc[0])) > 1e-10 or abs(auc - float(expected.auroc.iloc[0])) > 1e-10:
                raise RuntimeError(f"Locked scoring did not reproduce: {model} {episode.episode_id}")
            scores_by_episode.append(scores)
            reports.append({"model": model, "episode_id": episode.episode_id, "auprc": ap, "auroc": auc})
        matrix = np.stack(scores_by_episode)
        weighted = weighted_ap_for_all_episodes(*score_groups(matrix), truth, np.ones(len(truth)))
        exact = [average_precision_score(truth, scores) for scores in matrix]
        if not np.allclose(weighted, exact, rtol=0, atol=1e-12):
            raise RuntimeError(f"Bootstrap average precision disagrees with scikit-learn: {model}")
    tied_scores = np.array([[0.9, 0.9, 0.3, 0.1], [0.5, 0.5, 0.5, 0.2]])
    tied_truth = np.array([1, 0, 1, 0], dtype=np.uint8)
    tied_weights = np.array([2.0, 1.0, 1.0, 3.0])
    tied_ap = weighted_ap_for_all_episodes(*score_groups(tied_scores), tied_truth, tied_weights)
    expected_tied_ap = [average_precision_score(tied_truth, row, sample_weight=tied_weights) for row in tied_scores]
    if not np.allclose(tied_ap, expected_tied_ap, rtol=0, atol=1e-12):
        raise RuntimeError("Weighted AP mishandles mixed-label score ties")
    identical_scores = np.repeat(tied_scores[:1], 50, axis=0)
    paired = {model: {label: identical_scores for label in ("A", "B")} for model in PRIMARY}
    bootstrap = primary_bootstrap(paired, {"A": tied_truth, "B": tied_truth}, ["A", "B"], resamples=20)
    if bootstrap["paired_95_percentile_interval"] != [0.0, 0.0]:
        raise RuntimeError("Paired bootstrap did not preserve an exact zero contrast")
    result = {"status": "PASS_DEVELOPMENT_REPLAY_ONLY", "episodes": len(reports), "models": len(TRAIN_FILES), "protected_test_records_read": 0, "note": "Replays two ERM K=3 episodes per model against prior validation scores. Does not validate full final-test execution."}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
