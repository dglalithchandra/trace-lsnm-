"""Tune simple heads on base labels, then evaluate fixed novel-label episodes."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_protocols import (
    knn_class_margin_scores,
    l2_normalize,
    load_feature_split,
    logistic_fewshot_scores,
    prototype_margin_scores,
)


TUNING_REPLICATES = 10
KNN_CANDIDATES = (1, 3, 5)
LOGISTIC_C_CANDIDATES = (0.01, 0.1, 1.0)


def parse_indices(
    episode: object, record_index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    positives = json.loads(episode.support_positive_record_ids_json)
    negatives = json.loads(episode.strict_support_negative_record_ids_json)
    return (
        np.array([record_index[item] for item in positives], dtype=np.int64),
        np.array([record_index[item] for item in negatives], dtype=np.int64),
    )


def score_head(
    head: str,
    train_features: np.ndarray,
    validation_features: np.ndarray,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    parameter: float | int | None,
    seed: int,
) -> np.ndarray:
    if head == "prototype":
        return prototype_margin_scores(
            train_features,
            positive_indices,
            negative_indices,
            validation_features,
        )
    if head == "knn":
        return knn_class_margin_scores(
            train_features,
            positive_indices,
            negative_indices,
            validation_features,
            neighbors_per_class=int(parameter),
        )
    if head == "logistic":
        return logistic_fewshot_scores(
            train_features,
            positive_indices,
            negative_indices,
            validation_features,
            c_value=float(parameter),
            random_state=seed,
        )
    raise ValueError(f"Unknown head: {head}")


def macro_from_episode_rows(rows: pd.DataFrame) -> pd.DataFrame:
    per_label = (
        rows.groupby(
            ["head", "parameter", "support_protocol", "k", "label"],
            as_index=False,
        )[["query_prevalence", "auprc", "auprc_lift_over_prevalence"]]
        .mean()
    )
    return (
        per_label.groupby(
            ["head", "parameter", "support_protocol", "k"], as_index=False
        )[["query_prevalence", "auprc", "auprc_lift_over_prevalence"]]
        .mean()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        choices=("dinov2", "dinov2_l", "retfound", "flair", "trace_adapter"),
        default="dinov2",
        help="Frozen feature source to evaluate on the identical protocol.",
    )
    args = parser.parse_args()
    started = time.time()
    feature_dir = WORKSPACE / "data" / "features"
    protocol_dir = WORKSPACE / "data" / "protocols" / "trace_fewshot_v1"
    encoder_settings = {
        "dinov2": {
            "train": "rfmid_train_dinov2_vits14_geometry_v1.npz",
            "validation": "rfmid_validation_dinov2_vits14_geometry_v1.npz",
            "output": "dinov2_fewshot_v1",
            "id": "dinov2_vits14_geometry_v1",
            "display": "frozen DINOv2-S/14",
            "features": "384-value CLS token after TRACE geometry at 448 px",
            "contamination_tag": "generic_pretraining_overlap_unknown",
        },
        "dinov2_l": {
            "train": "rfmid_train_dinov2_vitl14_geometry448_v1.npz",
            "validation": "rfmid_validation_dinov2_vitl14_geometry448_v1.npz",
            "output": "dinov2_l_fewshot_v1",
            "id": "dinov2_vitl14_geometry448_v1",
            "display": "frozen DINOv2-L/14",
            "features": "1024-value CLS token after TRACE geometry at 448 px",
            "contamination_tag": "generic_pretraining_overlap_unknown",
        },
        "retfound": {
            "train": "rfmid_train_retfound_dinov2_meh_geometry448_v1.npz",
            "validation": "rfmid_validation_retfound_dinov2_meh_geometry448_v1.npz",
            "output": "retfound_fewshot_v1",
            "id": "retfound_dinov2_meh_geometry448_v1",
            "display": "frozen RETFound-DINOv2 MEH teacher backbone",
            "features": "1024-value CLS token after TRACE geometry at 448 px",
            "contamination_tag": "declared_alzeye_pretraining_rfmid_not_listed",
        },
        "flair": {
            "train": "rfmid_train_flair_geometry_v1.npz",
            "validation": "rfmid_validation_flair_geometry_v1.npz",
            "output": "flair_fewshot_v1",
            "id": "flair_geometry_v1",
            "display": "frozen FLAIR retinal image tower",
            "features": "512-value unit-normalised retinal image embedding",
            "contamination_tag": "possibly_seen_rfmid_listed_in_pretraining_assembly",
        },
        "trace_adapter": {
            "train": "rfmid_train_trace_rare_adapter_v1.npz",
            "validation": "rfmid_validation_trace_rare_adapter_v1.npz",
            "output": "trace_rare_adapter_fewshot_v1",
            "id": "trace_rare_adapter_v1",
            "display": "TRACE base-only rare-aware adapter over frozen DINOv2-S/14",
            "features": "384-value unit-normalised base-trained episodic embedding",
            "contamination_tag": "novel_positive_records_excluded_from_adapter_training",
        },
    }
    setting = encoder_settings[args.encoder]
    output_dir = WORKSPACE / "outputs" / str(setting["output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_feature_split(
        feature_dir / str(setting["train"]), "train"
    )
    validation = load_feature_split(
        feature_dir / str(setting["validation"]),
        "validation",
    )
    if train.class_order != validation.class_order:
        raise RuntimeError("Feature class orders differ")
    train_features = l2_normalize(train.embeddings)
    validation_features = l2_normalize(validation.embeddings)
    record_index = {record_id: index for index, record_id in enumerate(train.record_ids)}
    base_episodes = pd.read_csv(protocol_dir / "base_tuning_episode_plan.csv")
    novel_episodes = pd.read_csv(protocol_dir / "episode_plan.csv")
    base_episodes = base_episodes[base_episodes["replicate"] < TUNING_REPLICATES]

    negative_pool: dict[int, np.ndarray] = {
        index: np.flatnonzero(train.labels[:, index] == 0)
        for index in range(len(train.class_order))
    }

    tuning_rows: list[dict[str, object]] = []
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for number, episode in enumerate(base_episodes.itertuples(index=False), start=1):
        positive_indices, strict_negative_indices = parse_indices(episode, record_index)
        query_labels = validation.labels[:, episode.class_index]
        prevalence = float(query_labels.mean())
        protocols = {
            "abundant_negatives": negative_pool[int(episode.class_index)],
            "strict_k_negatives": strict_negative_indices,
        }
        for support_protocol, negative_indices in protocols.items():
            candidates: list[tuple[str, float | int | None]] = [("prototype", None)]
            candidates.extend(
                ("knn", value) for value in KNN_CANDIDATES if value <= episode.k
            )
            candidates.extend(("logistic", value) for value in LOGISTIC_C_CANDIDATES)
            for head, parameter in candidates:
                scores = score_head(
                    head,
                    train_features,
                    validation_features,
                    positive_indices,
                    negative_indices,
                    parameter,
                    int(episode.seed),
                )
                auprc = float(average_precision_score(query_labels, scores))
                tuning_rows.append(
                    {
                        "episode_id": episode.episode_id,
                        "label": episode.label,
                        "k": int(episode.k),
                        "replicate": int(episode.replicate),
                        "support_protocol": support_protocol,
                        "head": head,
                        "parameter": "none" if parameter is None else str(parameter),
                        "query_prevalence": prevalence,
                        "auprc": auprc,
                        "auprc_lift_over_prevalence": auprc - prevalence,
                    }
                )
        if number % 100 == 0:
            print(f"base_tuning {number}/{len(base_episodes)}", flush=True)

    tuning = pd.DataFrame(tuning_rows)
    tuning.to_csv(output_dir / "base_tuning_episode_results.csv", index=False)
    tuning_macro = macro_from_episode_rows(tuning)
    tuning_macro.to_csv(output_dir / "base_tuning_macro_results.csv", index=False)

    locked: dict[str, dict[str, dict[str, object]]] = {}
    for (support_protocol, k), group in tuning_macro.groupby(
        ["support_protocol", "k"]
    ):
        key = f"k{k}"
        locked.setdefault(support_protocol, {})[key] = {}
        best_per_head: list[pd.Series] = []
        for head, head_group in group.groupby("head"):
            ordered = head_group.sort_values(
                ["auprc_lift_over_prevalence", "parameter"],
                ascending=[False, True],
            )
            best = ordered.iloc[0]
            best_per_head.append(best)
            value: object = None
            if head == "knn":
                value = int(float(best["parameter"]))
            elif head == "logistic":
                value = float(best["parameter"])
            locked[support_protocol][key][head] = {
                "parameter": value,
                "base_macro_auprc": float(best["auprc"]),
                "base_macro_auprc_lift_over_prevalence": float(
                    best["auprc_lift_over_prevalence"]
                ),
            }
        selected = max(
            best_per_head,
            key=lambda row: float(row["auprc_lift_over_prevalence"]),
        )
        locked[support_protocol][key]["selected_head_on_base_labels"] = str(
            selected["head"]
        )

    lock_document = {
        "status": "LOCKED_FROM_BASE_LABELS_ONLY",
        "encoder": setting["id"],
        "tuning_replicates": TUNING_REPLICATES,
        "selection_metric": "macro_auprc_lift_over_query_prevalence",
        "candidates": {
            "prototype": [None],
            "knn_neighbors_per_class": list(KNN_CANDIDATES),
            "logistic_c": list(LOGISTIC_C_CANDIDATES),
        },
        "locked": locked,
        "novel_labels_used_during_tuning": [],
        "test_records_accessed": 0,
    }
    (output_dir / "locked_head_hyperparameters.json").write_text(
        json.dumps(lock_document, indent=2, sort_keys=True), encoding="utf-8"
    )

    novel_rows: list[dict[str, object]] = []
    episode_score_arrays: list[np.ndarray] = []
    episode_score_ids: list[str] = []
    episode_score_labels: list[str] = []
    episode_score_k: list[int] = []
    episode_score_replicates: list[int] = []
    episode_score_protocols: list[str] = []
    episode_score_heads: list[str] = []
    score_sums: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(len(validation.record_ids), dtype=np.float64)
    )
    score_counts: dict[str, int] = defaultdict(int)
    for number, episode in enumerate(novel_episodes.itertuples(index=False), start=1):
        positive_indices, strict_negative_indices = parse_indices(episode, record_index)
        query_labels = validation.labels[:, episode.class_index]
        prevalence = float(query_labels.mean())
        protocols = {
            "abundant_negatives": negative_pool[int(episode.class_index)],
            "strict_k_negatives": strict_negative_indices,
        }
        for support_protocol, negative_indices in protocols.items():
            settings = locked[support_protocol][f"k{episode.k}"]
            for head in ("prototype", "knn", "logistic"):
                parameter = settings[head]["parameter"]
                scores = score_head(
                    head,
                    train_features,
                    validation_features,
                    positive_indices,
                    negative_indices,
                    parameter,
                    int(episode.seed),
                )
                auprc = float(average_precision_score(query_labels, scores))
                novel_rows.append(
                    {
                        "episode_id": episode.episode_id,
                        "label": episode.label,
                        "class_index": int(episode.class_index),
                        "k": int(episode.k),
                        "replicate": int(episode.replicate),
                        "support_protocol": support_protocol,
                        "head": head,
                        "parameter": "none" if parameter is None else str(parameter),
                        "query_prevalence": prevalence,
                        "auprc": auprc,
                        "auprc_lift_over_prevalence": auprc - prevalence,
                        "selected_head_on_base_labels": (
                            head == settings["selected_head_on_base_labels"]
                        ),
                    }
                )
                score_key = (
                    f"{support_protocol}|k{episode.k}|{episode.label}|{head}"
                )
                score_sums[score_key] += scores
                score_counts[score_key] += 1
                episode_score_arrays.append(scores.astype(np.float32))
                episode_score_ids.append(str(episode.episode_id))
                episode_score_labels.append(str(episode.label))
                episode_score_k.append(int(episode.k))
                episode_score_replicates.append(int(episode.replicate))
                episode_score_protocols.append(support_protocol)
                episode_score_heads.append(head)
        if number % 100 == 0:
            print(f"novel_evaluation {number}/{len(novel_episodes)}", flush=True)

    novel = pd.DataFrame(novel_rows)
    novel.to_csv(output_dir / "novel_episode_results.csv", index=False)
    novel_per_label = (
        novel.groupby(
            [
                "support_protocol",
                "k",
                "label",
                "head",
                "parameter",
                "selected_head_on_base_labels",
            ],
            as_index=False,
        )[["query_prevalence", "auprc", "auprc_lift_over_prevalence"]]
        .mean()
    )
    novel_macro = (
        novel_per_label.groupby(
            [
                "support_protocol",
                "k",
                "head",
                "parameter",
                "selected_head_on_base_labels",
            ],
            as_index=False,
        )[["query_prevalence", "auprc", "auprc_lift_over_prevalence"]]
        .mean()
    )
    novel_per_label.to_csv(output_dir / "novel_per_label_results.csv", index=False)
    novel_macro.to_csv(output_dir / "novel_macro_results.csv", index=False)

    score_keys = sorted(score_sums)
    mean_scores = np.stack(
        [score_sums[key] / score_counts[key] for key in score_keys]
    ).astype(np.float32)
    np.savez_compressed(
        output_dir / "novel_mean_query_scores.npz",
        keys=np.array(score_keys),
        mean_scores=mean_scores,
        validation_record_ids=validation.record_ids,
        validation_labels=validation.labels,
        class_order=np.array(validation.class_order),
    )
    np.savez_compressed(
        output_dir / "novel_episode_query_scores.npz",
        episode_ids=np.array(episode_score_ids),
        labels=np.array(episode_score_labels),
        k=np.array(episode_score_k, dtype=np.int16),
        replicates=np.array(episode_score_replicates, dtype=np.int16),
        support_protocols=np.array(episode_score_protocols),
        heads=np.array(episode_score_heads),
        scores=np.stack(episode_score_arrays),
        validation_record_ids=validation.record_ids,
        validation_labels=validation.labels,
        class_order=np.array(validation.class_order),
    )

    summary = {
        "status": "PASS",
        "encoder": setting["display"],
        "features": setting["features"],
        "contamination_tag": setting["contamination_tag"],
        "base_label_tuning_episodes": len(base_episodes),
        "novel_label_evaluation_episodes": len(novel_episodes),
        "heads": ["prototype", "knn", "logistic"],
        "support_protocols": ["abundant_negatives", "strict_k_negatives"],
        "elapsed_seconds": time.time() - started,
        "test_records_accessed": 0,
        "warning": (
            "These are development novel-label results on a fixed validation "
            "query set, not final-test or clinical-performance estimates."
        ),
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nNovel macro results:\n" + novel_macro.to_string(index=False))


if __name__ == "__main__":
    main()

