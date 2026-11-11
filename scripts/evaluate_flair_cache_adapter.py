"""Base-tune a text-plus-image cache adapter and evaluate novel labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_protocols import l2_normalize, prototype_margin_scores


OUTPUT_DIR = WORKSPACE / "outputs" / "flair_cache_v1"
TUNING_REPLICATES = 10
ALPHA_CANDIDATES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def parse_support(
    episode: object, record_index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    positives = json.loads(episode.support_positive_record_ids_json)
    negatives = json.loads(episode.strict_support_negative_record_ids_json)
    return (
        np.array([record_index[item] for item in positives], dtype=np.int64),
        np.array([record_index[item] for item in negatives], dtype=np.int64),
    )


def macro_table(rows: pd.DataFrame) -> pd.DataFrame:
    per_label = (
        rows.groupby(["support_protocol", "k", "alpha", "label"], as_index=False)[
            ["query_prevalence", "auprc", "auprc_lift_over_prevalence"]
        ]
        .mean()
    )
    return (
        per_label.groupby(["support_protocol", "k", "alpha"], as_index=False)[
            ["query_prevalence", "auprc", "auprc_lift_over_prevalence"]
        ]
        .mean()
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol_dir = WORKSPACE / "data" / "protocols" / "trace_fewshot_v1"
    base_episodes = pd.read_csv(protocol_dir / "base_tuning_episode_plan.csv")
    base_episodes = base_episodes[base_episodes["replicate"] < TUNING_REPLICATES]
    novel_episodes = pd.read_csv(protocol_dir / "episode_plan.csv")

    with np.load(
        WORKSPACE / "data" / "features" / "rfmid_train_flair_geometry_v1.npz",
        allow_pickle=False,
    ) as p:
        train_embeddings = l2_normalize(p["embeddings"].astype(np.float32))
        train_labels = p["labels"].astype(np.uint8)
        train_ids = p["record_ids"].astype(str)
        class_order = p["class_order"].astype(str).tolist()
    with np.load(
        WORKSPACE / "data" / "features" / "rfmid_validation_flair_geometry_v1.npz",
        allow_pickle=False,
    ) as p:
        validation_embeddings = l2_normalize(p["embeddings"].astype(np.float32))
        validation_labels = p["labels"].astype(np.uint8)
        validation_ids = p["record_ids"].astype(str)
        if p["class_order"].astype(str).tolist() != class_order:
            raise RuntimeError("FLAIR train/validation class order mismatch")
    with np.load(
        WORKSPACE / "outputs" / "flair_zero_shot_v1" / "selected_zero_shot_scores.npz",
        allow_pickle=False,
    ) as p:
        zero_labels = p["labels"].astype(str).tolist()
        zero_scores = p["scores"].astype(np.float32)
        selected_variant = str(p["selected_variant"][0])
        if not np.array_equal(p["validation_labels"], validation_labels):
            raise RuntimeError("Zero-shot and feature labels differ")

    record_index = {record_id: index for index, record_id in enumerate(train_ids)}
    class_index = {label: index for index, label in enumerate(class_order)}
    zero_index = {label: index for index, label in enumerate(zero_labels)}
    negative_pool = {
        index: np.flatnonzero(train_labels[:, index] == 0)
        for index in range(len(class_order))
    }

    tuning_rows: list[dict[str, object]] = []
    for episode in base_episodes.itertuples(index=False):
        positive, strict_negative = parse_support(episode, record_index)
        y = validation_labels[:, episode.class_index]
        prevalence = float(y.mean())
        text_score = zero_scores[:, zero_index[episode.label]]
        for protocol, negative in (
            ("abundant_negatives", negative_pool[int(episode.class_index)]),
            ("strict_k_negatives", strict_negative),
        ):
            cache_score = prototype_margin_scores(
                train_embeddings,
                positive,
                negative,
                validation_embeddings,
            )
            for alpha in ALPHA_CANDIDATES:
                combined = text_score + alpha * cache_score
                auprc = float(average_precision_score(y, combined))
                tuning_rows.append(
                    {
                        "episode_id": episode.episode_id,
                        "label": episode.label,
                        "k": int(episode.k),
                        "replicate": int(episode.replicate),
                        "support_protocol": protocol,
                        "alpha": alpha,
                        "query_prevalence": prevalence,
                        "auprc": auprc,
                        "auprc_lift_over_prevalence": auprc - prevalence,
                    }
                )
    tuning = pd.DataFrame(tuning_rows)
    tuning.to_csv(OUTPUT_DIR / "base_alpha_tuning_episodes.csv", index=False)
    tuning_macro = macro_table(tuning)
    tuning_macro.to_csv(OUTPUT_DIR / "base_alpha_tuning_macro.csv", index=False)

    locked_alpha: dict[str, dict[str, float]] = {}
    for (protocol, k), group in tuning_macro.groupby(["support_protocol", "k"]):
        best = group.sort_values(
            ["auprc_lift_over_prevalence", "alpha"], ascending=[False, True]
        ).iloc[0]
        locked_alpha.setdefault(protocol, {})[f"k{k}"] = float(best["alpha"])
    lock = {
        "status": "LOCKED_FROM_BASE_LABELS_ONLY",
        "selected_text_prompt_variant": selected_variant,
        "alpha_candidates": list(ALPHA_CANDIDATES),
        "tuning_replicates": TUNING_REPLICATES,
        "locked_alpha": locked_alpha,
        "novel_labels_used_during_tuning": [],
        "test_records_accessed": 0,
    }
    (OUTPUT_DIR / "locked_cache_hyperparameters.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8"
    )

    novel_rows: list[dict[str, object]] = []
    score_arrays: list[np.ndarray] = []
    score_labels: list[str] = []
    score_k: list[int] = []
    score_replicates: list[int] = []
    score_protocols: list[str] = []
    for episode in novel_episodes.itertuples(index=False):
        positive, strict_negative = parse_support(episode, record_index)
        y = validation_labels[:, episode.class_index]
        prevalence = float(y.mean())
        text_score = zero_scores[:, zero_index[episode.label]]
        for protocol, negative in (
            ("abundant_negatives", negative_pool[int(episode.class_index)]),
            ("strict_k_negatives", strict_negative),
        ):
            alpha = locked_alpha[protocol][f"k{episode.k}"]
            cache_score = prototype_margin_scores(
                train_embeddings,
                positive,
                negative,
                validation_embeddings,
            )
            combined = text_score + alpha * cache_score
            auprc = float(average_precision_score(y, combined))
            novel_rows.append(
                {
                    "episode_id": episode.episode_id,
                    "label": episode.label,
                    "class_index": int(episode.class_index),
                    "k": int(episode.k),
                    "replicate": int(episode.replicate),
                    "support_protocol": protocol,
                    "alpha": alpha,
                    "query_prevalence": prevalence,
                    "auprc": auprc,
                    "auprc_lift_over_prevalence": auprc - prevalence,
                }
            )
            score_arrays.append(combined.astype(np.float32))
            score_labels.append(str(episode.label))
            score_k.append(int(episode.k))
            score_replicates.append(int(episode.replicate))
            score_protocols.append(protocol)

    novel = pd.DataFrame(novel_rows)
    novel.to_csv(OUTPUT_DIR / "novel_cache_episode_results.csv", index=False)
    novel_per_label = (
        novel.groupby(
            ["support_protocol", "k", "label", "alpha"], as_index=False
        )[["query_prevalence", "auprc", "auprc_lift_over_prevalence"]]
        .mean()
    )
    novel_macro = (
        novel_per_label.groupby(["support_protocol", "k", "alpha"], as_index=False)[
            ["query_prevalence", "auprc", "auprc_lift_over_prevalence"]
        ]
        .mean()
    )
    novel_per_label.to_csv(OUTPUT_DIR / "novel_cache_per_label.csv", index=False)
    novel_macro.to_csv(OUTPUT_DIR / "novel_cache_macro.csv", index=False)
    np.savez_compressed(
        OUTPUT_DIR / "novel_cache_query_scores.npz",
        labels=np.asarray(score_labels, dtype=np.str_),
        k=np.asarray(score_k, dtype=np.int16),
        replicates=np.asarray(score_replicates, dtype=np.int16),
        support_protocols=np.asarray(score_protocols, dtype=np.str_),
        scores=np.stack(score_arrays),
        validation_record_ids=validation_ids,
        validation_labels=validation_labels,
        class_order=np.asarray(class_order, dtype=np.str_),
    )
    summary = {
        "status": "PASS",
        "method": "FLAIR domain-knowledge text score plus alpha times image-cache prototype margin",
        "selected_text_prompt_variant": selected_variant,
        "locked_alpha": locked_alpha,
        "contamination_tag": "possibly_seen_rfmid_listed_in_pretraining_assembly",
        "test_records_accessed": 0,
        "warning": (
            "Alpha was selected on base labels only. RFMiD exposure during FLAIR "
            "pretraining prevents a clean unseen-data claim."
        ),
    }
    (OUTPUT_DIR / "cache_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nNovel cache macro results:\n" + novel_macro.to_string(index=False))


if __name__ == "__main__":
    main()

