"""Run Experiment 0 controls without loading the protected final test split."""

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

from trace_protocols import (
    l2_normalize,
    load_feature_split,
    prototype_margin_scores,
)


NEAR_DUPLICATE_SCREENING_COSINE = 0.9999


def build_acquisition_proxy_features(
    manifest: pd.DataFrame,
    train_record_ids: np.ndarray,
    validation_record_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Resolution and acquisition-domain features with train-only fitting."""

    selected_ids = set(train_record_ids.tolist()) | set(validation_record_ids.tolist())
    selected = manifest[manifest["record_id"].isin(selected_ids)].copy()
    if len(selected) != len(selected_ids) or not selected["record_id"].is_unique:
        raise RuntimeError("Feature records do not map one-to-one to manifest rows")
    row_by_id = selected.set_index("record_id")

    def numeric(ids: np.ndarray) -> np.ndarray:
        rows = row_by_id.loc[ids]
        width = pd.to_numeric(rows["width"], errors="raise").to_numpy(float)
        height = pd.to_numeric(rows["height"], errors="raise").to_numpy(float)
        byte_count = pd.to_numeric(
            rows["uncompressed_bytes"], errors="raise"
        ).to_numpy(float)
        return np.column_stack(
            [
                np.log1p(width),
                np.log1p(height),
                width / np.maximum(height, 1.0),
                np.log1p(byte_count),
            ]
        )

    train_numeric = numeric(train_record_ids)
    validation_numeric = numeric(validation_record_ids)
    mean = train_numeric.mean(axis=0)
    scale = np.maximum(train_numeric.std(axis=0), 1e-6)
    train_numeric = (train_numeric - mean) / scale
    validation_numeric = (validation_numeric - mean) / scale

    train_domains = row_by_id.loc[train_record_ids, "camera_domain"].fillna("").astype(str)
    validation_domains = (
        row_by_id.loc[validation_record_ids, "camera_domain"].fillna("").astype(str)
    )
    categories = sorted(train_domains.unique().tolist())
    category_index = {name: index for index, name in enumerate(categories)}

    def one_hot(values: pd.Series) -> np.ndarray:
        result = np.zeros((len(values), len(categories)), dtype=np.float32)
        for row_index, value in enumerate(values.tolist()):
            if value in category_index:
                result[row_index, category_index[value]] = 1.0
        return result

    train = np.column_stack([train_numeric, one_hot(train_domains)]).astype(np.float32)
    validation = np.column_stack(
        [validation_numeric, one_hot(validation_domains)]
    ).astype(np.float32)
    metadata = {
        "numeric_features": [
            "log_width",
            "log_height",
            "aspect_ratio",
            "log_uncompressed_bytes",
        ],
        "categorical_feature": "camera_domain_or_resolution_proxy",
        "train_categories": categories,
        "feature_width": train.shape[1],
        "warning": (
            "This is an acquisition-metadata proxy baseline, not a verified "
            "per-image camera-model baseline."
        ),
    }
    return train, validation, metadata


def episode_indices(
    row: object, record_index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    positives = json.loads(row.support_positive_record_ids_json)
    negatives = json.loads(row.strict_support_negative_record_ids_json)
    return (
        np.array([record_index[item] for item in positives], dtype=np.int64),
        np.array([record_index[item] for item in negatives], dtype=np.int64),
    )


def main() -> None:
    feature_dir = WORKSPACE / "data" / "features"
    protocol_dir = WORKSPACE / "data" / "protocols" / "trace_fewshot_v1"
    output_dir = WORKSPACE / "outputs" / "fewshot_protocol_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_feature_split(
        feature_dir / "rfmid_train_dinov2_vits14_geometry_v1.npz", "train"
    )
    validation = load_feature_split(
        feature_dir / "rfmid_validation_dinov2_vits14_geometry_v1.npz",
        "validation",
    )
    episodes = pd.read_csv(protocol_dir / "episode_plan.csv")
    labels = pd.read_csv(protocol_dir / "label_eligibility.csv")
    protocol = json.loads((protocol_dir / "protocol.json").read_text(encoding="utf-8"))

    manifest_columns = [
        "record_id",
        "final_split",
        "split_group_id",
        "width",
        "height",
        "uncompressed_bytes",
        "camera_domain",
    ]
    manifest = pd.read_csv(
        WORKSPACE / "data" / "manifests" / "trace_analysis_manifest.csv",
        usecols=manifest_columns,
    )
    selected_ids = set(train.record_ids.tolist()) | set(validation.record_ids.tolist())
    selected_manifest = manifest[manifest["record_id"].isin(selected_ids)].copy()
    split_by_id = selected_manifest.set_index("record_id")["final_split"].to_dict()
    if any(split_by_id[item] != "train" for item in train.record_ids):
        raise RuntimeError("A training feature record is not in the frozen train split")
    if any(split_by_id[item] != "validation" for item in validation.record_ids):
        raise RuntimeError("A validation feature record is not in the frozen validation split")
    group_split_counts = selected_manifest.groupby("split_group_id")["final_split"].nunique()
    known_group_crossings = int((group_split_counts > 1).sum())
    if known_group_crossings:
        raise RuntimeError("Known duplicate/related group crosses train and validation")

    metadata_train, metadata_validation, metadata_description = (
        build_acquisition_proxy_features(
            manifest, train.record_ids, validation.record_ids
        )
    )
    feature_sets = {
        "dinov2_vits14_geometry": (train.embeddings, validation.embeddings),
        "acquisition_metadata_proxy": (metadata_train, metadata_validation),
    }
    record_index = {record_id: index for index, record_id in enumerate(train.record_ids)}

    abundant_negative_indices = {
        int(row.class_index): np.flatnonzero(
            train.labels[:, int(row.class_index)] == 0
        )
        for row in labels[labels["role"].eq("novel")].itertuples(index=False)
    }
    result_rows: list[dict[str, object]] = []
    for episode in episodes.itertuples(index=False):
        positive_indices, strict_negative_indices = episode_indices(
            episode, record_index
        )
        query_labels = validation.labels[:, episode.class_index]
        prevalence = float(query_labels.mean())
        for feature_name, (support_features, query_features) in feature_sets.items():
            for negative_protocol, negative_indices in (
                (
                    "abundant_negatives",
                    abundant_negative_indices[int(episode.class_index)],
                ),
                ("strict_k_negatives", strict_negative_indices),
            ):
                scores = prototype_margin_scores(
                    support_features,
                    positive_indices,
                    negative_indices,
                    query_features,
                )
                auprc = float(average_precision_score(query_labels, scores))
                result_rows.append(
                    {
                        "episode_id": episode.episode_id,
                        "label": episode.label,
                        "k": int(episode.k),
                        "replicate": int(episode.replicate),
                        "feature_source": feature_name,
                        "support_protocol": negative_protocol,
                        "control": "observed_labels",
                        "query_prevalence": prevalence,
                        "auprc": auprc,
                        "auprc_lift_over_prevalence": auprc - prevalence,
                    }
                )
                if (
                    feature_name == "dinov2_vits14_geometry"
                    and negative_protocol == "abundant_negatives"
                ):
                    shuffle_rng = np.random.default_rng(int(episode.seed) ^ 0xA5A5A5A5)
                    shuffled = shuffle_rng.permutation(query_labels)
                    shuffled_auprc = float(average_precision_score(shuffled, scores))
                    result_rows.append(
                        {
                            "episode_id": episode.episode_id,
                            "label": episode.label,
                            "k": int(episode.k),
                            "replicate": int(episode.replicate),
                            "feature_source": feature_name,
                            "support_protocol": negative_protocol,
                            "control": "query_label_shuffle",
                            "query_prevalence": prevalence,
                            "auprc": shuffled_auprc,
                            "auprc_lift_over_prevalence": shuffled_auprc - prevalence,
                        }
                    )

    results = pd.DataFrame(result_rows)
    results.to_csv(output_dir / "sanity_episode_results.csv", index=False)
    per_label = (
        results.groupby(
            [
                "feature_source",
                "support_protocol",
                "control",
                "k",
                "label",
            ],
            as_index=False,
        )[["query_prevalence", "auprc", "auprc_lift_over_prevalence"]]
        .mean()
    )
    macro = (
        per_label.groupby(
            ["feature_source", "support_protocol", "control", "k"],
            as_index=False,
        )[["query_prevalence", "auprc", "auprc_lift_over_prevalence"]]
        .mean()
    )
    per_label.to_csv(output_dir / "sanity_per_label_summary.csv", index=False)
    macro.to_csv(output_dir / "sanity_macro_summary.csv", index=False)

    shuffle_macro = macro[
        macro["feature_source"].eq("dinov2_vits14_geometry")
        & macro["support_protocol"].eq("abundant_negatives")
        & macro["control"].eq("query_label_shuffle")
    ]
    maximum_shuffle_gap = float(
        np.max(
            np.abs(
                shuffle_macro["auprc"].to_numpy()
                - shuffle_macro["query_prevalence"].to_numpy()
            )
        )
    )

    train_normalized = l2_normalize(train.embeddings)
    validation_normalized = l2_normalize(validation.embeddings)
    similarities = validation_normalized @ train_normalized.T
    nearest_train = similarities.argmax(axis=1)
    nearest_similarity = similarities.max(axis=1)
    candidate_mask = nearest_similarity >= NEAR_DUPLICATE_SCREENING_COSINE
    near_duplicate_candidates = pd.DataFrame(
        {
            "validation_record_id": validation.record_ids[candidate_mask],
            "nearest_train_record_id": train.record_ids[
                nearest_train[candidate_mask]
            ],
            "cosine_similarity": nearest_similarity[candidate_mask],
            "status": "screening_candidate_not_confirmed_duplicate",
        }
    )
    near_duplicate_candidates.to_csv(
        output_dir / "near_duplicate_screening_candidates.csv", index=False
    )

    summary = {
        "version": protocol["version"],
        "status": "PASS" if maximum_shuffle_gap <= 0.02 else "FAIL",
        "checks": {
            "known_duplicate_or_related_groups_crossing_train_validation": (
                known_group_crossings
            ),
            "shuffle_macro_auprc_max_absolute_gap_from_prevalence": (
                maximum_shuffle_gap
            ),
            "shuffle_tolerance": 0.02,
            "near_duplicate_cosine_screening_threshold": (
                NEAR_DUPLICATE_SCREENING_COSINE
            ),
            "near_duplicate_screening_candidates": int(candidate_mask.sum()),
            "test_records_accessed": 0,
        },
        "feature_sources": {
            "dinov2_vits14_geometry": (
                "Frozen DINOv2-S/14 geometry-only CLS representation"
            ),
            "acquisition_metadata_proxy": metadata_description,
        },
        "interpretation": (
            "Experiment 0 is a protocol sanity check. Observed model values are "
            "development results, not final-test or clinical performance claims. "
            "Episode replicates share the same validation queries and must not be "
            "treated as independent bootstrap units."
        ),
    }
    (output_dir / "sanity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nMacro summaries:\n" + macro.to_string(index=False))
    if summary["status"] != "PASS":
        raise RuntimeError("Few-shot sanity controls failed")


if __name__ == "__main__":
    main()

