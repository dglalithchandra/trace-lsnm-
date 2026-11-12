"""Evaluate a matched TRACE source-balance arm against its intended reference."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_trace_source_balance_v1 import paired_hierarchical_interval
from scripts.run_dinov2_fewshot_benchmark import parse_indices, score_head
from trace_models.dinov2 import sha256_file
from trace_protocols import l2_normalize, load_feature_split


def feature_paths(tag: str) -> dict[str, Path]:
    directory = ROOT / "data/features"
    if tag == "frozen_dinov2_s":
        return {
            split: directory / f"rfmid_{split}_dinov2_vits14_geometry_v1.npz"
            for split in ("train", "validation")
        }
    if not re.fullmatch(r"[a-z0-9_]+", tag):
        raise RuntimeError("Feature tags must use lowercase letters, digits and underscores")
    return {
        split: directory / f"rfmid_{split}_trace_source_balance_{tag}.npz"
        for split in ("train", "validation")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-tag", required=True)
    parser.add_argument("--reference-tag", required=True)
    parser.add_argument("--contrast-name", required=True)
    args = parser.parse_args()
    if args.candidate_tag == args.reference_tag:
        raise RuntimeError("Candidate and reference tags must differ")
    if not re.fullmatch(r"[a-z0-9_]+", args.contrast_name):
        raise RuntimeError("Contrast name must use lowercase letters, digits and underscores")

    tags = {"candidate": args.candidate_tag, "reference": args.reference_tag}
    paths = {role: feature_paths(tag) for role, tag in tags.items()}
    episode_path = ROOT / "data/protocols/trace_fewshot_v1/episode_plan.csv"
    episodes = pd.read_csv(episode_path)
    episodes = episodes.loc[episodes.k.isin([3, 5])].copy()
    if episodes.groupby("k").size().to_dict() != {3: 550, 5: 550}:
        raise RuntimeError("Frozen K=3/K=5 episode plan changed")

    rows: list[dict[str, object]] = []
    reference_schema: tuple[np.ndarray, ...] | None = None
    feature_hashes: dict[str, dict[str, str]] = {}
    for role in ("reference", "candidate"):
        train = load_feature_split(paths[role]["train"], "train")
        query = load_feature_split(paths[role]["validation"], "validation")
        if len(train.record_ids) != 1917 or len(query.record_ids) != 639:
            raise RuntimeError("Unexpected RFMiD development record count")
        if train.class_order != query.class_order:
            raise RuntimeError("Train and validation class orders differ")
        schema = (train.record_ids, train.labels, query.record_ids, query.labels)
        if reference_schema is None:
            reference_schema = tuple(item.copy() for item in schema)
        elif not all(np.array_equal(a, b) for a, b in zip(reference_schema, schema, strict=True)):
            raise RuntimeError("Candidate and reference are not paired by IDs and labels")

        train_x = l2_normalize(train.embeddings)
        query_x = l2_normalize(query.embeddings)
        train_index = {record_id: index for index, record_id in enumerate(train.record_ids)}
        for number, episode in enumerate(episodes.itertuples(index=False), 1):
            class_index = train.class_order.index(str(episode.label))
            positives, _ = parse_indices(episode, train_index)
            negatives = np.flatnonzero(train.labels[:, class_index] == 0)
            truth = query.labels[:, class_index]
            scores = score_head(
                "logistic", train_x, query_x, positives, negatives, 1.0, int(episode.seed)
            )
            prevalence = float(truth.mean())
            ap = float(average_precision_score(truth, scores))
            rows.append({
                "model": role,
                "feature_tag": tags[role],
                "episode_id": episode.episode_id,
                "label": str(episode.label),
                "k": int(episode.k),
                "replicate": int(episode.replicate),
                "auprc": ap,
                "auroc": float(roc_auc_score(truth, scores)),
                "query_prevalence": prevalence,
                "ap_divided_by_prevalence": ap / prevalence,
            })
            if number % 250 == 0 or number == len(episodes):
                print(f"{role} {tags[role]}: {number}/{len(episodes)}", flush=True)
        feature_hashes[role] = {
            split: sha256_file(path) for split, path in paths[role].items()
        }

    output = ROOT / "outputs/trace_source_balance_v1" / args.contrast_name
    output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output / "episode_metrics.csv", index=False)
    per_label = results.groupby(["model", "feature_tag", "k", "label"], as_index=False).agg(
        auprc=("auprc", "mean"),
        auroc=("auroc", "mean"),
        ap_divided_by_prevalence=("ap_divided_by_prevalence", "mean"),
        query_prevalence=("query_prevalence", "first"),
    )
    per_label.to_csv(output / "per_label_results.csv", index=False)
    macro = per_label.groupby(["model", "feature_tag", "k"], as_index=False).agg(
        macro_auprc=("auprc", "mean"),
        macro_auroc=("auroc", "mean"),
        macro_ap_divided_by_prevalence=("ap_divided_by_prevalence", "mean"),
    )
    macro.to_csv(output / "macro_results.csv", index=False)

    intervals = {
        f"k{k}": paired_hierarchical_interval(results, "candidate", "reference", k)
        for k in (3, 5)
    }
    paired = per_label.pivot(index=["k", "label"], columns="model", values=["auprc", "auroc"])
    paired.columns = [f"{metric}_{model}" for metric, model in paired.columns]
    paired = paired.reset_index()
    paired["delta_auprc"] = paired.auprc_candidate - paired.auprc_reference
    paired["delta_auroc"] = paired.auroc_candidate - paired.auroc_reference
    paired.to_csv(output / "paired_per_label.csv", index=False)

    macro_pivot = macro.pivot(index="k", columns="model", values="macro_auroc")
    worst_macro_auroc_delta = float((macro_pivot.candidate - macro_pivot.reference).min())
    worst_label_delta = float(paired.delta_auprc.min())
    gate = {
        "k3_delta_at_least_0_02": bool(intervals["k3"]["delta_macro_auprc"] >= 0.02),
        "k5_delta_at_least_0_02": bool(intervals["k5"]["delta_macro_auprc"] >= 0.02),
        "macro_auroc_drop_no_more_than_0_01": bool(worst_macro_auroc_delta >= -0.01),
        "no_label_auprc_harmed_more_than_0_10": bool(worst_label_delta >= -0.10),
    }
    gate["screen_passed"] = all(gate.values())
    summary = {
        "status": "DEVELOPMENT_ONLY_MATCHED_CONTRAST",
        "contrast_name": args.contrast_name,
        "candidate_tag": args.candidate_tag,
        "reference_tag": args.reference_tag,
        "head": "logistic",
        "c": 1.0,
        "support_protocol": "abundant_negatives",
        "macro_results": macro.to_dict("records"),
        "paired_intervals": intervals,
        "gate": gate,
        "worst_per_label_delta_auprc": worst_label_delta,
        "worst_macro_auroc_delta": worst_macro_auroc_delta,
        "feature_hashes": feature_hashes,
        "episode_plan_sha256": sha256_file(episode_path),
        "protected_test_records_accessed": 0,
        "limitations": [
            "One matched training seed for this contrast.",
            "The same validation queries are reused across support episodes and arms.",
            "Patient identifiers are unavailable.",
            "The interval resamples labels and support episodes but holds query images fixed.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(macro.to_string(index=False))
    print(json.dumps({"paired_intervals": intervals, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
