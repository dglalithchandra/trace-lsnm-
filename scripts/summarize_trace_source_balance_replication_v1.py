"""Summarize the three predeclared A1 replication seeds without test access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_models.dinov2 import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-tags", nargs=3, required=True)
    args = parser.parse_args()
    if len(set(args.candidate_tags)) != 3:
        raise RuntimeError("Exactly three unique predeclared candidate tags are required")

    rows: list[dict[str, object]] = []
    label_frames: list[pd.DataFrame] = []
    baseline_reference: dict[int, tuple[float, float]] | None = None
    source_summaries: dict[str, str] = {}
    for tag in args.candidate_tags:
        run_dir = ROOT / "outputs/trace_source_balance_v1" / tag
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "DEVELOPMENT_ONLY_NOT_PROTECTED_TEST":
            raise RuntimeError(f"Unexpected evaluation status for {tag}")
        if summary.get("protected_test_records_accessed") != 0:
            raise RuntimeError(f"Protected-test access reported for {tag}")

        macro = pd.DataFrame(summary["macro_results"])
        baseline = macro.loc[macro.model.eq("a0_frozen_dinov2_s")]
        current_reference = {
            int(row.k): (float(row.macro_auprc), float(row.macro_auroc))
            for row in baseline.itertuples(index=False)
        }
        if baseline_reference is None:
            baseline_reference = current_reference
        elif current_reference != baseline_reference:
            raise RuntimeError("Frozen A0 baseline changed between seed evaluations")

        for k in (3, 5):
            candidate = macro.loc[macro.model.eq("a1_core_ssl") & macro.k.eq(k)].iloc[0]
            interval = summary["paired_intervals"][f"k{k}"]
            rows.append({
                "candidate_tag": tag,
                "k": k,
                "macro_auprc": float(candidate.macro_auprc),
                "macro_auroc": float(candidate.macro_auroc),
                "delta_macro_auprc": float(interval["delta_macro_auprc"]),
                "within_seed_interval_low": float(interval["ci95_labels_then_support_episodes"][0]),
                "within_seed_interval_high": float(interval["ci95_labels_then_support_episodes"][1]),
                "screen_gate_passed": bool(summary["gate"]["advance_to_replication"]),
            })
        labels = pd.read_csv(run_dir / "paired_per_label.csv")
        labels.insert(0, "candidate_tag", tag)
        label_frames.append(labels)
        source_summaries[tag] = sha256_file(summary_path)

    output = ROOT / "outputs/trace_source_balance_v1/replication"
    output.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output / "per_seed_results.csv", index=False)
    aggregate = per_seed.groupby("k", as_index=False).agg(
        seeds=("candidate_tag", "nunique"),
        mean_macro_auprc=("macro_auprc", "mean"),
        sd_macro_auprc=("macro_auprc", "std"),
        mean_macro_auroc=("macro_auroc", "mean"),
        mean_delta_macro_auprc=("delta_macro_auprc", "mean"),
        sd_delta_macro_auprc=("delta_macro_auprc", "std"),
        positive_delta_seeds=("delta_macro_auprc", lambda values: int((values > 0).sum())),
        margin_0_02_seeds=("delta_macro_auprc", lambda values: int((values >= 0.02).sum())),
    )
    aggregate.to_csv(output / "aggregate_results.csv", index=False)

    per_label = pd.concat(label_frames, ignore_index=True)
    per_label.to_csv(output / "per_seed_per_label.csv", index=False)
    label_aggregate = per_label.groupby(["k", "label"], as_index=False).agg(
        mean_delta_auprc=("delta_auprc", "mean"),
        min_delta_auprc=("delta_auprc", "min"),
        max_delta_auprc=("delta_auprc", "max"),
        positive_seeds=("delta_auprc", lambda values: int((values > 0).sum())),
    )
    label_aggregate.to_csv(output / "aggregate_per_label.csv", index=False)

    positive_by_k = {
        int(row.k): int(row.positive_delta_seeds)
        for row in aggregate.itertuples(index=False)
    }
    summary = {
        "status": "THREE_SEED_DEVELOPMENT_REPLICATION_COMPLETE",
        "candidate_tags": args.candidate_tags,
        "aggregate_results": aggregate.to_dict("records"),
        "replication_rule": "AUPRC delta is positive in at least two of three seeds at both K=3 and K=5.",
        "effect_sign_replicated": bool(all(positive_by_k[k] >= 2 for k in (3, 5))),
        "source_summary_sha256": source_summaries,
        "protected_test_records_accessed": 0,
        "limitations": [
            "Three seeds are too few for a reliable seed-level confidence interval.",
            "Within-seed intervals reuse the same validation queries and are not independent across seeds.",
            "Patient identifiers are unavailable.",
            "This is development evidence and is not a fresh external confirmation.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(per_seed.to_string(index=False))
    print(aggregate.to_string(index=False))
    print(json.dumps({"effect_sign_replicated": summary["effect_sign_replicated"]}, indent=2))


if __name__ == "__main__":
    main()
