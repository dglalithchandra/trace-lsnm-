"""Recalibrate the existing frozen DR baseline on a deterministic Paraguay split."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_bench_v2 import CalibrationRecord, run_label_budget_analysis


def main() -> None:
    scores_path = ROOT / "outputs/paraguay_external_shift_v1/scores.npz"
    manifest_path = ROOT / "data/manifests/paraguay_dr_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = {row["record_id"]: row for row in csv.DictReader(handle)}
    with np.load(scores_path, allow_pickle=False) as data:
        ids = data["record_ids"]
        labels = data["labels"]
        scores = data["probabilities"]
        threshold = float(data["threshold"][0])
        records = []
        for record_id, label, score in zip(ids, labels, scores, strict=True):
            row = manifest[str(record_id)]
            if row["protected_final_test"].lower() != "false" or int(row["any_dr"]) != int(label):
                raise RuntimeError(f"Invalid record role or label: {record_id}")
            group = row["exact_duplicate_group_id"] or str(record_id)
            bucket = int(hashlib.sha256(("trace-paraguay-v1:" + group).encode()).hexdigest()[:8], 16) % 10
            records.append(CalibrationRecord(str(record_id), "calibration" if bucket < 4 else "evaluation", int(label), float(score), group))
    result = run_label_budget_analysis(records, repeats=100, global_seed=20261112, frozen_source_threshold=threshold)
    result.update({"scope": "exploratory_previously_used_paraguay_baseline_not_A1", "patient_independence": "unknown_patient_ids_unavailable", "split_rule": "sha256 duplicate group or record id; 40 percent calibration buckets", "source_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "partition_counts": dict(Counter(r.partition for r in records)), "protected_test_records_accessed": 0})
    output = ROOT / "outputs/paraguay_recalibration_exploratory_v1/result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for budget in (0, 5, 10, 20, 50):
        rows = [row for row in result["rows"] if row["budget"] == budget]
        valid = [row for row in rows if row["status"].startswith("feasible")]
        print(budget, "feasible", len(valid), "/", len(rows))
        for category, metric in (("threshold", "sensitivity"), ("threshold", "specificity"), ("calibration", "brier_score"), ("calibration", "ece_equal_width")):
            values = [r[category][metric]["value"] for r in valid if r[category][metric]["value"] is not None]
            if values:
                print(category, metric, "median", round(float(np.median(values)), 4))
    print("partition_counts", result["partition_counts"], "frozen_threshold", threshold)


if __name__ == "__main__":
    main()
