#!/usr/bin/env python3
"""Run the dependency-light TRACE-Bench v2 synthetic smoke workflow."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_bench_v2 import (
    CalibrationRecord,
    CandidateRecord,
    PairedObservation,
    build_episode_plan,
    episode_plan_hash,
    evaluate_binary,
    load_exchange,
    macro_metrics,
    paired_bootstrap_difference,
    run_label_budget_analysis,
)
from trace_explanations import load_graph, lookup_explanations


DEFAULT_OUTPUT = ROOT / "outputs" / "trace_bench_v2" / "synthetic_smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run without images, weights, network access or GPU.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20261112)
    return parser.parse_args()


def synthetic_episode_records() -> list[CandidateRecord]:
    records = [
        CandidateRecord(
            record_id=f"synthetic_train:{index:02d}",
            split="development_train",
            labels={"SYN_A": int(index < 6), "SYN_B": int(index % 2 == 0)},
            patient_id=f"synthetic_train_patient:{index:02d}",
            group_id=f"synthetic_train_group:{index:02d}",
        )
        for index in range(12)
    ]
    records.extend(
        CandidateRecord(
            record_id=f"synthetic_query:{index:02d}",
            split="evaluation",
            labels={"SYN_A": int(index % 2 == 0), "SYN_B": int(index >= 3)},
            patient_id=f"synthetic_query_patient:{index:02d}",
            group_id=f"synthetic_query_group:{index:02d}",
        )
        for index in range(6)
    )
    return records


def synthetic_calibration_records() -> list[CalibrationRecord]:
    records = [
        CalibrationRecord(
            record_id=f"synthetic_calibration:{index:02d}",
            partition="calibration",
            target=index % 2,
            score=0.15 + 0.7 * (index % 2) + 0.01 * (index % 3),
            group_id=f"synthetic_calibration_group:{index:02d}",
        )
        for index in range(60)
    ]
    records.extend(
        CalibrationRecord(
            record_id=f"synthetic_evaluation:{index:02d}",
            partition="evaluation",
            target=target,
            score=score,
            group_id=f"synthetic_evaluation_group:{index:02d}",
        )
        for index, (target, score) in enumerate(
            [(0, 0.1), (1, 0.9), (0, 0.25), (1, 0.75), (0, 0.4), (1, 0.6)]
        )
    )
    return records


def synthetic_paired_observations() -> list[PairedObservation]:
    observations = []
    for label_id in ("SYN_A", "SYN_B"):
        for replicate in range(2):
            for index, target in enumerate((0, 1, 0, 1, 0, 1)):
                score_a = (0.1, 0.9, 0.2, 0.8, 0.3, 0.7)[index]
                score_b = (0.2, 0.8, 0.35, 0.65, 0.4, 0.6)[index]
                observations.append(
                    PairedObservation(
                        label_id=label_id,
                        episode_id=f"{label_id}:r{replicate}",
                        query_record_id=f"synthetic_query:{index:02d}",
                        query_group_id=f"synthetic_query_group:{index:02d}",
                        target=target,
                        score_a=score_a,
                        score_b=score_b,
                    )
                )
    return observations


def main() -> None:
    args = parse_args()
    if not args.synthetic:
        raise SystemExit("Only --synthetic is supported until Lalith supplies a development exchange.")

    exchange_path = (
        ROOT / "trace_bench_v2" / "examples" / "trace_dev_exchange_v1.synthetic.json"
    )
    exchange = load_exchange(exchange_path)
    plan = build_episode_plan(
        synthetic_episode_records(),
        ["SYN_A", "SYN_B"],
        k_values=(1, 3, 5, 10, 20),
        replicates=2,
        global_seed=args.seed,
    )
    plan_sha256 = episode_plan_hash(plan)
    per_label = [
        {
            "label_id": "SYN_A",
            "result": evaluate_binary(
            [0, 1, 0, 1, 0, 1],
            [0.1, 0.9, 0.25, 0.75, 0.4, 0.6],
            score_type="probability",
            calibration_status="uncalibrated_probability",
        ),
        },
        {
            "label_id": "SYN_B",
            "result": evaluate_binary(
            [0, 0, 1, 1, 0, 1],
            [0.2, 0.3, 0.8, 0.7, 0.4, 0.6],
            score_type="probability",
            calibration_status="uncalibrated_probability",
        ),
        },
    ]
    paired = paired_bootstrap_difference(
        synthetic_paired_observations(), resamples=100, seed=args.seed
    )
    recalibration = run_label_budget_analysis(
        synthetic_calibration_records(), repeats=2, global_seed=args.seed
    )
    graph = load_graph(
        ROOT
        / "data"
        / "knowledge_sources"
        / "curated"
        / "trace_explanation_graph_v1.json"
    )
    explanation = lookup_explanations(
        graph,
        [{"label_id": "RT", "score": 0.73}, {"label_id": "MS", "score": 0.41}],
        qc_flags=[
            {
                "name": "synthetic_retinal_coverage_fraction",
                "value": 0.9,
                "measurement_source": "synthetic_smoke",
            }
        ],
        provenance={"data_scope": "synthetic", "model_id": exchange["model"]["model_id"]},
    )
    result = {
        "status": "PASS_SYNTHETIC_SMOKE",
        "seed": args.seed,
        "exchange_contract_version": exchange["contract_version"],
        "episode_plan": {
            "hash": plan_sha256,
            "episode_count": len(plan.episodes),
            "unsupported": [asdict(item) for item in plan.unsupported],
            "patient_independence": plan.patient_independence,
        },
        "per_label_metrics": per_label,
        "macro_metrics": macro_metrics(
            [item["result"] for item in per_label],
            episode_plan_sha256=plan_sha256,
            seeds=[episode.seed for episode in plan.episodes],
        ),
        "paired_uncertainty": paired,
        "recalibration": {
            "budgets": recalibration["budgets"],
            "row_count": len(recalibration["rows"]),
            "evaluation_labels_used_for_fitting": recalibration[
                "evaluation_labels_used_for_fitting"
            ],
            "status_counts": {
                status: sum(row["status"] == status for row in recalibration["rows"])
                for status in sorted({row["status"] for row in recalibration["rows"]})
            },
        },
        "explanation": explanation,
        "resource_boundary": "no_images_no_weights_no_network_no_gpu",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"PASS synthetic TRACE-Bench v2 smoke: {len(plan.episodes)} episodes, "
        f"{len(recalibration['rows'])} recalibration rows -> {args.output}"
    )


if __name__ == "__main__":
    main()
