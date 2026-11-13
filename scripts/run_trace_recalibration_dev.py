#!/usr/bin/env python3
"""Run separated TRACE development recalibration from a JSON record list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_bench_v2 import CalibrationRecord, run_label_budget_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--frozen-threshold", type=float, default=0.5)
    parser.add_argument(
        "--source-score-type", choices=("ranking_score", "probability"), default="probability"
    )
    parser.add_argument("--source-calibration-status", default="frozen_source_probability")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records = [CalibrationRecord(**item) for item in payload]
    result = run_label_budget_analysis(
        records,
        repeats=args.repeats,
        global_seed=args.seed,
        frozen_source_threshold=args.frozen_threshold,
        source_score_type=args.source_score_type,
        source_calibration_status=args.source_calibration_status,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} with {len(result['rows'])} budget-repeat rows")


if __name__ == "__main__":
    main()
