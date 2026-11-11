"""Freeze RFMiD-v1-training support choices for exploratory EDID queries."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_protocols import load_feature_split

PROTOCOL = ROOT / "data/protocols/trace_fewshot_v1"
OUTPUT = ROOT / "data/protocols/trace_bench_v2/edid_support_schedule.csv"
SUMMARY = ROOT / "data/protocols/trace_bench_v2/edid_support_schedule_summary.json"
TRAIN = ROOT / "data/features/rfmid_train_dinov2_vits14_geometry_v1.npz"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    train = load_feature_split(TRAIN, "train")
    if len(train.record_ids) != 1917:
        raise RuntimeError("RFMiD training pool changed")
    ms = [row for row in load_rows(PROTOCOL / "episode_plan.csv") if row["label"] == "MS"]
    ode = [row for row in load_rows(PROTOCOL / "base_tuning_episode_plan.csv") if row["label"] == "ODE"]
    if len(ms) != 150 or len(ode) != 150:
        raise RuntimeError("Frozen MS or ODE episode schedule changed")
    rp_col = train.class_order.index("RP")
    positives = sorted(train.record_ids[train.labels[:, rp_col] == 1].tolist())
    negatives = sorted(train.record_ids[train.labels[:, rp_col] == 0].tolist())
    if len(positives) != 6:
        raise RuntimeError("Expected six RFMiD-v1-training RP positives")
    rp = []
    for k in (1, 3):
        for replicate, support in enumerate(itertools.combinations(positives, k)):
            episode_id = f"edid-exploratory-v1:RP:k{k}:r{replicate:03d}"
            seed = int.from_bytes(hashlib.sha256(episode_id.encode()).digest()[:4], "big")
            rng = np.random.default_rng(seed)
            strict_negative = sorted(rng.choice(negatives, size=k, replace=False).tolist())
            rp.append({
                "episode_id": episode_id,
                "label": "RP",
                "class_index": rp_col,
                "k": k,
                "replicate": replicate,
                "seed": seed,
                "support_positive_record_ids_json": json.dumps(support),
                "strict_support_negative_record_ids_json": json.dumps(strict_negative),
                "abundant_negative_pool_size": len(negatives),
                "query_split": "edid_exploratory",
                "query_policy": "frozen_edid_manifest_v2_target_vs_healthy",
            })
    combined = []
    for source, source_rows in (("MS", ms), ("RP", rp), ("ODE", ode)):
        for row in source_rows:
            item = dict(row)
            item["query_split"] = "edid_exploratory"
            item["query_policy"] = "frozen_edid_manifest_v2_target_vs_healthy"
            item["edid_target_class"] = {"MS": "Macular Scar", "RP": "Retinitis Pigmentosa", "ODE": "Disc Edema"}[source]
            item["edid_control_class"] = "Healthy"
            combined.append(item)
    if len(combined) != 326:
        raise RuntimeError("Unexpected episode count")
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    summary = {
        "status": "FROZEN_BEFORE_EDID_MODEL_SCORES",
        "episode_counts": {"MS": 150, "RP": 26, "ODE": 150},
        "rp_support_subsets": {"k1": 6, "k3": 20},
        "query_manifest": "data/manifests/edid_exploratory_manifest_v2.csv",
        "support_source": "RFMiD v1 training only",
        "test_records_accessed": 0,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
