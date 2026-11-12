"""Development-only labelled-example retrieval with frozen DINOv2-L features.

Retrieval shows recorded RFMiD labels. It does not infer missing findings or
prove that two images came from different patients (patient IDs are absent).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_trace_final_plan import sha256
from trace_protocols import l2_normalize, load_feature_split

TRAIN = ROOT / "data/features/rfmid_train_dinov2_vitl14_geometry448_v1.npz"
VALIDATION = ROOT / "data/features/rfmid_validation_dinov2_vitl14_geometry448_v1.npz"
MANIFEST = ROOT / "data/manifests/trace_analysis_manifest.csv"
OUTPUT = ROOT / "outputs/trace_retrieval_v1_dev"
TOP_K = 5


def metadata(ids: set[str]) -> dict[str, dict[str, str]]:
    rows = {}
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["record_id"] not in ids:
                continue
            if row["dataset"] != "RFMiD" or row["final_split"] not in {"train", "validation"}:
                raise RuntimeError("Retrieval requested a nondevelopment record")
            rows[row["record_id"]] = {key: row[key] for key in ("record_id", "source_split", "source_image_id", "final_split", "split_group_id", "camera_domain", "camera_domain_source", "patient_id_status", "quality_label_status")}
    if set(rows) != ids:
        raise RuntimeError("Development metadata is missing or duplicated")
    return rows


def random_hit_at_k(population: int, positive: int, k: int) -> float:
    if population < k or positive < 0 or positive > population:
        raise ValueError("Invalid retrieval pool counts")
    miss = 1.0
    for index in range(k):
        miss *= (population - positive - index) / (population - index)
    return 1.0 - max(miss, 0.0)


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite retrieval development output: {OUTPUT}")
    train = load_feature_split(TRAIN, "train")
    validation = load_feature_split(VALIDATION, "validation")
    if train.class_order != validation.class_order or train.embeddings.shape != (1917, 1024) or validation.embeddings.shape != (639, 1024):
        raise RuntimeError("Frozen DINOv2-L feature pairing changed")
    if set(train.record_ids) & set(validation.record_ids):
        raise RuntimeError("Training and validation IDs overlap")
    info = metadata(set(train.record_ids) | set(validation.record_ids))
    train_x = l2_normalize(train.embeddings)
    query_x = l2_normalize(validation.embeddings)
    similarities = query_x @ train_x.T
    neighbor_rows = []
    neighbor_indices = np.empty((len(validation.record_ids), TOP_K), dtype=np.int32)
    for query_index, query_id in enumerate(validation.record_ids):
        query_meta = info[str(query_id)]
        allowed = np.array([info[str(train_id)]["split_group_id"] != query_meta["split_group_id"] for train_id in train.record_ids], dtype=bool)
        if allowed.sum() < TOP_K:
            raise RuntimeError("Too few independent known groups for retrieval")
        values = similarities[query_index].copy()
        values[~allowed] = -np.inf
        top = np.argsort(-values, kind="stable")[:TOP_K]
        neighbor_indices[query_index] = top
        for rank, train_index in enumerate(top, 1):
            train_id = str(train.record_ids[train_index])
            support_meta = info[train_id]
            positive_labels = [label for label, value in zip(train.class_order, train.labels[train_index]) if value == 1]
            neighbor_rows.append({"query_record_id": str(query_id), "support_record_id": train_id, "rank": rank, "cosine_similarity": float(values[train_index]), "support_positive_labels_json": json.dumps(positive_labels), "support_source_split": support_meta["source_split"], "support_source_image_id": support_meta["source_image_id"], "support_split_group_id": support_meta["split_group_id"], "support_camera_domain": support_meta["camera_domain"], "support_camera_domain_source": support_meta["camera_domain_source"], "same_reported_camera_domain": bool(support_meta["camera_domain"] == query_meta["camera_domain"])})
    novel_labels = json.loads((ROOT / "data/protocols/trace_fewshot_v1/protocol.json").read_text(encoding="utf-8"))["novel_labels"]
    metric_rows = []
    for label in novel_labels:
        column = train.class_order.index(label)
        query_positive = np.flatnonzero(validation.labels[:, column] == 1)
        train_positive = int(train.labels[:, column].sum())
        if len(query_positive) < 5 or train_positive < TOP_K:
            metric_rows.append({"label": label, "query_positive_count": len(query_positive), "train_positive_count": train_positive, "status": "descriptive_insufficient_cases", "recorded_label_fraction_at_5": np.nan, "query_hit_at_5": np.nan, "random_pool_fraction": train_positive / len(train.record_ids), "random_hit_at_5": random_hit_at_k(len(train.record_ids), train_positive, TOP_K)})
            continue
        hits = train.labels[neighbor_indices[query_positive], column]
        metric_rows.append({"label": label, "query_positive_count": len(query_positive), "train_positive_count": train_positive, "status": "descriptive", "recorded_label_fraction_at_5": float(hits.mean()), "query_hit_at_5": float(hits.any(axis=1).mean()), "random_pool_fraction": train_positive / len(train.record_ids), "random_hit_at_5": random_hit_at_k(len(train.record_ids), train_positive, TOP_K)})
    neighbors = pd.DataFrame(neighbor_rows)
    metrics = pd.DataFrame(metric_rows)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    neighbors.to_csv(OUTPUT / "validation_top5.csv", index=False)
    metrics.to_csv(OUTPUT / "novel_label_retrieval.csv", index=False)
    summary = {"status": "DEVELOPMENT_ONLY_RETRIEVAL", "encoder": "frozen_dinov2_l", "query_split": "validation", "index_split": "train", "index_images": len(train.record_ids), "query_images": len(validation.record_ids), "top_k": TOP_K, "index_feature_sha256": sha256(TRAIN), "query_feature_sha256": sha256(VALIDATION), "manifest_sha256": sha256(MANIFEST), "retrieval_rows": len(neighbors), "mean_same_reported_camera_domain_fraction": float(neighbors.same_reported_camera_domain.mean()), "camera_metadata_warning": "RFMiD camera domains are inferred from image dimensions rather than verified per-image camera records", "label_warning": "Recorded-label overlap is not clinical precision because missing labels are not confirmed negatives", "patient_warning": "No RFMiD patient IDs; known duplicate groups excluded but different-eye retrieval cannot be guaranteed", "protected_test_images_accessed": 0}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "retrieval_rows": len(neighbors), "novel_labels": len(metrics), "test_images_accessed": 0}, indent=2))


if __name__ == "__main__":
    main()
