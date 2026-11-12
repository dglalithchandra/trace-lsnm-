"""Descriptive cross-reported-domain retrieval diagnostic on development data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_trace_retrieval_dev import OUTPUT as BASE_OUTPUT, TRAIN, VALIDATION, TOP_K, metadata, random_hit_at_k
from scripts.freeze_trace_final_plan import sha256
from trace_protocols import l2_normalize, load_feature_split

OUT = ROOT / "outputs/trace_retrieval_cross_domain_dev"


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Cross-domain diagnostic already exists")
    train = load_feature_split(TRAIN, "train")
    query = load_feature_split(VALIDATION, "validation")
    if train.class_order != query.class_order or len(train.record_ids) != 1917 or len(query.record_ids) != 639:
        raise RuntimeError("Development feature schema changed")
    info = metadata(set(train.record_ids) | set(query.record_ids))
    train_domains = np.asarray([info[str(record_id)]["camera_domain"] for record_id in train.record_ids])
    query_domains = np.asarray([info[str(record_id)]["camera_domain"] for record_id in query.record_ids])
    similarities = l2_normalize(query.embeddings) @ l2_normalize(train.embeddings).T
    selected = np.empty((len(query.record_ids), TOP_K), dtype=np.int32)
    available = np.empty(len(query.record_ids), dtype=np.int32)
    rows = []
    for query_index, query_id in enumerate(query.record_ids):
        permitted = train_domains != query_domains[query_index]
        permitted &= np.asarray([info[str(record_id)]["split_group_id"] != info[str(query_id)]["split_group_id"] for record_id in train.record_ids])
        available[query_index] = int(permitted.sum())
        if available[query_index] < TOP_K:
            raise RuntimeError(f"Too few different-domain examples for {query_id}")
        candidate_scores = similarities[query_index].copy()
        candidate_scores[~permitted] = -np.inf
        selected[query_index] = np.argsort(-candidate_scores, kind="stable")[:TOP_K]
        for rank, train_index in enumerate(selected[query_index], 1):
            rows.append({"query_record_id": str(query_id), "support_record_id": str(train.record_ids[train_index]), "rank": rank, "cosine_similarity": float(candidate_scores[train_index]), "query_reported_domain": query_domains[query_index], "support_reported_domain": train_domains[train_index], "support_positive_labels_json": json.dumps([label for label, value in zip(train.class_order, train.labels[train_index]) if value == 1])})
    novel_labels = json.loads((ROOT / "data/protocols/trace_fewshot_v1/protocol.json").read_text(encoding="utf-8"))["novel_labels"]
    unrestricted = pd.read_csv(BASE_OUTPUT / "novel_label_retrieval.csv").set_index("label")
    results = []
    for label in novel_labels:
        column = train.class_order.index(label)
        positive_queries = np.flatnonzero(query.labels[:, column] == 1)
        observed = train.labels[selected[positive_queries], column]
        has_positive_support = np.asarray([int(train.labels[train_domains != query_domains[index], column].sum()) >= 1 for index in positive_queries])
        random_hits = []
        for index in positive_queries:
            pool = train.labels[train_domains != query_domains[index], column]
            random_hits.append(random_hit_at_k(len(pool), int(pool.sum()), TOP_K))
        results.append({"label": label, "positive_queries": len(positive_queries), "queries_with_other_domain_positive_support": int(has_positive_support.sum()), "other_domain_recorded_label_fraction_at_5": float(observed.mean()) if len(positive_queries) else np.nan, "other_domain_hit_at_5": float(observed.any(axis=1).mean()) if len(positive_queries) else np.nan, "other_domain_random_hit_at_5": float(np.mean(random_hits)) if random_hits else np.nan, "unrestricted_recorded_label_fraction_at_5": float(unrestricted.loc[label, "recorded_label_fraction_at_5"]), "unrestricted_hit_at_5": float(unrestricted.loc[label, "query_hit_at_5"])})
    OUT.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(OUT / "cross_domain_top5.csv", index=False)
    pd.DataFrame(results).to_csv(OUT / "novel_label_comparison.csv", index=False)
    train_counts = {domain: int((train_domains == domain).sum()) for domain in sorted(set(train_domains))}
    random_same_domain = float(np.mean([train_counts[domain] / len(train_domains) for domain in query_domains]))
    base_summary = json.loads((BASE_OUTPUT / "summary.json").read_text(encoding="utf-8"))
    summary = {"status": "DESCRIPTIVE_DEVELOPMENT_ONLY", "train_images": len(train.record_ids), "validation_images": len(query.record_ids), "different_reported_domain_pool_min": int(available.min()), "different_reported_domain_pool_median": float(np.median(available)), "observed_unrestricted_same_domain_fraction": base_summary["mean_same_reported_camera_domain_fraction"], "random_pool_same_domain_fraction": random_same_domain, "reported_domain_train_counts": train_counts, "feature_sha256": {"train": sha256(TRAIN), "validation": sha256(VALIDATION)}, "limitations": ["Reported domains are partly inferred from image dimensions", "Pool sizes and case mix differ by domain", "Only 5–17 positive validation images per novel label", "RFMiD patient IDs are unavailable", "Recorded-label overlap is not clinical retrieval precision"], "protected_test_images_accessed": 0}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "observed_same_domain": summary["observed_unrestricted_same_domain_fraction"], "random_same_domain": random_same_domain, "test_images_accessed": 0}, indent=2))


if __name__ == "__main__":
    main()
