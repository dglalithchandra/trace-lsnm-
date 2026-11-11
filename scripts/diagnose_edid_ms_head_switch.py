"""Post-hoc diagnostic: keep the RETFound MS head fixed across support sizes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dinov2_fewshot_benchmark import parse_indices, score_head
from trace_protocols import l2_normalize, load_feature_split


def main() -> None:
    train = load_feature_split(ROOT / "data/features/rfmid_train_retfound_dinov2_meh_geometry448_v1.npz", "train")
    query = load_feature_split(ROOT / "data/features/edid_exploratory_retfound_geometry448_v1.npz", "validation")
    audit = pd.read_csv(ROOT / "data/protocols/trace_bench_v2/edid_preprocessing_audit.csv")
    schedule = pd.read_csv(ROOT / "data/protocols/trace_bench_v2/edid_support_schedule.csv")
    schedule = schedule[schedule.label.eq("MS")]
    if len(schedule) != 150:
        raise RuntimeError("MS schedule changed")
    ms_index = train.class_order.index("MS")
    mask = audit.class_folder.isin(("Macular Scar", "Healthy")).to_numpy()
    truth = query.labels[mask, query.class_order.index("MS")]
    if (truth.sum(), len(truth)) != (326, 1103):
        raise RuntimeError("EDID MS query changed")
    support_index = {item: index for index, item in enumerate(train.record_ids)}
    negatives = np.flatnonzero(train.labels[:, ms_index] == 0)
    train_x, query_x = l2_normalize(train.embeddings), l2_normalize(query.embeddings)
    results = []
    for head, parameter in (("knn", 1), ("logistic", 1.0)):
        for k in (1, 3, 5):
            predictions = []
            for episode in schedule[schedule.k.eq(k)].itertuples(index=False):
                positives, _ = parse_indices(episode, support_index)
                scores = score_head(head, train_x, query_x, positives, negatives, parameter, int(episode.seed))
                predictions.append(scores[mask])
            average = np.mean(predictions, axis=0)
            results.append({"encoder": "retfound", "label": "MS", "head": head, "parameter": parameter, "k": k, "support_subsets": len(predictions), "auprc": average_precision_score(truth, average), "auroc": roc_auc_score(truth, average), "analysis_status": "POST_HOC_HEAD_SWITCH_DIAGNOSTIC_NOT_MODEL_SELECTION"})
            print(f"{head} K={k}: AUPRC {results[-1]['auprc']:.3f}, AUROC {results[-1]['auroc']:.3f}", flush=True)
    out = ROOT / "outputs/edid_exploratory_transfer_v1/retfound/ms_same_head_diagnostic.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    (out.parent / "ms_same_head_diagnostic_summary.json").write_text(json.dumps({"status": "POST_HOC_DIAGNOSTIC", "head_choices": {"knn": 1, "logistic_C": 1.0}, "result_rows": len(results), "warning": "These heads were compared after inspecting EDID results; do not select a head from this comparison and call EDID confirmatory.", "protected_test_images_opened": 0}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
