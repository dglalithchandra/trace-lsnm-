"""Run the locked paired RFMiD v1 final comparison exactly once after release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_trace_final_v1_test_features import LOCK, MODELS, OUTPUT, RELEASE, verify_release
from scripts.freeze_trace_final_plan import sha256
from scripts.run_dinov2_fewshot_benchmark import parse_indices, score_head
from trace_protocols import l2_normalize, load_feature_split

TRAIN_FILES = {
    "frozen_dinov2_s": "rfmid_train_dinov2_vits14_geometry_v1.npz",
    "frozen_dinov2_l": "rfmid_train_dinov2_vitl14_geometry448_v1.npz",
    "frozen_retfound": "rfmid_train_retfound_dinov2_meh_geometry448_v1.npz",
    "plain_crop_matched_epoch5": "rfmid_train_trace_ssl_plain_crop_matched_epoch5.npz",
    "full_field_epoch5": "rfmid_train_trace_ssl_full_field_epoch5.npz",
}
PRIMARY = ("full_field_epoch5", "frozen_dinov2_l")
BOOTSTRAP_SEED = 20261112


def load_test(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"record_ids", "embeddings", "labels", "class_order", "split"} or str(payload["split"][0]) != "test":
            raise RuntimeError(f"Malformed locked test feature file: {path}")
        ids = payload["record_ids"].astype(str)
        x = payload["embeddings"].astype(np.float32)
        y = payload["labels"].astype(np.uint8)
        classes = tuple(payload["class_order"].astype(str).tolist())
    if x.shape[0] != len(ids) or y.shape != (len(ids), len(classes)) or len(ids) != len(set(ids)):
        raise RuntimeError(f"Unpaired test features: {path}")
    if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
        raise RuntimeError(f"Nonfinite features or invalid labels: {path}")
    return ids, x, y, classes


def score_groups(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort scores once and identify equal-score groups for exact weighted AP."""
    orders = np.argsort(-scores, axis=1, kind="mergesort")
    sorted_scores = np.take_along_axis(scores, orders, axis=1)
    starts = np.ones_like(sorted_scores, dtype=bool)
    ends = np.ones_like(sorted_scores, dtype=bool)
    starts[:, 1:] = sorted_scores[:, 1:] != sorted_scores[:, :-1]
    ends[:, :-1] = sorted_scores[:, :-1] != sorted_scores[:, 1:]
    return orders, starts, ends


def weighted_ap_for_all_episodes(orders: np.ndarray, starts: np.ndarray, ends: np.ndarray, truth: np.ndarray, query_weights: np.ndarray) -> np.ndarray:
    """Exact AP for resampled queries, including mixed-label score ties."""
    weighted_positive = (query_weights * truth)[orders]
    weighted_total = query_weights[orders]
    cumulative_positive = np.cumsum(weighted_positive, axis=1)
    cumulative_total = np.cumsum(weighted_total, axis=1)
    precision = np.divide(cumulative_positive, cumulative_total, out=np.zeros_like(cumulative_positive, dtype=np.float64), where=cumulative_total > 0)
    positive_before_group = np.maximum.accumulate(np.where(starts, cumulative_positive - weighted_positive, 0.0), axis=1)
    positive_in_group = cumulative_positive - positive_before_group
    return np.sum(np.where(ends, precision * positive_in_group, 0.0), axis=1) / weighted_positive.sum(axis=1)


def primary_bootstrap(scores: dict[str, dict[str, np.ndarray]], truths: dict[str, np.ndarray], labels: list[str], *, resamples: int = 5000) -> dict[str, object]:
    """Resample labels, support draws and within-label positive/negative queries."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    grouped_scores = {model: {label: score_groups(scores[model][label]) for label in labels} for model in PRIMARY}
    deltas = np.empty(resamples, dtype=np.float64)
    for iteration in range(len(deltas)):
        chosen_labels = rng.integers(0, len(labels), size=len(labels))
        label_differences = []
        for chosen in chosen_labels:
            label = labels[chosen]
            truth = truths[label]
            positive = np.flatnonzero(truth == 1)
            negative = np.flatnonzero(truth == 0)
            weights = np.bincount(np.concatenate((rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True))), minlength=len(truth)).astype(np.float64)
            support_draws = rng.integers(0, 50, size=50)
            newer = weighted_ap_for_all_episodes(*grouped_scores[PRIMARY[0]][label], truth, weights)[support_draws].mean()
            older = weighted_ap_for_all_episodes(*grouped_scores[PRIMARY[1]][label], truth, weights)[support_draws].mean()
            label_differences.append(newer - older)
        deltas[iteration] = float(np.mean(label_differences))
        if (iteration + 1) % 500 == 0:
            print(f"primary bootstrap {iteration + 1}/{len(deltas)}", flush=True)
    return {"resamples": len(deltas), "seed": BOOTSTRAP_SEED, "paired_95_percentile_interval": np.quantile(deltas, [0.025, 0.975]).tolist(), "resampling_units": "labels; support draws within sampled label; positive and negative test queries within label", "ties": "Equal-score groups receive the same precision, matching scikit-learn average_precision_score with sample weights"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-final-test", action="store_true")
    args = parser.parse_args()
    if not args.score_final_test:
        raise RuntimeError("Pass --score-final-test only after the locked release gate")
    excluded, _ = verify_release()
    summary_path = OUTPUT / "extraction_summary.json"
    if not summary_path.is_file() or (OUTPUT / "evaluation_summary.json").exists():
        raise RuntimeError("Five test feature files must exist and final evaluation must not be repeated")
    extraction = json.loads(summary_path.read_text(encoding="utf-8"))
    if extraction.get("release_sha256") != sha256(RELEASE) or extraction.get("excluded_record_ids") != excluded:
        raise RuntimeError("Test extraction and scoring releases differ")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "data/protocols/trace_fewshot_v1/protocol.json").read_text(encoding="utf-8"))
    episode_path = ROOT / "data/protocols/trace_fewshot_v1/episode_plan.csv"
    if sha256(episode_path) != lock["inputs_sha256"][episode_path.relative_to(ROOT).as_posix()]:
        raise RuntimeError("Support schedule changed")
    episodes = pd.read_csv(episode_path)
    labels = list(protocol["novel_labels"])
    if set(episodes.label) != set(labels) or len(episodes) != 1650 or set(episodes.k) != {1, 3, 5}:
        raise RuntimeError("Episode schedule differs from locked protocol")
    ordered_episodes = episodes.sort_values(["label", "k", "replicate"], kind="stable")
    pairs = None
    truth_by_label: dict[str, np.ndarray] = {}
    primary_scores: dict[str, dict[str, np.ndarray]] = {model: {} for model in PRIMARY}
    metric_rows = []
    score_arrays = {}
    model_files = {}
    for model, expected_dim in MODELS:
        train_path = ROOT / "data/features" / TRAIN_FILES[model]
        relative = train_path.relative_to(ROOT).as_posix()
        if sha256(train_path) != lock["inputs_sha256"][relative]:
            raise RuntimeError(f"Locked training features changed: {model}")
        test_path = OUTPUT / f"{model}_test_features.npz"
        if sha256(test_path) != extraction["feature_sha256"][model]:
            raise RuntimeError(f"Test feature hash changed: {model}")
        train = load_feature_split(train_path, "train")
        query_ids, query_x, query_y, classes = load_test(test_path)
        if train.embeddings.shape != (1917, expected_dim) or query_x.shape != (extraction["test_records_included"], expected_dim) or train.class_order != classes:
            raise RuntimeError(f"Feature shape or class order changed: {model}")
        if set(query_ids) & set(train.record_ids):
            raise RuntimeError("Training and test record IDs overlap")
        current_pairs = (train.record_ids, train.labels, query_ids, query_y)
        if pairs is None:
            pairs = tuple(array.copy() for array in current_pairs)
        elif not all(np.array_equal(a, b) for a, b in zip(pairs, current_pairs)):
            raise RuntimeError("Model feature files do not share IDs and labels")
        train_x = l2_normalize(train.embeddings)
        query_x = l2_normalize(query_x)
        train_index = {record_id: index for index, record_id in enumerate(train.record_ids)}
        predictions = np.empty((len(ordered_episodes), len(query_ids)), dtype=np.float32)
        for episode_index, episode in enumerate(ordered_episodes.itertuples(index=False)):
            label = str(episode.label)
            label_index = classes.index(label)
            positive, _ = parse_indices(episode, train_index)
            if len(positive) != int(episode.k) or not np.all(train.labels[positive, label_index] == 1):
                raise RuntimeError(f"Invalid positive support draw: {episode.episode_id}")
            negative = np.flatnonzero(train.labels[:, label_index] == 0)
            truth = query_y[:, label_index]
            scores = score_head("logistic", train_x, query_x, positive, negative, 1.0, int(episode.seed))
            if not np.isfinite(scores).all():
                raise RuntimeError(f"Nonfinite scores: {model} {episode.episode_id}")
            predictions[episode_index] = scores
            positives = int(truth.sum())
            negatives = int(len(truth) - positives)
            if positives >= 1 and negatives >= 1:
                metric_rows.append({"model": model, "episode_id": episode.episode_id, "label": label, "k": int(episode.k), "replicate": int(episode.replicate), "positives": positives, "negatives": negatives, "prevalence": float(truth.mean()), "auprc": float(average_precision_score(truth, scores)), "auroc": float(roc_auc_score(truth, scores))})
            if model in PRIMARY and int(episode.k) == 3:
                primary_scores[model].setdefault(label, []).append(scores.astype(np.float64))
                truth_by_label[label] = truth
            if (episode_index + 1) % 100 == 0:
                print(f"{model} episodes {episode_index + 1}/{len(ordered_episodes)}", flush=True)
        score_arrays[model] = predictions
        model_files[model] = {"train_sha256": sha256(train_path), "test_sha256": sha256(test_path)}
    metrics = pd.DataFrame(metric_rows)
    label_counts = pd.DataFrame([{"label": label, "positives": int(truth_by_label[label].sum()), "negatives": int(len(truth_by_label[label]) - truth_by_label[label].sum()), "prevalence": float(truth_by_label[label].mean())} for label in labels])
    per_label = metrics.groupby(["model", "label", "k"], as_index=False).agg(auprc=("auprc", "mean"), auroc=("auroc", "mean"), positives=("positives", "first"), negatives=("negatives", "first"), prevalence=("prevalence", "first"))
    per_label["auprc_over_prevalence"] = per_label.auprc / per_label.prevalence
    eligible = [label for label in labels if int(truth_by_label[label].sum()) >= 5 and int(len(truth_by_label[label]) - truth_by_label[label].sum()) >= 5]
    if len(eligible) < 8:
        bootstrap = None
        decision = "NO_PRIMARY_SUPERIORITY_CLAIM_FEWER_THAN_EIGHT_ELIGIBLE_LABELS"
    else:
        for model in PRIMARY:
            primary_scores[model] = {label: np.stack(primary_scores[model][label]) for label in eligible}
            if any(primary_scores[model][label].shape != (50, len(pairs[2])) for label in eligible):
                raise RuntimeError("Incomplete primary support draws")
        bootstrap = primary_bootstrap(primary_scores, truth_by_label, eligible)
        decision = "PRIMARY_COMPARISON_COMPUTED"
    primary_table = per_label[per_label.k.eq(3) & per_label.label.isin(eligible)]
    primary_macro = primary_table.groupby("model", as_index=False).agg(macro_auprc=("auprc", "mean"), macro_auroc=("auroc", "mean"), macro_auprc_over_prevalence=("auprc_over_prevalence", "mean"))
    learning_curve = per_label[per_label.label.isin(eligible)].groupby(["model", "k"], as_index=False).agg(macro_auprc=("auprc", "mean"), macro_auroc=("auroc", "mean"), macro_auprc_over_prevalence=("auprc_over_prevalence", "mean"))
    observed = None
    effect_at_least_point_02_supported = False
    if len(eligible) >= 8:
        observed = float(primary_macro.loc[primary_macro.model.eq(PRIMARY[0]), "macro_auprc"].iloc[0] - primary_macro.loc[primary_macro.model.eq(PRIMARY[1]), "macro_auprc"].iloc[0])
        lower = bootstrap["paired_95_percentile_interval"][0]
        decision = "EVIDENCE_OF_SUPERIORITY" if lower > 0 else "SUPERIORITY_NOT_ESTABLISHED"
        effect_at_least_point_02_supported = bool(lower > 0.02)
    macro_by_model = dict(zip(primary_macro.model, primary_macro.macro_auprc))
    secondary_paired_deltas = {f"full_field_epoch5_minus_{other}": float(macro_by_model["full_field_epoch5"] - macro_by_model[other]) for other in ("plain_crop_matched_epoch5", "frozen_dinov2_s", "frozen_retfound") if "full_field_epoch5" in macro_by_model and other in macro_by_model}
    np.savez_compressed(OUTPUT / "per_image_episode_scores.npz", record_ids=pairs[2], episode_ids=ordered_episodes.episode_id.to_numpy(dtype=str), model_names=np.asarray([name for name, _ in MODELS]), scores=np.stack([score_arrays[name] for name, _ in MODELS]))
    metrics.to_csv(OUTPUT / "episode_metrics.csv", index=False)
    label_counts.to_csv(OUTPUT / "label_counts.csv", index=False)
    per_label.to_csv(OUTPUT / "per_label_metrics.csv", index=False)
    primary_macro.to_csv(OUTPUT / "primary_macro_metrics.csv", index=False)
    learning_curve.to_csv(OUTPUT / "descriptive_learning_curve.csv", index=False)
    result = {"status": decision, "primary_contrast": f"{PRIMARY[0]} minus {PRIMARY[1]}", "eligible_labels": eligible, "eligible_label_count": len(eligible), "observed_macro_auprc_difference": observed, "primary_bootstrap": bootstrap, "effect_at_least_point_02_supported": effect_at_least_point_02_supported, "secondary_observed_paired_macro_deltas": secondary_paired_deltas, "query_count": len(pairs[2]), "excluded_record_ids": excluded, "model_feature_hashes": model_files, "episode_schedule_sha256": sha256(episode_path), "per_image_episode_scores_sha256": sha256(OUTPUT / "per_image_episode_scores.npz"), "software_versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__}, "caveats": ["Patient IDs unavailable", "Operational low-support labels are not clinical rarity estimates", "RFMiD 2.0 secondary scoring remains blocked"]}
    (OUTPUT / "evaluation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": decision, "eligible_labels": eligible, "primary_delta": observed, "primary_interval": bootstrap["paired_95_percentile_interval"] if bootstrap else None}, indent=2))


if __name__ == "__main__":
    main()
