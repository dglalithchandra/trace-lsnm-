"""Train/select a base-only episodic TRACE rare-disease metric adapter."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_models import RareAwareAdapter
from trace_protocols import l2_normalize


SEED = 20261111
STEPS = 1_200
EVAL_EVERY = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
QUERY_POSITIVES = 12
QUERY_NEGATIVES = 24
DEV_FRACTION = 0.15
DEV_EPISODES_PER_LABEL = 20
OUTPUT = WORKSPACE / "models" / "rare_aware_adapter_v1"


@dataclass(frozen=True)
class Candidate:
    name: str
    identity_weight: float
    consistency_weight: float


CANDIDATES = (
    Candidate("metric_only", identity_weight=0.10, consistency_weight=0.0),
    Candidate("metric_consistency", identity_weight=0.10, consistency_weight=0.50),
    Candidate("conservative_consistency", identity_weight=0.50, consistency_weight=0.25),
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def find_split(
    eligible_indices: np.ndarray, labels: np.ndarray, base_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    for attempt in range(10_000):
        rng = np.random.default_rng(SEED + attempt)
        shuffled = rng.permutation(eligible_indices)
        dev_size = int(round(len(shuffled) * DEV_FRACTION))
        dev = np.sort(shuffled[:dev_size])
        fit = np.sort(shuffled[dev_size:])
        fit_counts = labels[fit][:, base_indices].sum(axis=0)
        dev_counts = labels[dev][:, base_indices].sum(axis=0)
        if np.all(fit_counts >= 20) and np.all(dev_counts >= 5):
            return fit, dev
    raise RuntimeError("unable_to_construct_base_covered_internal_split")


def build_dev_episodes(
    fit: np.ndarray,
    dev: np.ndarray,
    labels: np.ndarray,
    base_indices: np.ndarray,
) -> list[dict[str, object]]:
    episodes = []
    for class_index in base_indices:
        fit_positive = fit[labels[fit, class_index] == 1]
        fit_negative = fit[labels[fit, class_index] == 0]
        for replicate in range(DEV_EPISODES_PER_LABEL):
            rng = np.random.default_rng(
                stable_seed(f"dev|{int(class_index)}|{replicate}")
            )
            episodes.append(
                {
                    "class_index": int(class_index),
                    "positive_support": np.sort(
                        rng.choice(fit_positive, size=5, replace=False)
                    ),
                    "negative_support": np.sort(
                        rng.choice(fit_negative, size=5, replace=False)
                    ),
                    "query": dev,
                }
            )
    return episodes


def adapt_numpy(
    model: RareAwareAdapter, values: np.ndarray, device: torch.device
) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), 512):
            tensor = torch.from_numpy(values[start : start + 512]).to(device)
            outputs.append(model(tensor).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def dev_macro_auprc(
    embeddings: np.ndarray,
    labels: np.ndarray,
    episodes: list[dict[str, object]],
) -> float:
    values = []
    for episode in episodes:
        positive = embeddings[episode["positive_support"]].mean(axis=0)
        negative = embeddings[episode["negative_support"]].mean(axis=0)
        positive /= max(float(np.linalg.norm(positive)), 1e-12)
        negative /= max(float(np.linalg.norm(negative)), 1e-12)
        query_indices = episode["query"]
        scores = embeddings[query_indices] @ positive - embeddings[query_indices] @ negative
        query_labels = labels[query_indices, int(episode["class_index"])]
        values.append(float(average_precision_score(query_labels, scores)))
    return float(np.mean(values))


def main() -> None:
    seed_all(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    feature_path = WORKSPACE / "data/features/rfmid_train_dinov2_vits14_geometry_v1.npz"
    stress_path = WORKSPACE / "data/features/rfmid_train_dinov2_camera_stress_v1.npz"
    with np.load(feature_path, allow_pickle=False) as payload:
        record_ids = payload["record_ids"].astype(str)
        clean = l2_normalize(payload["embeddings"].astype(np.float32))
        labels = payload["labels"].astype(np.uint8)
        class_order = payload["class_order"].astype(str)
    with np.load(stress_path, allow_pickle=False) as payload:
        if payload["record_ids"].astype(str).tolist() != record_ids.tolist():
            raise RuntimeError("stress_feature_record_order_mismatch")
        stressed = l2_normalize(payload["stressed_embeddings"].astype(np.float32))

    protocol_dir = WORKSPACE / "data/protocols/trace_fewshot_v1"
    pool = pd.read_csv(protocol_dir / "base_adapter_pool.csv")
    label_table = pd.read_csv(protocol_dir / "label_eligibility.csv")
    if pool["record_id"].tolist() != record_ids.tolist():
        raise RuntimeError("adapter_pool_record_order_mismatch")
    eligible = pool["eligible_for_base_adapter"].to_numpy(dtype=bool)
    if np.any(pool.loc[eligible, "has_novel_positive"].to_numpy(dtype=bool)):
        raise RuntimeError("novel_positive_leakage_in_adapter_pool")
    eligible_indices = np.flatnonzero(eligible)
    base_indices = label_table.loc[
        label_table["role"].eq("base"), "class_index"
    ].to_numpy(dtype=np.int64)
    novel_indices = label_table.loc[
        label_table["role"].eq("novel"), "class_index"
    ].to_numpy(dtype=np.int64)
    fit, dev = find_split(eligible_indices, labels, base_indices)
    if np.any(labels[fit][:, novel_indices]) or np.any(labels[dev][:, novel_indices]):
        raise RuntimeError("novel_label_positive_entered_internal_split")
    dev_episodes = build_dev_episodes(fit, dev, labels, base_indices)
    baseline_dev = dev_macro_auprc(clean, labels, dev_episodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_tensor = torch.from_numpy(clean).to(device)
    stress_tensor = torch.from_numpy(stressed).to(device)
    label_tensor = torch.from_numpy(labels).to(device)

    candidate_summaries = []
    candidate_states: dict[str, dict[str, torch.Tensor]] = {}
    all_history = []
    for candidate_index, candidate in enumerate(CANDIDATES):
        seed_all(SEED)
        model = RareAwareAdapter(width=clean.shape[1], hidden_width=128).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        rng = np.random.default_rng(SEED)
        best_dev = -np.inf
        best_step = 0
        best_state = None
        running = {"task": 0.0, "identity": 0.0, "consistency": 0.0, "total": 0.0}
        for step in range(1, STEPS + 1):
            class_index = int(rng.choice(base_indices))
            k = int(rng.choice((1, 3, 5)))
            positive_pool = fit[labels[fit, class_index] == 1]
            negative_pool = fit[labels[fit, class_index] == 0]
            positive_choice = rng.choice(
                positive_pool,
                size=k + min(QUERY_POSITIVES, len(positive_pool) - k),
                replace=False,
            )
            negative_choice = rng.choice(
                negative_pool, size=k + QUERY_NEGATIVES, replace=False
            )
            support_positive = positive_choice[:k]
            query_positive = positive_choice[k:]
            support_negative = negative_choice[:k]
            query_negative = negative_choice[k:]
            query = np.concatenate([query_positive, query_negative])
            support = np.concatenate([support_positive, support_negative])
            used = np.unique(np.concatenate([support, query]))

            model.train()
            optimizer.zero_grad(set_to_none=True)
            adapted_support = model(clean_tensor[support])
            adapted_query = model(clean_tensor[query])
            positive_prototype = functional.normalize(
                adapted_support[:k].mean(dim=0), dim=0
            )
            negative_prototype = functional.normalize(
                adapted_support[k:].mean(dim=0), dim=0
            )
            scores = 10.0 * (
                adapted_query @ positive_prototype
                - adapted_query @ negative_prototype
            )
            targets = torch.cat(
                [
                    torch.ones(len(query_positive), device=device),
                    torch.zeros(len(query_negative), device=device),
                ]
            )
            task_loss = functional.binary_cross_entropy_with_logits(scores, targets)
            adapted_used = model(clean_tensor[used])
            identity_loss = torch.mean(
                1.0 - torch.sum(adapted_used * clean_tensor[used], dim=1)
            )
            adapted_stress = model(stress_tensor[used])
            consistency_loss = torch.mean(
                1.0 - torch.sum(adapted_used * adapted_stress, dim=1)
            )
            total = (
                task_loss
                + candidate.identity_weight * identity_loss
                + candidate.consistency_weight * consistency_loss
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for name, value in (
                ("task", task_loss),
                ("identity", identity_loss),
                ("consistency", consistency_loss),
                ("total", total),
            ):
                running[name] += float(value.item())

            if step % EVAL_EVERY == 0:
                adapted = adapt_numpy(model, clean, device)
                score = dev_macro_auprc(adapted, labels, dev_episodes)
                row = {
                    "candidate": candidate.name,
                    "step": step,
                    "dev_macro_auprc": score,
                    "baseline_dev_macro_auprc": baseline_dev,
                    **{
                        f"mean_{name}_loss": value / EVAL_EVERY
                        for name, value in running.items()
                    },
                }
                all_history.append(row)
                print(
                    f"rare_adapter candidate={candidate.name} step={step}/{STEPS} "
                    f"dev_auprc={score:.6f} baseline={baseline_dev:.6f}",
                    flush=True,
                )
                if score > best_dev + 1e-8:
                    best_dev = score
                    best_step = step
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                running = {name: 0.0 for name in running}
        if best_state is None:
            raise RuntimeError(f"no_best_state:{candidate.name}")
        candidate_states[candidate.name] = best_state
        candidate_summaries.append(
            {
                "candidate": candidate.name,
                "identity_weight": candidate.identity_weight,
                "consistency_weight": candidate.consistency_weight,
                "best_step": best_step,
                "best_internal_dev_macro_auprc": best_dev,
                "delta_vs_frozen_dinov2": best_dev - baseline_dev,
            }
        )

    ordered = sorted(
        candidate_summaries,
        key=lambda item: (
            float(item["best_internal_dev_macro_auprc"]),
            -float(item["consistency_weight"]),
        ),
        reverse=True,
    )
    selected = ordered[0]
    selection_pass = float(selected["best_internal_dev_macro_auprc"]) >= baseline_dev
    decision = "frozen_for_novel_evaluation" if selection_pass else "not_selected"
    checkpoint = OUTPUT / "rare_aware_adapter_v1.pt"
    torch.save(
        {
            "architecture": "rare_aware_residual_adapter_384_128_384",
            "state_dict": candidate_states[str(selected["candidate"])],
            "selected_candidate": selected,
            "input_contract": "l2_normalized_frozen_dinov2_vits14_embedding",
            "training_seed": SEED,
            "fit_record_ids": record_ids[fit].tolist(),
            "internal_dev_record_ids": record_ids[dev].tolist(),
            "base_class_indices": base_indices.tolist(),
            "novel_class_indices_excluded": novel_indices.tolist(),
            "development_decision": decision,
            "research_only": True,
        },
        checkpoint,
    )
    pd.DataFrame(all_history).to_csv(OUTPUT / "training_history.csv", index=False)
    pd.DataFrame(candidate_summaries).to_csv(
        OUTPUT / "candidate_selection.csv", index=False
    )
    result = {
        "status": "PASS",
        "decision": decision,
        "selection_rule": (
            "highest train-only internal-dev base-label macro AUPRC; must not be "
            "below frozen DINOv2 baseline"
        ),
        "baseline_internal_dev_macro_auprc": baseline_dev,
        "selected": selected,
        "candidates": candidate_summaries,
        "eligible_records": len(eligible_indices),
        "fit_records": len(fit),
        "internal_dev_records": len(dev),
        "base_labels": [class_order[index] for index in base_indices],
        "novel_labels_excluded": [class_order[index] for index in novel_indices],
        "novel_positive_records_used": 0,
        "validation_records_accessed": 0,
        "protected_test_records_accessed": 0,
        "checkpoint": str(checkpoint.relative_to(WORKSPACE)).replace("\\", "/"),
    }
    (OUTPUT / "training_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
