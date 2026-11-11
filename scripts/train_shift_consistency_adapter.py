"""Train/evaluate a residual paired-view adapter with frozen DINOv2/RFMiD head."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts.evaluate_photometric_candidates_with_model import (
    macro_metrics,
    paired_macro_auprc_delta_ci,
    representation_domain_metrics,
    threshold_metrics,
)
from trace_models import ShiftConsistencyAdapter, load_rfmid_bundle


SEED = 20261108
MAX_EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
INTERNAL_DEV_FRACTION = 0.15


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def load_pairs(split: str) -> dict[str, np.ndarray]:
    path = (
        WORKSPACE
        / "data"
        / "features"
        / f"rfmid_{split}_dinov2_camera_stress_v1.npz"
    )
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["split"][0]) != split:
            raise RuntimeError(f"Pair split mismatch: {split}")
        return {
            "record_ids": payload["record_ids"].astype(str),
            "clean": payload["clean_embeddings"].astype(np.float32),
            "stress": payload["stressed_embeddings"].astype(np.float32),
            "labels": payload["labels"].astype(np.float32),
            "class_order": payload["class_order"].astype(str),
            "fallback": payload["fallback_to_clean"].astype(np.uint8),
        }


def main() -> None:
    set_seed()
    train = load_pairs("train")
    validation = load_pairs("validation")
    if not np.array_equal(train["class_order"], validation["class_order"]):
        raise RuntimeError("Train/validation class order mismatch")
    class_order = train["class_order"].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head, bundle = load_rfmid_bundle(
        WORKSPACE
        / "models"
        / "rfmid_dinov2_linear_v1"
        / "rfmid_dinov2_linear_v1.pt",
        device=device,
    )
    if class_order != bundle["class_order"]:
        raise RuntimeError("Pair/model class order mismatch")
    head.eval().requires_grad_(False)
    mean_numpy = (
        torch.as_tensor(bundle["feature_mean"], dtype=torch.float32)
        .detach()
        .cpu()
        .numpy()
    )
    scale_numpy = (
        torch.as_tensor(bundle["feature_scale"], dtype=torch.float32)
        .detach()
        .cpu()
        .numpy()
    )

    def standardize(values: np.ndarray) -> np.ndarray:
        return ((values - mean_numpy) / scale_numpy).astype(np.float32)

    train_clean = standardize(train["clean"])
    train_stress = standardize(train["stress"])
    val_clean = standardize(validation["clean"])
    val_stress = standardize(validation["stress"])

    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(len(train_clean))
    internal_dev_size = int(round(len(permutation) * INTERNAL_DEV_FRACTION))
    internal_dev_indices = np.sort(permutation[:internal_dev_size])
    fitting_indices = np.sort(permutation[internal_dev_size:])
    if set(fitting_indices) & set(internal_dev_indices):
        raise RuntimeError("Internal adapter split overlap")

    fitting_dataset = TensorDataset(
        torch.from_numpy(train_clean[fitting_indices]),
        torch.from_numpy(train_stress[fitting_indices]),
        torch.from_numpy(train["labels"][fitting_indices]),
    )
    generator = torch.Generator().manual_seed(SEED)
    fitting_loader = DataLoader(
        fitting_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    internal_dev_tensors = (
        torch.from_numpy(train_clean[internal_dev_indices]).to(device),
        torch.from_numpy(train_stress[internal_dev_indices]).to(device),
        torch.from_numpy(train["labels"][internal_dev_indices]).to(device),
    )

    adapter = ShiftConsistencyAdapter(width=384, hidden_width=128).to(device)
    supported = np.asarray(
        [
            bundle["support_status"][name]
            != "unsupported_no_training_positives"
            for name in class_order
        ]
    )
    supported_tensor = torch.tensor(supported, dtype=torch.float32, device=device)
    positive_weights = torch.tensor(
        bundle["positive_weights"], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weights, reduction="none"
    )
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    def loss_terms(
        clean_input: torch.Tensor,
        stress_input: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        clean_output = adapter(clean_input)
        stress_output = adapter(stress_input)
        with torch.no_grad():
            baseline_clean_logits = head(clean_input)
        clean_logits = head(clean_output)
        stress_logits = head(stress_output)
        clean_task_matrix = criterion(clean_logits, targets) * supported_tensor
        stress_task_matrix = criterion(stress_logits, targets) * supported_tensor
        normalizer = targets.shape[0] * supported_tensor.sum()
        task = (
            clean_task_matrix.sum() + stress_task_matrix.sum()
        ) / (2.0 * normalizer)
        identity = functional.mse_loss(clean_output, clean_input)
        pair_alignment = functional.smooth_l1_loss(stress_output, clean_input)
        clean_distillation = functional.mse_loss(
            torch.sigmoid(clean_logits[:, supported]),
            torch.sigmoid(baseline_clean_logits[:, supported]),
        )
        pair_prediction = functional.mse_loss(
            torch.sigmoid(stress_logits[:, supported]),
            torch.sigmoid(clean_logits[:, supported].detach()),
        )
        total = (
            task
            + 2.0 * identity
            + 0.5 * pair_alignment
            + 2.0 * clean_distillation
            + pair_prediction
        )
        return {
            "total": total,
            "task": task,
            "identity": identity,
            "pair_alignment": pair_alignment,
            "clean_distillation": clean_distillation,
            "pair_prediction": pair_prediction,
        }

    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        adapter.train()
        running = {name: 0.0 for name in ("total", "task", "identity", "pair_alignment", "clean_distillation", "pair_prediction")}
        seen = 0
        for clean_batch, stress_batch, target_batch in fitting_loader:
            clean_batch = clean_batch.to(device)
            stress_batch = stress_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            terms = loss_terms(clean_batch, stress_batch, target_batch)
            terms["total"].backward()
            optimizer.step()
            batch_size = len(clean_batch)
            seen += batch_size
            for name in running:
                running[name] += float(terms[name].item()) * batch_size

        adapter.eval()
        with torch.inference_mode():
            dev_terms = loss_terms(*internal_dev_tensors)
        dev_loss = float(dev_terms["total"].item())
        row: dict[str, float | int] = {
            "epoch": epoch,
            **{f"train_{name}": value / seen for name, value in running.items()},
            **{f"internal_dev_{name}": float(value.item()) for name, value in dev_terms.items()},
        }
        history.append(row)
        if dev_loss < best_loss - 1e-6:
            best_loss = dev_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in adapter.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"adapter_training epoch={epoch} train={running['total']/seen:.6f} "
                f"internal_dev={dev_loss:.6f}",
                flush=True,
            )
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("No adapter state recorded")
    adapter.load_state_dict(best_state, strict=True)
    adapter.eval()

    val_clean_tensor = torch.from_numpy(val_clean).to(device)
    val_stress_tensor = torch.from_numpy(val_stress).to(device)
    with torch.inference_mode():
        adapted_clean = adapter(val_clean_tensor)
        adapted_stress = adapter(val_stress_tensor)
        baseline_clean_logits = head(val_clean_tensor) / float(bundle["temperature"])
        baseline_stress_logits = head(val_stress_tensor) / float(bundle["temperature"])
        adapted_clean_logits = head(adapted_clean) / float(bundle["temperature"])
        adapted_stress_logits = head(adapted_stress) / float(bundle["temperature"])
    probability_sets = {
        "baseline_clean": torch.sigmoid(baseline_clean_logits).cpu().numpy(),
        "baseline_stress": torch.sigmoid(baseline_stress_logits).cpu().numpy(),
        "adapter_clean": torch.sigmoid(adapted_clean_logits).cpu().numpy(),
        "adapter_stress": torch.sigmoid(adapted_stress_logits).cpu().numpy(),
    }
    adapted_clean_numpy = adapted_clean.cpu().numpy()
    adapted_stress_numpy = adapted_stress.cpu().numpy()
    labels = validation["labels"]
    positives = labels.sum(axis=0)
    robust = supported & (positives >= 5) & (positives < len(labels))
    metrics: dict[str, dict[str, object]] = {}
    for name, probabilities in probability_sets.items():
        auprc, auroc = macro_metrics(labels, probabilities, robust)
        metrics[name] = {
            "macro_auprc_robust_21_classes": auprc,
            "macro_auroc_robust_21_classes": auroc,
            "threshold_metrics": threshold_metrics(
                labels, probabilities, bundle["thresholds"]
            ),
        }
    metrics["adapter_clean_vs_baseline_clean"] = {
        "macro_auprc_delta": (
            metrics["adapter_clean"]["macro_auprc_robust_21_classes"]
            - metrics["baseline_clean"]["macro_auprc_robust_21_classes"]
        ),
        "paired_bootstrap": paired_macro_auprc_delta_ci(
            labels,
            probability_sets["baseline_clean"],
            probability_sets["adapter_clean"],
            robust,
        ),
    }
    metrics["adapter_stress_vs_baseline_stress"] = {
        "macro_auprc_delta": (
            metrics["adapter_stress"]["macro_auprc_robust_21_classes"]
            - metrics["baseline_stress"]["macro_auprc_robust_21_classes"]
        ),
        "paired_bootstrap": paired_macro_auprc_delta_ci(
            labels,
            probability_sets["baseline_stress"],
            probability_sets["adapter_stress"],
            robust,
        ),
    }
    metrics["pair_stability"] = {
        "baseline_mean_absolute_score_change": float(
            np.mean(
                np.abs(
                    probability_sets["baseline_stress"]
                    - probability_sets["baseline_clean"]
                )
            )
        ),
        "adapter_mean_absolute_score_change": float(
            np.mean(
                np.abs(
                    probability_sets["adapter_stress"]
                    - probability_sets["adapter_clean"]
                )
            )
        ),
    }

    with np.load(
        WORKSPACE / "outputs" / "rfmid_baseline" / "baseline_predictions.npz",
        allow_pickle=False,
    ) as payload:
        domains = payload["proxy_domains"].astype(str)
    domain_metrics = {
        "baseline_clean": representation_domain_metrics(
            val_clean, domains, np.zeros(384), np.ones(384)
        ),
        "adapter_clean": representation_domain_metrics(
            adapted_clean_numpy, domains, np.zeros(384), np.ones(384)
        ),
    }
    normal = labels.sum(axis=1) == 0
    domain_metrics["normal_only_baseline_clean"] = representation_domain_metrics(
        val_clean[normal], domains[normal], np.zeros(384), np.ones(384)
    )
    domain_metrics["normal_only_adapter_clean"] = representation_domain_metrics(
        adapted_clean_numpy[normal],
        domains[normal],
        np.zeros(384),
        np.ones(384),
    )

    clean_delta = float(
        metrics["adapter_clean_vs_baseline_clean"]["macro_auprc_delta"]
    )
    stress_delta = float(
        metrics["adapter_stress_vs_baseline_stress"]["macro_auprc_delta"]
    )
    baseline_clean_sensitivity = float(
        metrics["baseline_clean"]["threshold_metrics"]["micro_sensitivity"]
    )
    adapter_clean_sensitivity = float(
        metrics["adapter_clean"]["threshold_metrics"]["micro_sensitivity"]
    )
    stability_improved = (
        metrics["pair_stability"]["adapter_mean_absolute_score_change"]
        < metrics["pair_stability"]["baseline_mean_absolute_score_change"]
    )
    passes = (
        clean_delta >= 0.0
        and adapter_clean_sensitivity >= baseline_clean_sensitivity
        and stress_delta > 0.0
        and stability_improved
    )

    output_dir = WORKSPACE / "models" / "rfmid_shift_adapter_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "rfmid_shift_adapter_v1.pt"
    torch.save(
        {
            "architecture": "residual_shift_consistency_adapter_384_128_384",
            "state_dict": {
                name: value.detach().cpu()
                for name, value in adapter.state_dict().items()
            },
            "input_space": "training-standardized_dinov2_vits14_embedding",
            "training_seed": SEED,
            "fitting_record_ids": train["record_ids"][fitting_indices].tolist(),
            "internal_dev_record_ids": train["record_ids"][internal_dev_indices].tolist(),
            "best_epoch": best_epoch,
            "best_internal_dev_loss": best_loss,
            "development_decision": "provisional_pass" if passes else "not_selected",
            "research_only": True,
        },
        checkpoint,
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    np.savez_compressed(
        output_dir / "validation_adapter_outputs.npz",
        record_ids=validation["record_ids"],
        labels=labels.astype(np.uint8),
        class_order=np.asarray(class_order, dtype=np.str_),
        baseline_clean_probabilities=probability_sets["baseline_clean"].astype(np.float32),
        baseline_stress_probabilities=probability_sets["baseline_stress"].astype(np.float32),
        adapter_clean_probabilities=probability_sets["adapter_clean"].astype(np.float32),
        adapter_stress_probabilities=probability_sets["adapter_stress"].astype(np.float32),
        adapter_clean_embeddings=adapted_clean_numpy.astype(np.float32),
        adapter_stress_embeddings=adapted_stress_numpy.astype(np.float32),
        proxy_domains=domains.astype(np.str_),
        split=np.asarray(["validation"], dtype=np.str_),
    )
    result = {
        "status": "PASS",
        "adapter_decision": "provisional_pass" if passes else "not_selected",
        "decision_rule": {
            "clean_macro_auprc_delta_at_least": 0.0,
            "clean_sensitivity_must_not_decrease": True,
            "stress_macro_auprc_delta_greater_than": 0.0,
            "paired_score_stability_must_improve": True,
        },
        "train_records": len(train["record_ids"]),
        "adapter_fitting_records": len(fitting_indices),
        "adapter_internal_dev_records": len(internal_dev_indices),
        "validation_records": len(validation["record_ids"]),
        "best_epoch": best_epoch,
        "best_internal_dev_loss": best_loss,
        "metrics": metrics,
        "domain_metrics": domain_metrics,
        "checkpoint": str(checkpoint.relative_to(WORKSPACE)).replace("\\", "/"),
        "synthetic_shift_disclaimer": (
            "results establish bounded synthetic-shift robustness only, not unseen-camera generalization"
        ),
        "validation_reuse_limit": (
            "RFMiD validation was previously used for the disease head; this remains development evidence"
        ),
        "test_records_accessed": 0,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
