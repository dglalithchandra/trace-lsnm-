"""Train, validate and globally calibrate a frozen-DINOv2 RFMiD linear head.

The final test split is not loaded. Outputs with no positive training examples
remain present in the 45-column interface but are explicitly unsupported.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_models.rfmid_dinov2 import RFMiDLinearHead


SEED = 20261108
EMBEDDING_WIDTH = 384
EPOCHS = 400
PATIENCE = 50
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_features(path: Path, expected_split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as payload:
        split = str(payload["split"][0])
        if split != expected_split:
            raise RuntimeError(f"Feature split mismatch: {split} != {expected_split}")
        embeddings = payload["embeddings"].astype(np.float32)
        labels = payload["labels"].astype(np.float32)
        record_ids = payload["record_ids"].astype(str)
        class_order = payload["class_order"].astype(str).tolist()
    if embeddings.shape[1] != EMBEDDING_WIDTH:
        raise RuntimeError(f"Unexpected embedding width: {embeddings.shape}")
    if labels.shape != (len(embeddings), len(class_order)):
        raise RuntimeError("Feature/label shape mismatch")
    if len(set(record_ids.tolist())) != len(record_ids):
        raise RuntimeError("Duplicate record IDs in feature file")
    return embeddings, labels, record_ids, class_order


def macro_average_precision(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_mask: np.ndarray,
) -> float:
    values = [
        average_precision_score(labels[:, index], probabilities[:, index])
        for index in np.flatnonzero(class_mask)
    ]
    return float(np.mean(values))


def stable_binary_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    labels_flat = labels.ravel()
    probability_flat = probabilities.ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels_flat)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        included = (probability_flat >= lower) & (
            probability_flat < upper if upper < 1.0 else probability_flat <= upper
        )
        count = int(included.sum())
        if count:
            value += count / total * abs(
                float(labels_flat[included].mean())
                - float(probability_flat[included].mean())
            )
    return float(value)


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5, 0.0
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    index = int(np.nanargmax(f1))
    return float(thresholds[index]), float(f1[index])


def main() -> None:
    set_seed(SEED)
    feature_dir = WORKSPACE / "data" / "features"
    train_path = feature_dir / "rfmid_train_dinov2_vits14_geometry_v1.npz"
    validation_path = feature_dir / "rfmid_validation_dinov2_vits14_geometry_v1.npz"
    train_x, train_y, train_ids, class_order = load_features(train_path, "train")
    val_x, val_y, val_ids, validation_order = load_features(validation_path, "validation")
    if class_order != validation_order or len(class_order) != 45:
        raise RuntimeError("Train/validation class order mismatch")
    if set(train_ids) & set(val_ids):
        raise RuntimeError("Record leakage between train and validation feature files")

    feature_mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = np.maximum(feature_scale, 1e-6)
    train_x = (train_x - feature_mean) / feature_scale
    val_x = (val_x - feature_mean) / feature_scale

    train_positive = train_y.sum(axis=0).astype(np.int64)
    val_positive = val_y.sum(axis=0).astype(np.int64)
    supported = (train_positive > 0) & (train_positive < len(train_y))
    val_evaluable = supported & (val_positive > 0) & (val_positive < len(val_y))
    robust_val = val_evaluable & (val_positive >= 5)
    unsupported_names = [
        class_order[index] for index in np.flatnonzero(~supported)
    ]

    negatives = len(train_y) - train_positive
    positive_weight = np.ones(len(class_order), dtype=np.float32)
    positive_weight[supported] = np.sqrt(
        negatives[supported] / np.maximum(train_positive[supported], 1)
    )
    positive_weight = np.clip(positive_weight, 1.0, 20.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RFMiDLinearHead(EMBEDDING_WIDTH, len(class_order)).to(device)
    supported_tensor = torch.tensor(supported, dtype=torch.float32, device=device)
    pos_weight_tensor = torch.tensor(positive_weight, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_tensor = torch.from_numpy(val_x).to(device)

    best_metric = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        examples = 0
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss_matrix = criterion(logits, targets) * supported_tensor
            loss = loss_matrix.sum() / (targets.shape[0] * supported_tensor.sum())
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * targets.shape[0]
            examples += targets.shape[0]

        model.eval()
        with torch.inference_mode():
            val_logits = model(val_tensor).cpu().numpy()
        val_probabilities = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -40, 40)))
        validation_map = macro_average_precision(val_y, val_probabilities, robust_val)
        history.append(
            {
                "epoch": epoch,
                "training_loss": running_loss / examples,
                "validation_macro_auprc_robust_classes": validation_map,
            }
        )
        if validation_map > best_metric + 1e-6:
            best_metric = validation_map
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch % 25 == 0 or epoch == 1:
            print(
                f"head_training epoch={epoch} loss={running_loss/examples:.6f} "
                f"val_macro_auprc={validation_map:.6f}",
                flush=True,
            )
        if stale_epochs >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("No best RFMiD head state was recorded")
    model.load_state_dict(best_state, strict=True)

    # Outputs with no positive training examples are prior-only placeholders,
    # not learned detectors. Zero their weights and set a smoothed prevalence.
    with torch.no_grad():
        for index in np.flatnonzero(~supported):
            prevalence = (float(train_positive[index]) + 0.5) / (len(train_y) + 1.0)
            prior_logit = math.log(prevalence / (1.0 - prevalence))
            model.classifier.weight[index].zero_()
            model.classifier.bias[index].fill_(prior_logit)

    model.eval()
    with torch.inference_mode():
        val_logits = model(val_tensor).cpu().numpy()
    supported_logits = val_logits[:, supported]
    supported_labels = val_y[:, supported]
    temperature_result = minimize_scalar(
        lambda value: stable_binary_nll(supported_logits / value, supported_labels),
        bounds=(0.25, 5.0),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not temperature_result.success:
        raise RuntimeError(f"Temperature fitting failed: {temperature_result.message}")
    temperature = float(temperature_result.x)
    raw_probabilities = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -40, 40)))
    calibrated_probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(val_logits / temperature, -40, 40))
    )

    per_class: list[dict[str, object]] = []
    thresholds: list[float | None] = []
    support_status: dict[str, str] = {}
    for index, name in enumerate(class_order):
        train_count = int(train_positive[index])
        val_count = int(val_positive[index])
        row: dict[str, object] = {
            "class_index": index,
            "label": name,
            "train_positive": train_count,
            "validation_positive": val_count,
            "validation_negative": int(len(val_y) - val_count),
            "positive_weight": float(positive_weight[index]),
        }
        threshold: float | None = None
        if not supported[index]:
            status = "unsupported_no_training_positives"
        elif val_count == 0:
            status = "trained_but_unevaluable_no_validation_positives"
        else:
            status = (
                "validation_evaluable"
                if val_count >= 5
                else "provisional_low_validation_support"
            )
            row["auroc"] = float(
                roc_auc_score(val_y[:, index], calibrated_probabilities[:, index])
            )
            row["auprc"] = float(
                average_precision_score(
                    val_y[:, index], calibrated_probabilities[:, index]
                )
            )
            row["brier"] = float(
                brier_score_loss(
                    val_y[:, index], calibrated_probabilities[:, index]
                )
            )
            if val_count >= 5:
                threshold, validation_f1 = best_f1_threshold(
                    val_y[:, index], calibrated_probabilities[:, index]
                )
                row["validation_selected_threshold"] = threshold
                row["validation_f1_at_selected_threshold"] = validation_f1
        row["support_status"] = status
        thresholds.append(threshold)
        support_status[name] = status
        per_class.append(row)

    supported_raw = raw_probabilities[:, supported]
    supported_calibrated = calibrated_probabilities[:, supported]
    aggregate = {
        "macro_auroc_evaluable_classes": float(
            np.mean(
                [
                    roc_auc_score(val_y[:, index], calibrated_probabilities[:, index])
                    for index in np.flatnonzero(val_evaluable)
                ]
            )
        ),
        "macro_auprc_evaluable_classes": macro_average_precision(
            val_y, calibrated_probabilities, val_evaluable
        ),
        "macro_auprc_robust_classes": macro_average_precision(
            val_y, calibrated_probabilities, robust_val
        ),
        "micro_auroc_supported_outputs": float(
            roc_auc_score(supported_labels.ravel(), supported_calibrated.ravel())
        ),
        "micro_auprc_supported_outputs": float(
            average_precision_score(supported_labels.ravel(), supported_calibrated.ravel())
        ),
        "raw_nll_supported_outputs": stable_binary_nll(
            supported_logits, supported_labels
        ),
        "calibrated_nll_supported_outputs": stable_binary_nll(
            supported_logits / temperature, supported_labels
        ),
        "raw_brier_supported_outputs": float(
            np.mean((supported_raw - supported_labels) ** 2)
        ),
        "calibrated_brier_supported_outputs": float(
            np.mean((supported_calibrated - supported_labels) ** 2)
        ),
        "raw_ece_supported_outputs": expected_calibration_error(
            supported_labels, supported_raw
        ),
        "calibrated_ece_supported_outputs": expected_calibration_error(
            supported_labels, supported_calibrated
        ),
    }

    output_dir = WORKSPACE / "models" / "rfmid_dinov2_linear_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "rfmid_dinov2_linear_v1.pt"
    bundle = {
        "architecture": "frozen_dinov2_vits14_plus_linear_multilabel_head",
        "embedding_width": EMBEDDING_WIDTH,
        "class_order": class_order,
        "head_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "feature_mean": torch.from_numpy(feature_mean),
        "feature_scale": torch.from_numpy(feature_scale),
        "temperature": temperature,
        "thresholds": thresholds,
        "support_status": support_status,
        "training_positive_counts": train_positive.tolist(),
        "validation_positive_counts": val_positive.tolist(),
        "positive_weights": positive_weight.tolist(),
        "preprocessing": "geometry_only_trace-preprocessing-dev-v0.1",
        "input_size": 448,
        "seed": SEED,
        "research_only": True,
    }
    torch.save(bundle, bundle_path)
    checkpoint_sha256 = sha256_file(bundle_path)

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(per_class).to_csv(
        output_dir / "per_class_validation_metrics.csv", index=False
    )
    summary = {
        "status": "PASS",
        "research_only": True,
        "test_records_accessed": 0,
        "train_records": len(train_y),
        "validation_records": len(val_y),
        "outputs": len(class_order),
        "learnable_outputs": int(supported.sum()),
        "unsupported_outputs": unsupported_names,
        "validation_evaluable_outputs": int(val_evaluable.sum()),
        "robust_validation_outputs_at_least_5_positives": int(robust_val.sum()),
        "best_epoch": best_epoch,
        "early_stopping_metric": "macro AUPRC over supported classes with at least 5 validation positives",
        "best_early_stopping_metric": best_metric,
        "temperature": temperature,
        "aggregate_validation_metrics": aggregate,
        "checkpoint": str(bundle_path.relative_to(WORKSPACE)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_sha256,
        "feature_files": {
            "train": {
                "path": str(train_path.relative_to(WORKSPACE)).replace("\\", "/"),
                "sha256": sha256_file(train_path),
            },
            "validation": {
                "path": str(validation_path.relative_to(WORKSPACE)).replace("\\", "/"),
                "sha256": sha256_file(validation_path),
            },
        },
        "limitations": [
            "HR and ODPM have no positive training examples and are not learned detectors.",
            "Thresholds are selected only for classes with at least five validation positives.",
            "Validation metrics for very rare classes have high statistical uncertainty.",
            "No final-test performance has been measured.",
            "The model is not approved for clinical use.",
        ],
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
