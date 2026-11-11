"""Reusable 45-output RFMiD head for frozen DINOv2 embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


DECISION_TOLERANCE = 1e-6


class RFMiDLinearHead(nn.Module):
    def __init__(self, embedding_width: int = 384, disease_count: int = 45) -> None:
        super().__init__()
        self.classifier = nn.Linear(embedding_width, disease_count)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(embedding)


def load_rfmid_bundle(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> tuple[RFMiDLinearHead, dict[str, Any]]:
    bundle = torch.load(Path(path), map_location=device, weights_only=False)
    model = RFMiDLinearHead(
        embedding_width=int(bundle["embedding_width"]),
        disease_count=len(bundle["class_order"]),
    )
    model.load_state_dict(bundle["head_state_dict"], strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, bundle


def probabilities_from_embedding(
    model: RFMiDLinearHead,
    embedding: torch.Tensor,
    bundle: dict[str, Any],
) -> torch.Tensor:
    mean = torch.as_tensor(bundle["feature_mean"], device=embedding.device)
    scale = torch.as_tensor(bundle["feature_scale"], device=embedding.device)
    temperature = float(bundle["temperature"])
    standardized = (embedding - mean) / scale
    with torch.inference_mode():
        logits = model(standardized) / temperature
        return torch.sigmoid(logits)
