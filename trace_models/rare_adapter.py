"""Small identity-initialised adapter for base-trained few-shot metric learning."""

from __future__ import annotations

import torch
from torch import nn


class RareAwareAdapter(nn.Module):
    def __init__(self, width: int = 384, hidden_width: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, width),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        adapted = embedding + self.network(embedding)
        return nn.functional.normalize(adapted, dim=-1)
