"""Residual post-encoder adapter for paired camera-shift consistency."""

from __future__ import annotations

import torch
from torch import nn


class ShiftConsistencyAdapter(nn.Module):
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

    def forward(self, standardized_embedding: torch.Tensor) -> torch.Tensor:
        return standardized_embedding + self.network(standardized_embedding)
