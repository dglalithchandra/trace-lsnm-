"""Encoder-specific tensor adapters kept separate from medical preprocessing."""

from .dinov2 import prepare_dinov2_tensor

__all__ = ["prepare_dinov2_tensor"]
