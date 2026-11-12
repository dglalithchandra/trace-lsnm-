"""TRACE disease-model components."""

from .rfmid_dinov2 import (
    RFMiDLinearHead,
    load_rfmid_bundle,
    probabilities_from_embedding,
)
from .rfmid_scorer import TraceRFMiDScorer
from .fewshot import TraceFewShotHead
from .retrieval import TraceRetrievalIndex
from .rare_adapter import RareAwareAdapter
from .shift_adapter import ShiftConsistencyAdapter
from .source_balance import TraceA1Encoder

__all__ = [
    "RFMiDLinearHead",
    "RareAwareAdapter",
    "ShiftConsistencyAdapter",
    "TraceRFMiDScorer",
    "TraceA1Encoder",
    "TraceFewShotHead",
    "TraceRetrievalIndex",
    "load_rfmid_bundle",
    "probabilities_from_embedding",
]
