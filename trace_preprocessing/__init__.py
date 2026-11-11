"""TRACE camera-ID-agnostic preprocessing for colour fundus photographs."""

from .config import PreprocessingConfig
from .pipeline import TracePreprocessor
from .types import PreprocessingResult, ProcessingStatus, ReferenceProfile

__all__ = [
    "PreprocessingConfig",
    "PreprocessingResult",
    "ProcessingStatus",
    "ReferenceProfile",
    "TracePreprocessor",
]
