"""Leakage-controlled evaluation protocols for TRACE."""

from .fewshot import (
    FeatureSplit,
    FewShotProtocolConfig,
    build_episode_plan,
    classify_labels,
    load_feature_split,
    validate_episode_plan,
)
from .evaluation import (
    knn_class_margin_scores,
    l2_normalize,
    logistic_fewshot_scores,
    prototype_margin_scores,
)

__all__ = [
    "FeatureSplit",
    "FewShotProtocolConfig",
    "build_episode_plan",
    "classify_labels",
    "load_feature_split",
    "validate_episode_plan",
    "l2_normalize",
    "prototype_margin_scores",
    "knn_class_margin_scores",
    "logistic_fewshot_scores",
]
