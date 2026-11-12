"""Reproducible development-only benchmarking utilities for TRACE."""

from .exchange import ExchangeValidationError, load_exchange, validate_exchange
from .episodes import (
    CandidateRecord,
    Episode,
    EpisodeConfigurationError,
    EpisodePlan,
    UnsupportedSetting,
    build_episode_plan,
    episode_plan_hash,
)
from .metrics import evaluate_binary, macro_metrics
from .paired import PairedObservation, paired_bootstrap_difference

__all__ = [
    "CandidateRecord",
    "Episode",
    "EpisodeConfigurationError",
    "EpisodePlan",
    "ExchangeValidationError",
    "PairedObservation",
    "UnsupportedSetting",
    "build_episode_plan",
    "episode_plan_hash",
    "evaluate_binary",
    "load_exchange",
    "macro_metrics",
    "paired_bootstrap_difference",
    "validate_exchange",
]
