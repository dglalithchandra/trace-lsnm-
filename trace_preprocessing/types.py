"""Data contracts shared by TRACE preprocessing components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class ProcessingStatus(StrEnum):
    ACCEPT = "accept"
    WARNING = "accept_with_warning"
    REJECT = "reject"


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class FundusDetection:
    mask: np.ndarray
    center_x: float
    center_y: float
    radius: float
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    retinal_coverage: float
    confidence: float


@dataclass(frozen=True)
class GeometryTransform:
    source_width: int
    source_height: int
    crop_x: int
    crop_y: int
    crop_size: int
    pad_left: int
    pad_top: int
    output_size: int

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        scale = self.output_size / self.crop_size
        output_x = (x - self.crop_x + self.pad_left) * scale
        output_y = (y - self.crop_y + self.pad_top) * scale
        return output_x, output_y

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhotometricStats:
    red_mean: float
    green_mean: float
    blue_mean: float
    red_green_ratio: float
    blue_green_ratio: float
    saturation_median: float
    brightness_median: float
    contrast_p95_p05: float
    gamma_to_midgray: float
    illumination_nonuniformity: float
    dark_clip_fraction: float
    bright_clip_fraction: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class QualityMetrics:
    retinal_coverage: float
    blur_laplacian_variance: float
    noise_mad: float
    glare_fraction: float
    compression_blockiness: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ReferenceTarget:
    median: float
    lower: float
    upper: float


@dataclass(frozen=True)
class ReferenceProfile:
    profile_id: str
    source_split: str
    sample_count: int
    targets: dict[str, ReferenceTarget]
    status: str = "development_reference_not_final"

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "source_split": self.source_split,
            "sample_count": self.sample_count,
            "status": self.status,
            "targets": {name: asdict(target) for name, target in self.targets.items()},
        }


@dataclass(frozen=True)
class CorrectionParameters:
    red_gain: float = 1.0
    green_gain: float = 1.0
    blue_gain: float = 1.0
    saturation_gain: float = 1.0
    brightness_shift: float = 0.0
    contrast_gain: float = 1.0
    gamma: float = 1.0
    illumination_strength: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonMetrics:
    mean_absolute_change: float
    p95_absolute_change: float
    structural_similarity: float
    dark_clip_increase: float
    bright_clip_increase: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class AnnotationPreservation:
    annotation_name: str
    source_foreground_pixels: int
    transformed_foreground_pixels: int
    source_region_clipped_fraction: float
    mean_absolute_change_in_region: float
    p95_absolute_change_in_region: float
    bright_clip_fraction_after: float
    dark_clip_fraction_after: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessingResult:
    image_id: str
    status: ProcessingStatus
    reasons: list[str]
    warnings: list[str]
    original_image: np.ndarray | None
    geometry_only_image: np.ndarray | None
    processed_image: np.ndarray | None
    retinal_mask: np.ndarray | None
    geometry_transform: GeometryTransform | None
    stats_before: PhotometricStats | None
    stats_after: PhotometricStats | None
    quality_before: QualityMetrics | None
    quality_after: QualityMetrics | None
    corrections: CorrectionParameters | None
    comparison: ComparisonMetrics | None
    annotation_preservation: dict[str, AnnotationPreservation]
    preprocessing_version: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def audit_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "status": self.status.value,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "geometry_transform": (
                self.geometry_transform.to_dict() if self.geometry_transform else None
            ),
            "stats_before": self.stats_before.to_dict() if self.stats_before else None,
            "stats_after": self.stats_after.to_dict() if self.stats_after else None,
            "quality_before": self.quality_before.to_dict() if self.quality_before else None,
            "quality_after": self.quality_after.to_dict() if self.quality_after else None,
            "corrections": self.corrections.to_dict() if self.corrections else None,
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "annotation_preservation": {
                name: result.to_dict()
                for name, result in self.annotation_preservation.items()
            },
            "preprocessing_version": self.preprocessing_version,
            "metadata": self.metadata,
        }
