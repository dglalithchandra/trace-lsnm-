"""Versioned configuration for the TRACE preprocessing module.

The defaults are deliberately conservative development limits.  They are not
clinically validated thresholds and must not be promoted to final acceptance
criteria until the frozen disease model and validation bootstrap are available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class GeometryConfig:
    output_size: int = 448
    background_threshold: int = 12
    minimum_input_side: int = 256
    minimum_retinal_coverage: float = 0.20
    crop_margin_fraction: float = 0.025
    common_mask_radius_fraction: float = 0.485


@dataclass(frozen=True)
class PhotometricBounds:
    maximum_channel_gain_change: float = 0.12
    minimum_saturation_gain: float = 0.85
    maximum_saturation_gain: float = 1.15
    minimum_contrast_gain: float = 0.85
    maximum_contrast_gain: float = 1.15
    minimum_gamma: float = 0.85
    maximum_gamma: float = 1.18
    maximum_brightness_shift: float = 0.08
    maximum_illumination_strength: float = 0.35


@dataclass(frozen=True)
class QualityPolicy:
    # Hard image-change guards.  Blur/noise/glare measurements are recorded but
    # remain warnings until validation supplies defensible operating thresholds.
    maximum_mean_absolute_change: float = 0.12
    maximum_p95_absolute_change: float = 0.32
    minimum_structural_similarity: float = 0.82
    maximum_bright_clip_increase: float = 0.015
    maximum_dark_clip_increase: float = 0.020
    minimum_output_retinal_coverage: float = 0.70
    provisional_blur_warning_below: float = 18.0
    provisional_glare_warning_above: float = 0.03


@dataclass(frozen=True)
class PreprocessingConfig:
    version: str = "trace-preprocessing-dev-v0.2"
    modality: str = "color_fundus_photograph"
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    photometric_bounds: PhotometricBounds = field(default_factory=PhotometricBounds)
    quality: QualityPolicy = field(default_factory=QualityPolicy)
    preserve_original: bool = True
    apply_photometric_correction: bool = False
    policy_status: str = (
        "geometry_and_quality_active_photometric_harmonisation_not_selected"
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
