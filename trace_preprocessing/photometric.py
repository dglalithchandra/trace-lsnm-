"""Bounded, image-adaptive photometric harmonisation."""

from __future__ import annotations

import cv2
import numpy as np

from .config import PhotometricBounds
from .types import CorrectionParameters, PhotometricStats, ReferenceProfile


def _bounded_ratio(target: float, observed: float, maximum_change: float) -> float:
    ratio = target / max(observed, 1e-6)
    return float(np.clip(ratio, 1.0 - maximum_change, 1.0 + maximum_change))


def _target_if_outside(value: float, lower: float, median: float, upper: float) -> float:
    return value if lower <= value <= upper else median


def derive_corrections(
    stats: PhotometricStats,
    reference: ReferenceProfile,
    bounds: PhotometricBounds,
) -> CorrectionParameters:
    targets = reference.targets
    required = {
        "red_green_ratio",
        "blue_green_ratio",
        "saturation_median",
        "brightness_median",
        "contrast_p95_p05",
        "gamma_to_midgray",
        "illumination_nonuniformity",
    }
    missing = required - targets.keys()
    if missing:
        raise ValueError(f"reference_profile_missing_targets:{sorted(missing)}")

    red_target = _target_if_outside(
        stats.red_green_ratio,
        targets["red_green_ratio"].lower,
        targets["red_green_ratio"].median,
        targets["red_green_ratio"].upper,
    )
    blue_target = _target_if_outside(
        stats.blue_green_ratio,
        targets["blue_green_ratio"].lower,
        targets["blue_green_ratio"].median,
        targets["blue_green_ratio"].upper,
    )
    saturation_target = _target_if_outside(
        stats.saturation_median,
        targets["saturation_median"].lower,
        targets["saturation_median"].median,
        targets["saturation_median"].upper,
    )
    brightness_target = _target_if_outside(
        stats.brightness_median,
        targets["brightness_median"].lower,
        targets["brightness_median"].median,
        targets["brightness_median"].upper,
    )
    contrast_target = _target_if_outside(
        stats.contrast_p95_p05,
        targets["contrast_p95_p05"].lower,
        targets["contrast_p95_p05"].median,
        targets["contrast_p95_p05"].upper,
    )
    gamma_target = _target_if_outside(
        stats.gamma_to_midgray,
        targets["gamma_to_midgray"].lower,
        targets["gamma_to_midgray"].median,
        targets["gamma_to_midgray"].upper,
    )
    illumination_excess = max(
        0.0,
        stats.illumination_nonuniformity - targets["illumination_nonuniformity"].upper,
    )
    illumination_scale = illumination_excess / max(
        stats.illumination_nonuniformity, 1e-6
    )

    return CorrectionParameters(
        red_gain=_bounded_ratio(
            red_target, stats.red_green_ratio, bounds.maximum_channel_gain_change
        ),
        green_gain=1.0,
        blue_gain=_bounded_ratio(
            blue_target, stats.blue_green_ratio, bounds.maximum_channel_gain_change
        ),
        saturation_gain=float(
            np.clip(
                saturation_target / max(stats.saturation_median, 1e-6),
                bounds.minimum_saturation_gain,
                bounds.maximum_saturation_gain,
            )
        ),
        brightness_shift=float(
            np.clip(
                brightness_target - stats.brightness_median,
                -bounds.maximum_brightness_shift,
                bounds.maximum_brightness_shift,
            )
        ),
        contrast_gain=float(
            np.clip(
                contrast_target / max(stats.contrast_p95_p05, 1e-6),
                bounds.minimum_contrast_gain,
                bounds.maximum_contrast_gain,
            )
        ),
        gamma=float(
            np.clip(
                stats.gamma_to_midgray / max(gamma_target, 1e-6),
                bounds.minimum_gamma,
                bounds.maximum_gamma,
            )
        ),
        illumination_strength=float(
            np.clip(illumination_scale, 0.0, bounds.maximum_illumination_strength)
        ),
    )


def scale_corrections(
    parameters: CorrectionParameters,
    strength: float,
) -> CorrectionParameters:
    """Interpolate a correction between identity (0) and full strength (1)."""

    if not 0.0 <= strength <= 1.0:
        raise ValueError("correction_strength_must_be_between_zero_and_one")
    return CorrectionParameters(
        red_gain=1.0 + strength * (parameters.red_gain - 1.0),
        green_gain=1.0 + strength * (parameters.green_gain - 1.0),
        blue_gain=1.0 + strength * (parameters.blue_gain - 1.0),
        saturation_gain=1.0 + strength * (parameters.saturation_gain - 1.0),
        brightness_shift=strength * parameters.brightness_shift,
        contrast_gain=1.0 + strength * (parameters.contrast_gain - 1.0),
        gamma=1.0 + strength * (parameters.gamma - 1.0),
        illumination_strength=strength * parameters.illumination_strength,
    )


def select_correction_components(
    parameters: CorrectionParameters,
    enabled: set[str] | frozenset[str],
) -> CorrectionParameters:
    """Keep only named photometric components, setting all others to identity."""

    allowed = {
        "colour_balance",
        "saturation",
        "brightness",
        "contrast",
        "gamma",
        "illumination",
    }
    unknown = set(enabled) - allowed
    if unknown:
        raise ValueError(f"unknown_correction_components:{sorted(unknown)}")
    return CorrectionParameters(
        red_gain=parameters.red_gain if "colour_balance" in enabled else 1.0,
        green_gain=parameters.green_gain if "colour_balance" in enabled else 1.0,
        blue_gain=parameters.blue_gain if "colour_balance" in enabled else 1.0,
        saturation_gain=(
            parameters.saturation_gain if "saturation" in enabled else 1.0
        ),
        brightness_shift=(
            parameters.brightness_shift if "brightness" in enabled else 0.0
        ),
        contrast_gain=parameters.contrast_gain if "contrast" in enabled else 1.0,
        gamma=parameters.gamma if "gamma" in enabled else 1.0,
        illumination_strength=(
            parameters.illumination_strength if "illumination" in enabled else 0.0
        ),
    )


def _correct_illumination(
    rgb: np.ndarray, mask: np.ndarray, strength: float
) -> np.ndarray:
    if strength <= 0.0:
        return rgb
    lab = cv2.cvtColor(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    luminance = lab[..., 0].astype(np.float32) / 255.0
    kernel = max(31, ((min(luminance.shape) // 5) | 1))
    background = cv2.GaussianBlur(luminance, (kernel, kernel), 0)
    valid = mask > 0
    target = float(np.median(background[valid]))
    correction = target / np.maximum(background, 0.08)
    corrected = luminance * ((1.0 - strength) + strength * correction)
    lab[..., 0] = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def apply_corrections(
    image: np.ndarray,
    mask: np.ndarray,
    parameters: CorrectionParameters,
) -> np.ndarray:
    valid = mask > 0
    rgb = image.astype(np.float32) / 255.0
    rgb = _correct_illumination(rgb, mask, parameters.illumination_strength)

    gains = np.asarray(
        [parameters.red_gain, parameters.green_gain, parameters.blue_gain],
        dtype=np.float32,
    )
    rgb = rgb * gains.reshape(1, 1, 3)

    # Gamma affects mid-tones; contrast and brightness are applied around the
    # retinal luminance midpoint.  All values remain bounded to display range.
    rgb = np.power(np.clip(rgb, 0.0, 1.0), parameters.gamma)
    rgb = (rgb - 0.5) * parameters.contrast_gain + 0.5 + parameters.brightness_shift
    rgb = np.clip(rgb, 0.0, 1.0)

    hsv = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * parameters.saturation_gain, 0, 255)
    corrected = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    corrected[~valid] = 0
    return corrected
