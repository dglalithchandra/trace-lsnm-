"""Before/after image comparison and provisional safety decisions."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .config import QualityPolicy
from .types import (
    ComparisonMetrics,
    PhotometricStats,
    ProcessingStatus,
    QualityMetrics,
)


def compare_images(
    geometry_only: np.ndarray,
    processed: np.ndarray,
    mask: np.ndarray,
    stats_before: PhotometricStats,
    stats_after: PhotometricStats,
) -> ComparisonMetrics:
    valid = mask > 0
    difference = np.abs(
        processed.astype(np.float32) - geometry_only.astype(np.float32)
    ) / 255.0
    valid_difference = difference[valid]
    gray_before = cv2.cvtColor(geometry_only, cv2.COLOR_RGB2GRAY)
    gray_after = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    masked_before = gray_before.copy()
    masked_after = gray_after.copy()
    masked_before[~valid] = 0
    masked_after[~valid] = 0
    ssim = structural_similarity(masked_before, masked_after, data_range=255)
    return ComparisonMetrics(
        mean_absolute_change=float(valid_difference.mean()),
        p95_absolute_change=float(np.percentile(valid_difference, 95)),
        structural_similarity=float(ssim),
        dark_clip_increase=float(
            stats_after.dark_clip_fraction - stats_before.dark_clip_fraction
        ),
        bright_clip_increase=float(
            stats_after.bright_clip_fraction - stats_before.bright_clip_fraction
        ),
    )


def decide_quality(
    comparison: ComparisonMetrics,
    quality_before: QualityMetrics,
    quality_after: QualityMetrics,
    policy: QualityPolicy,
) -> tuple[ProcessingStatus, list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    if comparison.mean_absolute_change > policy.maximum_mean_absolute_change:
        reasons.append("mean_image_change_exceeds_development_guard")
    if comparison.p95_absolute_change > policy.maximum_p95_absolute_change:
        reasons.append("p95_image_change_exceeds_development_guard")
    if comparison.structural_similarity < policy.minimum_structural_similarity:
        reasons.append("structural_similarity_below_development_guard")
    if comparison.bright_clip_increase > policy.maximum_bright_clip_increase:
        reasons.append("bright_clipping_increased_too_much")
    if comparison.dark_clip_increase > policy.maximum_dark_clip_increase:
        reasons.append("dark_clipping_increased_too_much")
    if quality_after.retinal_coverage < policy.minimum_output_retinal_coverage:
        reasons.append("output_retinal_coverage_below_guard")

    if quality_before.blur_laplacian_variance < policy.provisional_blur_warning_below:
        warnings.append("provisional_blur_warning")
    if quality_before.glare_fraction > policy.provisional_glare_warning_above:
        warnings.append("provisional_glare_warning")
    if quality_after.noise_mad > quality_before.noise_mad * 1.25 + 1e-6:
        warnings.append("noise_increased_after_processing")
    if quality_after.compression_blockiness > quality_before.compression_blockiness * 1.25 + 1e-6:
        warnings.append("blockiness_increased_after_processing")

    if reasons:
        return ProcessingStatus.REJECT, reasons, warnings
    if warnings:
        return ProcessingStatus.WARNING, reasons, warnings
    return ProcessingStatus.ACCEPT, reasons, warnings
