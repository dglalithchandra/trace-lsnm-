"""Input validation and fundus-field detection."""

from __future__ import annotations

import cv2
import numpy as np

from .config import GeometryConfig
from .types import FundusDetection, ValidationResult


def validate_rgb_fundus_candidate(
    image: np.ndarray, config: GeometryConfig
) -> ValidationResult:
    reasons: list[str] = []
    warnings: list[str] = []
    if not isinstance(image, np.ndarray):
        return ValidationResult(False, ("input_is_not_numpy_array",))
    if image.ndim != 3 or image.shape[2] != 3:
        reasons.append("input_must_have_three_rgb_channels")
    if image.dtype != np.uint8:
        reasons.append("input_must_be_uint8_rgb")
    if image.ndim >= 2 and min(image.shape[:2]) < config.minimum_input_side:
        reasons.append("input_resolution_below_minimum")
    if image.size and not np.isfinite(image.astype(np.float32)).all():
        reasons.append("input_contains_non_finite_values")
    return ValidationResult(not reasons, tuple(reasons), tuple(warnings))


def detect_fundus_field(image: np.ndarray, config: GeometryConfig) -> FundusDetection:
    """Find the largest bright connected component against the dark fundus border."""

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold = max(config.background_threshold, int(np.percentile(gray, 2)))
    candidate = (gray > threshold).astype(np.uint8) * 255
    kernel_size = max(5, (min(image.shape[:2]) // 160) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("fundus_field_not_detected")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    image_area = float(image.shape[0] * image.shape[1])
    coverage = area / max(image_area, 1.0)
    if coverage < config.minimum_retinal_coverage:
        raise ValueError("retinal_coverage_below_minimum")

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    x, y, width, height = cv2.boundingRect(contour)
    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
    circularity = float(np.clip(4.0 * np.pi * area / (perimeter * perimeter), 0.0, 1.0))
    center_offset = np.hypot(
        center_x - image.shape[1] / 2.0,
        center_y - image.shape[0] / 2.0,
    ) / max(np.hypot(image.shape[1] / 2.0, image.shape[0] / 2.0), 1.0)
    confidence = float(np.clip(0.65 * circularity + 0.35 * (1.0 - center_offset), 0.0, 1.0))

    return FundusDetection(
        mask=mask,
        center_x=float(center_x),
        center_y=float(center_y),
        radius=float(radius),
        bbox_x=int(x),
        bbox_y=int(y),
        bbox_width=int(width),
        bbox_height=int(height),
        retinal_coverage=float(coverage),
        confidence=confidence,
    )
