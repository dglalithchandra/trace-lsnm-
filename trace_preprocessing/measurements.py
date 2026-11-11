"""Model-independent photometric and technical-quality measurements."""

from __future__ import annotations

import cv2
import numpy as np

from .types import PhotometricStats, QualityMetrics


def _valid_pixels(mask: np.ndarray) -> np.ndarray:
    valid = mask > 0
    if int(valid.sum()) < 64:
        raise ValueError("retinal_mask_has_too_few_pixels")
    return valid


def _illumination_nonuniformity(luminance: np.ndarray, valid: np.ndarray) -> float:
    height, width = luminance.shape
    tile_medians: list[float] = []
    for row in range(4):
        for column in range(4):
            y0, y1 = row * height // 4, (row + 1) * height // 4
            x0, x1 = column * width // 4, (column + 1) * width // 4
            tile_valid = valid[y0:y1, x0:x1]
            if tile_valid.mean() < 0.25:
                continue
            values = luminance[y0:y1, x0:x1][tile_valid]
            tile_medians.append(float(np.median(values)))
    if len(tile_medians) < 4:
        return 1.0
    low, high = np.percentile(tile_medians, [10, 90])
    return float(np.clip(high - low, 0.0, 1.0))


def measure_photometry(image: np.ndarray, mask: np.ndarray) -> PhotometricStats:
    valid = _valid_pixels(mask)
    rgb = image.astype(np.float32) / 255.0
    red_mean, green_mean, blue_mean = (
        float(rgb[..., channel][valid].mean()) for channel in range(3)
    )
    green_safe = max(green_mean, 1e-6)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[..., 1] / 255.0
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    luminance = lab[..., 0] / 255.0
    valid_luminance = luminance[valid]
    brightness = float(np.median(valid_luminance))
    p05, p95 = np.percentile(valid_luminance, [5, 95])
    brightness_safe = float(np.clip(brightness, 1e-4, 1.0 - 1e-4))
    gamma_to_midgray = float(np.clip(np.log(0.5) / np.log(brightness_safe), 0.25, 4.0))

    return PhotometricStats(
        red_mean=red_mean,
        green_mean=green_mean,
        blue_mean=blue_mean,
        red_green_ratio=float(red_mean / green_safe),
        blue_green_ratio=float(blue_mean / green_safe),
        saturation_median=float(np.median(saturation[valid])),
        brightness_median=brightness,
        contrast_p95_p05=float(p95 - p05),
        gamma_to_midgray=gamma_to_midgray,
        illumination_nonuniformity=_illumination_nonuniformity(luminance, valid),
        dark_clip_fraction=float(np.mean(valid_luminance <= 0.01)),
        bright_clip_fraction=float(np.mean(valid_luminance >= 0.99)),
    )


def _compression_blockiness(gray: np.ndarray, valid: np.ndarray) -> float:
    gray_float = gray.astype(np.float32) / 255.0
    column_boundaries = np.arange(8, gray.shape[1], 8)
    row_boundaries = np.arange(8, gray.shape[0], 8)
    vertical_boundary = np.abs(
        gray_float[:, column_boundaries] - gray_float[:, column_boundaries - 1]
    )
    horizontal_boundary = np.abs(
        gray_float[row_boundaries, :] - gray_float[row_boundaries - 1, :]
    )
    vertical_valid = valid[:, column_boundaries] & valid[:, column_boundaries - 1]
    horizontal_valid = valid[row_boundaries, :] & valid[row_boundaries - 1, :]
    values: list[float] = []
    if vertical_valid.any():
        values.append(float(vertical_boundary[vertical_valid].mean()))
    if horizontal_valid.any():
        values.append(float(horizontal_boundary[horizontal_valid].mean()))
    return float(np.mean(values)) if values else 0.0


def measure_quality(image: np.ndarray, mask: np.ndarray) -> QualityMetrics:
    valid = _valid_pixels(mask)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    erode_size = max(3, (min(gray.shape) // 80) | 1)
    eroded = cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size)),
    )
    inner = eroded > 0
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    blur_variance = float(np.var(laplacian[inner])) if inner.any() else 0.0

    median = cv2.medianBlur(gray, 5).astype(np.float32)
    high_frequency = gray.astype(np.float32) - median
    noise_mad = float(np.median(np.abs(high_frequency[inner]))) / 255.0 if inner.any() else 1.0

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    glare = (hsv[..., 2] >= 245) & (hsv[..., 1] <= 45) & valid
    glare_fraction = float(glare.sum() / max(valid.sum(), 1))

    return QualityMetrics(
        retinal_coverage=float(valid.mean()),
        blur_laplacian_variance=blur_variance,
        noise_mad=noise_mad,
        glare_fraction=glare_fraction,
        compression_blockiness=_compression_blockiness(gray, valid),
    )
