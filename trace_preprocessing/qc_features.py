"""Deployment-available, model-independent features for artifact-aware QC."""

from __future__ import annotations

import cv2
import numpy as np

from .measurements import measure_photometry, measure_quality


def qc_feature_names() -> tuple[str, ...]:
    return (
        "red_mean",
        "green_mean",
        "blue_mean",
        "red_green_ratio",
        "blue_green_ratio",
        "saturation_median",
        "brightness_median",
        "contrast_p95_p05",
        "gamma_to_midgray",
        "illumination_nonuniformity",
        "dark_clip_fraction",
        "bright_clip_fraction",
        "retinal_coverage",
        "log_blur_laplacian_variance",
        "noise_mad",
        "glare_fraction",
        "compression_blockiness",
        "highlight_loose_fraction",
        "highlight_strict_fraction",
        "largest_highlight_component_fraction",
        "gradient_mean",
        "gradient_p90",
        "edge_density",
        "peripheral_gradient_mean",
        "central_gradient_mean",
        "peripheral_to_central_gradient_ratio",
        "peripheral_brightness_mean",
        "central_brightness_mean",
        "peripheral_to_central_brightness_ratio",
        "local_brightness_median_std",
        "gray_entropy",
        "vessel_proxy_fraction",
        "vessel_proxy_peripheral_fraction",
    )


def _fraction(mask: np.ndarray, valid: np.ndarray) -> float:
    return float(np.sum(mask & valid) / max(int(valid.sum()), 1))


def extract_qc_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return finite absolute QC features from one normalized image."""

    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != mask.shape:
        raise ValueError("qc_feature_image_mask_shape_mismatch")
    valid = mask > 0
    if int(valid.sum()) < 64:
        raise ValueError("qc_feature_mask_too_small")
    photometry = measure_photometry(image, mask)
    quality = measure_quality(image, mask)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    value = hsv[..., 2]
    saturation = hsv[..., 1]

    loose_highlight = (value >= 225) & (saturation <= 80) & valid
    strict_highlight = (value >= 245) & (saturation <= 45) & valid
    components, labels, stats, _ = cv2.connectedComponentsWithStats(
        loose_highlight.astype(np.uint8), connectivity=8
    )
    largest_component = 0
    if components > 1:
        largest_component = int(stats[1:, cv2.CC_STAT_AREA].max())
    largest_highlight_fraction = largest_component / max(int(valid.sum()), 1)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    gradient_values = gradient[valid]

    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]].astype(np.float32)
    valid_y, valid_x = np.nonzero(valid)
    center_x = float(valid_x.mean())
    center_y = float(valid_y.mean())
    radius = max(
        float(
            np.sqrt((valid_x - center_x) ** 2 + (valid_y - center_y) ** 2).max()
        ),
        1.0,
    )
    radial = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2) / radius
    central = valid & (radial <= 0.55)
    peripheral = valid & (radial >= 0.72)
    central_gradient = float(gradient[central].mean()) if central.any() else 0.0
    peripheral_gradient = (
        float(gradient[peripheral].mean()) if peripheral.any() else 0.0
    )
    central_brightness = float(gray[central].mean()) if central.any() else 0.0
    peripheral_brightness = (
        float(gray[peripheral].mean()) if peripheral.any() else 0.0
    )

    tile_medians = []
    height, width = mask.shape
    for row in range(4):
        for column in range(4):
            y0, y1 = row * height // 4, (row + 1) * height // 4
            x0, x1 = column * width // 4, (column + 1) * width // 4
            tile_valid = valid[y0:y1, x0:x1]
            if tile_valid.mean() >= 0.25:
                tile_medians.append(
                    float(np.median(gray[y0:y1, x0:x1][tile_valid]))
                )

    histogram, _ = np.histogram(gray[valid], bins=64, range=(0.0, 1.0))
    probabilities = histogram.astype(np.float64) / max(histogram.sum(), 1)
    nonzero = probabilities > 0
    entropy = float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))

    green = image[..., 1].astype(np.float32) / 255.0
    local_background = cv2.GaussianBlur(green, (0, 0), sigmaX=3.0)
    vessel_proxy = (green < local_background - 0.035) & valid

    values = np.asarray(
        [
            *photometry.to_dict().values(),
            quality.retinal_coverage,
            np.log1p(quality.blur_laplacian_variance),
            quality.noise_mad,
            quality.glare_fraction,
            quality.compression_blockiness,
            _fraction(loose_highlight, valid),
            _fraction(strict_highlight, valid),
            largest_highlight_fraction,
            float(gradient_values.mean()),
            float(np.quantile(gradient_values, 0.90)),
            float(np.mean(gradient_values >= 0.12)),
            peripheral_gradient,
            central_gradient,
            peripheral_gradient / max(central_gradient, 1e-6),
            peripheral_brightness,
            central_brightness,
            peripheral_brightness / max(central_brightness, 1e-6),
            float(np.std(tile_medians)) if tile_medians else 0.0,
            entropy,
            _fraction(vessel_proxy, valid),
            _fraction(vessel_proxy, peripheral),
        ],
        dtype=np.float32,
    )
    if len(values) != len(qc_feature_names()) or not np.isfinite(values).all():
        raise RuntimeError("invalid_qc_feature_vector")
    return values
