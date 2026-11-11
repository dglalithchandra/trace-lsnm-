from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.adapters import prepare_dinov2_tensor
from trace_preprocessing.config import GeometryConfig
from trace_preprocessing.geometry import transform_mask
from trace_preprocessing.loader import ManifestImageLoader
from trace_preprocessing.photometric import (
    scale_corrections,
    select_correction_components,
)
from trace_preprocessing.reference import build_reference_profile
from trace_preprocessing.types import CorrectionParameters
from trace_preprocessing.stress import (
    apply_artifact_stress,
    apply_camera_stress,
    sample_artifact_stress_parameters,
    sample_stress_parameters,
)
from trace_preprocessing.qc_features import extract_qc_features, qc_feature_names


WORKSPACE = Path(__file__).resolve().parents[1]


def make_synthetic_fundus(
    *, brightness_scale: float = 1.0, uneven: bool = False
) -> np.ndarray:
    height, width = 600, 800
    image = np.zeros((height, width, 3), dtype=np.uint8)
    center = (400, 300)
    radius = 250
    colour = np.clip(np.asarray([155, 92, 52]) * brightness_scale, 0, 255).astype(np.uint8)
    cv2.circle(image, center, radius, tuple(int(value) for value in colour), cv2.FILLED)
    cv2.circle(image, (485, 285), 24, (225, 182, 95), cv2.FILLED)
    for end in ((250, 190), (230, 340), (390, 155), (330, 420), (550, 390)):
        cv2.line(image, (485, 285), end, (105, 38, 35), 5, cv2.LINE_AA)
    cv2.circle(image, (360, 310), 5, (230, 205, 120), cv2.FILLED)
    if uneven:
        overlay = np.zeros_like(image)
        cv2.circle(overlay, (520, 205), 170, (55, 55, 55), cv2.FILLED)
        retinal = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(retinal, center, radius, 255, cv2.FILLED)
        image = cv2.add(image, overlay, mask=retinal)
    return image


def test_geometry_pipeline_preserves_original_and_returns_common_output() -> None:
    image = make_synthetic_fundus()
    original_copy = image.copy()
    config = PreprocessingConfig(apply_photometric_correction=False)
    result = TracePreprocessor(config).process(image, image_id="synthetic-1")

    assert result.status in {ProcessingStatus.ACCEPT, ProcessingStatus.WARNING}
    assert np.array_equal(image, original_copy)
    assert np.array_equal(result.original_image, original_copy)
    assert result.geometry_only_image is not None
    assert result.geometry_only_image.shape == (448, 448, 3)
    assert result.processed_image is not None
    assert result.retinal_mask is not None
    assert result.retinal_mask.shape == (448, 448)
    assert np.all(result.processed_image[result.retinal_mask == 0] == 0)
    assert "photometric_correction_skipped_without_reference" not in result.warnings


def test_reference_is_train_only_and_correction_stays_bounded() -> None:
    geometry_pipeline = TracePreprocessor(
        PreprocessingConfig(apply_photometric_correction=False)
    )
    development_images = [
        make_synthetic_fundus(brightness_scale=scale, uneven=index == 2)
        for index, scale in enumerate((0.90, 1.00, 1.08))
    ]
    development_results = [
        geometry_pipeline.process(image, image_id=f"reference-{index}")
        for index, image in enumerate(development_images)
    ]
    stats = [result.stats_before for result in development_results]
    assert all(item is not None for item in stats)
    reference = build_reference_profile(
        [item for item in stats if item is not None],
        ["train"] * len(stats),
        profile_name="unit-test-reference",
    )
    with pytest.raises(ValueError, match="train_only"):
        build_reference_profile(
            [item for item in stats if item is not None],
            ["train", "train", "validation"],
        )

    result = TracePreprocessor().process(
        make_synthetic_fundus(brightness_scale=0.72, uneven=True),
        image_id="corrected",
        reference=reference,
    )
    assert result.processed_image is not None
    assert result.corrections is not None
    bounds = PreprocessingConfig().photometric_bounds
    assert 1.0 - bounds.maximum_channel_gain_change <= result.corrections.red_gain <= 1.0 + bounds.maximum_channel_gain_change
    assert bounds.minimum_saturation_gain <= result.corrections.saturation_gain <= bounds.maximum_saturation_gain
    assert bounds.minimum_gamma <= result.corrections.gamma <= bounds.maximum_gamma
    assert result.corrections.illumination_strength <= bounds.maximum_illumination_strength
    json.dumps(result.audit_dict())


def test_annotation_transform_and_dinov2_adapter() -> None:
    source_mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.circle(source_mask, (400, 300), 30, 255, cv2.FILLED)
    result = TracePreprocessor(
        PreprocessingConfig(apply_photometric_correction=False)
    ).process(
        make_synthetic_fundus(),
        image_id="annotation",
        annotation_masks={"synthetic_lesion": source_mask},
    )
    assert result.geometry_transform is not None
    transformed = transform_mask(
        source_mask,
        result.geometry_transform,
        result.retinal_mask,
    )
    assert transformed.shape == (448, 448)
    assert transformed.max() == 255
    transformed_center = result.geometry_transform.transform_point(400, 300)
    assert transformed_center[0] == pytest.approx(224, abs=2)
    assert transformed_center[1] == pytest.approx(224, abs=2)
    preservation = result.annotation_preservation["synthetic_lesion"]
    assert preservation.transformed_foreground_pixels > 0
    assert preservation.source_region_clipped_fraction == pytest.approx(0.0)
    assert preservation.mean_absolute_change_in_region == pytest.approx(0.0)

    tensor = prepare_dinov2_tensor(result.processed_image)
    assert tuple(tensor.shape) == (3, 448, 448)
    assert tensor.dtype == torch.float32


def test_invalid_image_is_rejected() -> None:
    invalid = np.zeros((100, 100), dtype=np.uint8)
    result = TracePreprocessor().process(invalid, image_id="invalid")
    assert result.status == ProcessingStatus.REJECT
    assert "input_must_have_three_rgb_channels" in result.reasons


def test_scale_corrections_interpolates_from_identity() -> None:
    full = CorrectionParameters(
        red_gain=1.12,
        green_gain=1.0,
        blue_gain=0.88,
        saturation_gain=1.10,
        brightness_shift=-0.08,
        contrast_gain=0.90,
        gamma=1.18,
        illumination_strength=0.30,
    )
    identity = scale_corrections(full, 0.0)
    half = scale_corrections(full, 0.5)
    assert identity == CorrectionParameters()
    assert half.red_gain == pytest.approx(1.06)
    assert half.blue_gain == pytest.approx(0.94)
    assert half.saturation_gain == pytest.approx(1.05)
    assert half.brightness_shift == pytest.approx(-0.04)
    assert half.contrast_gain == pytest.approx(0.95)
    assert half.gamma == pytest.approx(1.09)
    assert half.illumination_strength == pytest.approx(0.15)


def test_scale_corrections_rejects_out_of_range_strength() -> None:
    with pytest.raises(ValueError, match="correction_strength"):
        scale_corrections(CorrectionParameters(), 1.01)


def test_select_correction_components_keeps_only_requested_operation() -> None:
    full = CorrectionParameters(
        red_gain=1.1,
        blue_gain=0.9,
        saturation_gain=1.1,
        brightness_shift=0.05,
        contrast_gain=0.9,
        gamma=1.1,
        illumination_strength=0.2,
    )
    colour_only = select_correction_components(full, {"colour_balance"})
    assert colour_only.red_gain == pytest.approx(1.1)
    assert colour_only.blue_gain == pytest.approx(0.9)
    assert colour_only.saturation_gain == 1.0
    assert colour_only.brightness_shift == 0.0
    assert colour_only.contrast_gain == 1.0
    assert colour_only.gamma == 1.0
    assert colour_only.illumination_strength == 0.0
    with pytest.raises(ValueError, match="unknown_correction_components"):
        select_correction_components(full, {"sharpness"})


def test_camera_stress_is_deterministic_and_preserves_mask() -> None:
    result = TracePreprocessor(
        PreprocessingConfig(apply_photometric_correction=False)
    ).process(make_synthetic_fundus(), image_id="stress-source")
    assert result.processed_image is not None and result.retinal_mask is not None
    parameters = sample_stress_parameters(
        "stress-source", seed_namespace="unit-test"
    )
    first = apply_camera_stress(
        result.processed_image,
        result.retinal_mask,
        parameters,
        image_id="stress-source",
        seed_namespace="unit-test",
    )
    second = apply_camera_stress(
        result.processed_image,
        result.retinal_mask,
        parameters,
        image_id="stress-source",
        seed_namespace="unit-test",
    )
    assert first.shape == result.processed_image.shape
    assert np.array_equal(first, second)
    assert np.all(first[result.retinal_mask == 0] == 0)


def test_artifact_stresses_are_deterministic_and_preserve_mask() -> None:
    result = TracePreprocessor(
        PreprocessingConfig(apply_photometric_correction=False)
    ).process(make_synthetic_fundus(), image_id="artifact-source")
    assert result.geometry_only_image is not None
    assert result.retinal_mask is not None
    for style in ("glare", "resolution_loss", "narrow_field_of_view"):
        parameters = sample_artifact_stress_parameters(
            "artifact-source", style=style, seed_namespace="unit-test"
        )
        first = apply_artifact_stress(
            result.geometry_only_image,
            result.retinal_mask,
            parameters,
            image_id="artifact-source",
            seed_namespace="unit-test",
        )
        second = apply_artifact_stress(
            result.geometry_only_image,
            result.retinal_mask,
            parameters,
            image_id="artifact-source",
            seed_namespace="unit-test",
        )
        assert np.array_equal(first, second)
        assert np.all(first[result.retinal_mask == 0] == 0)
        assert np.any(first != result.geometry_only_image)


def test_qc_features_are_finite_and_named() -> None:
    result = TracePreprocessor(
        PreprocessingConfig(apply_photometric_correction=False)
    ).process(make_synthetic_fundus(), image_id="qc-feature-source")
    assert result.geometry_only_image is not None
    assert result.retinal_mask is not None
    values = extract_qc_features(result.geometry_only_image, result.retinal_mask)
    assert values.shape == (len(qc_feature_names()),)
    assert np.isfinite(values).all()


def test_manifest_loader_blocks_test_and_streams_training_image() -> None:
    loader = ManifestImageLoader(WORKSPACE)
    test_id = loader.record_ids(split="validation", limit=1)[0]
    image, row = loader.load_image(test_id)
    assert row["final_split"] == "validation"
    assert image.ndim == 3 and image.shape[2] == 3
    blocked_id = loader.manifest.loc[
        loader.manifest["final_split"] == "test", "record_id"
    ].iloc[0]
    with pytest.raises(PermissionError, match="test_split_access"):
        loader.load_image(blocked_id)


def test_manifest_loader_reads_idrid_annotation_masks() -> None:
    loader = ManifestImageLoader(WORKSPACE)
    record_id = loader.record_ids(
        split="train", dataset="IDRiD_Segmentation", limit=1
    )[0]
    image, _ = loader.load_image(record_id)
    masks = loader.load_annotation_masks(record_id)
    assert masks
    assert {"microaneurysms", "hard_exudates", "optic_disc"} <= set(masks)
    assert all(mask.shape == image.shape[:2] for mask in masks.values())


def test_fives_loader_reads_image_and_vessel_mask() -> None:
    loader = ManifestImageLoader(
        WORKSPACE,
        WORKSPACE / "data" / "manifests" / "fives_manifest.csv",
    )
    record_id = loader.record_ids(split="train", limit=1)[0]
    image, row = loader.load_image(record_id)
    masks = loader.load_annotation_masks(record_id)
    assert row["_effective_split"] == "train"
    assert image.shape == (2048, 2048, 3)
    assert masks["vessels"].shape == image.shape[:2]
    blocked_id = loader.manifest.loc[
        loader.manifest["_effective_split"] == "test", "record_id"
    ].iloc[0]
    with pytest.raises(PermissionError, match="test_split_access"):
        loader.load_image(blocked_id)
