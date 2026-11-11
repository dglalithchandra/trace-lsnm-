from __future__ import annotations

import torch

from trace_models.rfmid_dinov2 import (
    DECISION_TOLERANCE,
    RFMiDLinearHead,
    probabilities_from_embedding,
)
from trace_models.shift_adapter import ShiftConsistencyAdapter
from trace_models.rare_adapter import RareAwareAdapter


def test_probabilities_have_expected_shape_and_range() -> None:
    model = RFMiDLinearHead(embedding_width=4, disease_count=3).eval()
    bundle = {
        "feature_mean": torch.zeros(4),
        "feature_scale": torch.ones(4),
        "temperature": 1.0,
    }
    probabilities = probabilities_from_embedding(
        model, torch.zeros((2, 4)), bundle
    )
    assert probabilities.shape == (2, 3)
    assert torch.isfinite(probabilities).all()
    assert torch.all((probabilities >= 0) & (probabilities <= 1))


def test_temperature_changes_score_sharpness_not_shape() -> None:
    model = RFMiDLinearHead(embedding_width=2, disease_count=1).eval()
    with torch.no_grad():
        model.classifier.weight.fill_(1.0)
        model.classifier.bias.zero_()
    embedding = torch.ones((1, 2))
    base = {
        "feature_mean": torch.zeros(2),
        "feature_scale": torch.ones(2),
        "temperature": 1.0,
    }
    sharper = {**base, "temperature": 0.5}
    base_score = probabilities_from_embedding(model, embedding, base)
    sharper_score = probabilities_from_embedding(model, embedding, sharper)
    assert sharper_score.shape == base_score.shape
    assert sharper_score.item() > base_score.item()


def test_shift_adapter_starts_as_exact_identity() -> None:
    adapter = ShiftConsistencyAdapter(width=4, hidden_width=3).eval()
    inputs = torch.randn(5, 4)
    with torch.inference_mode():
        outputs = adapter(inputs)
    assert torch.equal(outputs, inputs)


def test_rare_adapter_starts_as_normalized_identity() -> None:
    adapter = RareAwareAdapter(width=4, hidden_width=3).eval()
    inputs = torch.randn(5, 4)
    with torch.inference_mode():
        outputs = adapter(inputs)
    expected = torch.nn.functional.normalize(inputs, dim=-1)
    assert torch.allclose(outputs, expected, atol=1e-7)
    assert torch.allclose(outputs.norm(dim=1), torch.ones(5), atol=1e-6)


def test_threshold_tolerance_absorbs_float32_roundoff() -> None:
    threshold = 0.8
    rounded_score = threshold - 5e-7
    assert rounded_score >= threshold - DECISION_TOLERANCE
