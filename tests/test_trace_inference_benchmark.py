from __future__ import annotations

import pytest

from scripts.benchmark_trace_a1_inference import summarize_timings


def test_timing_summary_uses_all_images() -> None:
    result = summarize_timings([0.1, 0.1], batch_size=4)
    assert result["images_per_second"] == pytest.approx(40.0)
    assert result["median_batch_milliseconds"] == pytest.approx(100.0)
    assert result["iterations"] == 2


def test_timing_summary_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        summarize_timings([0.0], batch_size=1)
