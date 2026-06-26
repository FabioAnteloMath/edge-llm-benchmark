"""Tests for the decision tree."""

from __future__ import annotations

import pandas as pd
import pytest

from edge_llm_bench.decision_tree import (
    VALID_STRATEGIES,
    Recommendation,
    _pareto_mask,
    _strategy_balanced,
    _strategy_max_quality,
    _strategy_max_speed,
    _strategy_min_ram,
    recommend,
)


@pytest.fixture
def fixture_results() -> pd.DataFrame:
    """Five configs spanning the quality/speed/RAM space."""
    return pd.DataFrame(
        [
            # high quality, slow, lots of RAM
            {
                "model_id": "big-70b",
                "format": "Q4_K_M",
                "ram_peak_gb": 42.0,
                "mmlu_acc": 0.82,
                "tokens_per_s_p50": 8.0,
                "ttft_ms_p50": 900,
                "humaneval_pass1": 0.55,
            },
            # mid quality, balanced
            {
                "model_id": "mid-32b",
                "format": "Q5_K_M",
                "ram_peak_gb": 22.0,
                "mmlu_acc": 0.75,
                "tokens_per_s_p50": 25.0,
                "ttft_ms_p50": 300,
                "humaneval_pass1": 0.40,
            },
            # lower quality, fast
            {
                "model_id": "small-14b",
                "format": "Q8_0",
                "ram_peak_gb": 16.0,
                "mmlu_acc": 0.68,
                "tokens_per_s_p50": 45.0,
                "ttft_ms_p50": 150,
                "humaneval_pass1": 0.30,
            },
            # tiny, very fast
            {
                "model_id": "tiny-7b",
                "format": "Q4_K_M",
                "ram_peak_gb": 5.0,
                "mmlu_acc": 0.55,
                "tokens_per_s_p50": 70.0,
                "ttft_ms_p50": 100,
                "humaneval_pass1": 0.20,
            },
            # dominated: low quality AND slow
            {
                "model_id": "dominated",
                "format": "Q4_K_M",
                "ram_peak_gb": 30.0,
                "mmlu_acc": 0.60,
                "tokens_per_s_p50": 10.0,
                "ttft_ms_p50": 700,
                "humaneval_pass1": 0.25,
            },
        ]
    )


@pytest.fixture
def mid_profile() -> dict:
    return {"host_name": "mid", "ram_available_gb": 24.0}


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------


def test_pareto_mask_finds_frontier(fixture_results: pd.DataFrame) -> None:
    mask = _pareto_mask(fixture_results, x="tokens_per_s_p50", y="mmlu_acc", higher_better_x=True)
    # big-70b (high MMLU, slow), mid-32b (both good), small-14b (fast), tiny-7b (fastest)
    # dominated should be excluded
    assert mask[4] is False  # the "dominated" row
    assert sum(mask) >= 2


def test_pareto_mask_handles_missing_columns() -> None:
    df = pd.DataFrame({"foo": [1, 2]})
    mask = _pareto_mask(df, x="missing", y="also_missing")
    assert mask == [False, False]


# ---------------------------------------------------------------------------
# Strategy sanity
# ---------------------------------------------------------------------------


def test_max_quality_picks_top_mmlu(fixture_results: pd.DataFrame) -> None:
    out = _strategy_max_quality(fixture_results)
    assert out.iloc[0]["model_id"] == "big-70b"


def test_max_speed_respects_ttft_cap(fixture_results: pd.DataFrame) -> None:
    out = _strategy_max_speed(fixture_results)
    # big-70b has TTFT 900 → excluded; tiny-7b should be top
    assert "big-70b" not in out["model_id"].values


def test_min_ram_requires_quality(fixture_results: pd.DataFrame) -> None:
    out = _strategy_min_ram(fixture_results)
    # tiny-7b has MMLU=0.55 < 0.70 → excluded
    assert "tiny-7b" not in out["model_id"].values


def test_balanced_returns_pareto_only(fixture_results: pd.DataFrame) -> None:
    out = _strategy_balanced(fixture_results)
    # dominated should be absent
    assert "dominated" not in out["model_id"].values


# ---------------------------------------------------------------------------
# recommend (orchestrator)
# ---------------------------------------------------------------------------


def test_recommend_balanced_returns_recommendations(
    fixture_results: pd.DataFrame, mid_profile: dict
) -> None:
    recs = recommend(mid_profile, fixture_results, strategy="balanced")
    assert 1 <= len(recs) <= 5
    assert all(isinstance(r, Recommendation) for r in recs)
    assert recs[0].rank == 1


def test_recommend_filters_unrunnable(fixture_results: pd.DataFrame) -> None:
    tiny_profile = {"host_name": "tiny", "ram_available_gb": 10.0}
    recs = recommend(tiny_profile, fixture_results, strategy="max-quality")
    # big-70b needs 42 GB → excluded; mid-32b needs 22 GB → also excluded
    ids = {r.model_id for r in recs}
    assert "big-70b" not in ids
    assert "mid-32b" not in ids


def test_recommend_unknown_strategy_raises(fixture_results: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        recommend({}, fixture_results, strategy="nonsense")


def test_recommend_empty_runnable_returns_empty() -> None:
    df = pd.DataFrame({"model_id": ["x"], "format": ["Q4"], "ram_peak_gb": [100.0]})
    recs = recommend({"ram_available_gb": 1.0}, df, strategy="balanced")
    assert recs == []


def test_valid_strategies_complete() -> None:
    expected = {"max-quality", "max-speed", "min-ram", "balanced", "code"}
    assert set(VALID_STRATEGIES) == expected
