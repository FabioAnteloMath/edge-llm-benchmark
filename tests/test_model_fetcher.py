"""Tests for the model fetcher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from edge_llm_bench.model_fetcher import FetchError, fetch_model


def test_fetch_model_passes_token_from_env(tmp_path: Path) -> None:
    fake_snapshot = MagicMock(return_value=str(tmp_path / "snap"))
    with (
        patch.dict("os.environ", {"HF_TOKEN": "env-token-xyz"}, clear=False),
        patch("edge_llm_bench.model_fetcher.snapshot_download", fake_snapshot),
    ):
        result = fetch_model("microsoft/Phi-4-14B-Instruct", cache_dir=tmp_path)

    assert result == tmp_path / "snap"
    kwargs = fake_snapshot.call_args.kwargs
    assert kwargs["token"] == "env-token-xyz"
    # Resumable downloads are the default in modern huggingface_hub
    # (resume_download kwarg was removed; the library always resumes partial files).
    assert "cache_dir" in kwargs


def test_fetch_model_explicit_token_wins_over_env(tmp_path: Path) -> None:
    fake_snapshot = MagicMock(return_value=str(tmp_path / "snap"))
    with (
        patch.dict("os.environ", {"HF_TOKEN": "env-token"}, clear=False),
        patch("edge_llm_bench.model_fetcher.snapshot_download", fake_snapshot),
    ):
        fetch_model(
            "microsoft/Phi-4-14B-Instruct",
            hf_token="explicit-token",
            cache_dir=tmp_path,
        )
    assert fake_snapshot.call_args.kwargs["token"] == "explicit-token"


def _fake_response() -> MagicMock:
    """A mock HTTP response that satisfies HuggingFace error constructors."""
    resp = MagicMock()
    resp.headers = {}
    resp.status_code = 403
    return resp


def test_fetch_model_gated_repo_raises_actionable(tmp_path: Path) -> None:
    fake_snapshot = MagicMock(side_effect=GatedRepoError("gated", response=_fake_response()))
    with patch("edge_llm_bench.model_fetcher.snapshot_download", fake_snapshot):
        with pytest.raises(FetchError) as excinfo:
            fetch_model("meta-llama/Llama-3.3-70B-Instruct", cache_dir=tmp_path)
    msg = str(excinfo.value)
    assert "gated" in msg.lower() or "huggingface-cli login" in msg


def test_fetch_model_not_found_raises_actionable(tmp_path: Path) -> None:
    fake_snapshot = MagicMock(
        side_effect=RepositoryNotFoundError("missing", response=_fake_response())
    )
    with patch("edge_llm_bench.model_fetcher.snapshot_download", fake_snapshot):
        with pytest.raises(FetchError) as excinfo:
            fetch_model("nonexistent/repo", cache_dir=tmp_path)
    assert "not found" in str(excinfo.value).lower()


def test_fetch_model_network_error_raises_actionable(tmp_path: Path) -> None:
    fake_snapshot = MagicMock(side_effect=ConnectionError("boom"))
    with patch("edge_llm_bench.model_fetcher.snapshot_download", fake_snapshot):
        with pytest.raises(FetchError) as excinfo:
            fetch_model("any/repo", cache_dir=tmp_path)
    assert "retry" in str(excinfo.value).lower()


def test_fetch_model_allow_patterns_passed_through(tmp_path: Path) -> None:
    fake_snapshot = MagicMock(return_value=str(tmp_path / "snap"))
    with (
        patch.dict("os.environ", {}, clear=False),
        patch("edge_llm_bench.model_fetcher.snapshot_download", fake_snapshot),
    ):
        # strip HF_TOKEN so it's None
        with patch.dict("os.environ", {"HF_TOKEN": ""}, clear=False):
            fetch_model(
                "microsoft/Phi-4-14B-Instruct",
                cache_dir=tmp_path,
                allow_patterns=["*.gguf", "*.md"],
            )
    assert fake_snapshot.call_args.kwargs["allow_patterns"] == ["*.gguf", "*.md"]


def test_fetch_model_aborts_when_disk_full(tmp_path: Path) -> None:
    fake_snapshot = MagicMock()
    with patch(
        "edge_llm_bench.model_fetcher._check_disk",
        side_effect=FetchError("no disk"),
    ):
        with pytest.raises(FetchError, match="no disk"):
            fetch_model("any/repo", cache_dir=tmp_path)
    fake_snapshot.assert_not_called()
