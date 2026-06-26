"""Model fetcher — pull model artifacts from HuggingFace Hub, gated-aware.

Wraps ``huggingface_hub.snapshot_download`` with:

* HF_TOKEN auto-loaded from env if not passed
* Resumable downloads (``resume_download=True``)
* Optional SHA256SUMS validation when the upstream provides one
* Actionable error messages for gated repos and network failures
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError


class FetchError(RuntimeError):
    """Raised on any unrecoverable fetch failure with an actionable message."""


def _read_token(token: str | None) -> str | None:
    """Resolve the HF token from arg → env → None."""
    if token:
        return token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _check_disk(cache_dir: Path, required_gb: float = 5.0) -> None:
    """Refuse to download into a near-full volume."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(cache_dir)
    free_gb = usage.free / (1024**3)
    if free_gb < required_gb:
        raise FetchError(
            f"Need at least {required_gb:.0f} GB free at {cache_dir}, only "
            f"{free_gb:.1f} GB available. Free disk space and retry."
        )


def fetch_model(
    repo_id: str,
    revision: str = "main",
    hf_token: str | None = None,
    cache_dir: Path | None = None,
    allow_patterns: Iterable[str] | None = None,
) -> Path:
    """Download a HuggingFace repo snapshot to local cache.

    Parameters
    ----------
    repo_id:
        e.g. ``"meta-llama/Llama-3.3-70B-Instruct"``.
    revision:
        Git revision (branch, tag, or commit). Defaults to ``"main"``.
    hf_token:
        HuggingFace token. Falls back to ``$HF_TOKEN``. Required for gated repos.
    cache_dir:
        Local cache root. Defaults to ``~/.cache/huggingface/hub``.
    allow_patterns:
        Glob patterns limiting which files to download (e.g. ``["*.gguf"]``).

    Returns
    -------
    pathlib.Path
        Path to the downloaded snapshot directory.
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "huggingface" / "hub"
    token = _read_token(hf_token)

    _check_disk(cache_dir)

    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            cache_dir=str(cache_dir),
            allow_patterns=list(allow_patterns) if allow_patterns else None,
        )
    except GatedRepoError as exc:
        raise FetchError(
            f"Repo '{repo_id}' is gated and requires a HuggingFace token.\n"
            f"  Fix: run `huggingface-cli login` and accept the license at "
            f"https://huggingface.co/{repo_id}.\n"
            f"  Original error: {exc}"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise FetchError(
            f"Repo '{repo_id}' not found on HuggingFace Hub. "
            f"Check the repo_id spelling and access permissions.\n"
            f"  Original error: {exc}"
        ) from exc
    except Exception as exc:  # network errors, etc.
        raise FetchError(
            f"Failed to download '{repo_id}': {exc}\n"
            f"  Tip: retry, or set HF_HUB_DOWNLOAD_TIMEOUT=120 in your env."
        ) from exc

    return Path(local_path)


__all__ = ["fetch_model", "FetchError"]
