"""Eval dataset downloader — produces versioned subsets for the benchmark.

Run once: ``python scripts/download_datasets.py``. Datasets land in
``datasets/`` with stable filenames and are then committed to the repo so
benchmarks reproduce without network access.

We deliberately ship small subsets (MMLU 200, IFEval 100, HumanEval 30,
WikiText-103 sample 1 MB) — keeping full benchmarks in git would be hundreds
of MB and is not necessary for relative comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# MMLU (200 random from cais/mmlu "all" subset)
# ---------------------------------------------------------------------------


def fetch_mmlu(n: int = 200, seed: int = 42) -> Path:
    """Pull MMLU 'all' configuration, take a deterministic random subset.

    MMLU has 14k+ test questions across 57 subjects; 200 random questions
    give ~3.5 per subject on average — enough for a stable accuracy estimate
    on a per-run basis (full benchmark would take 5+ hours per model).
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    out = DATASETS_DIR / "mmlu_subset.jsonl"
    if out.exists():
        return out

    ds = load_dataset("cais/mmlu", "all", split="test")
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    rows = []
    for i in indices[:n]:
        item = ds[i]
        rows.append(
            {
                "question": item["question"],
                "choices": item["choices"],
                "answer": item["answer"],  # int 0..3
                "subject": item["subject"],
            }
        )
    _write_jsonl(out, rows)
    return out


# ---------------------------------------------------------------------------
# IFEval (100 prompts from google/IFEval)
# ---------------------------------------------------------------------------


def fetch_ifeval(n: int = 100, seed: int = 42) -> Path:
    from datasets import load_dataset  # type: ignore[import-untyped]

    out = DATASETS_DIR / "ifeval_subset.jsonl"
    if out.exists():
        return out

    ds = load_dataset("google/IFEval", split="train")
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    rows = []
    for i in indices[:n]:
        item = ds[i]
        rows.append(
            {
                "prompt": item["prompt"],
                "instruction_id_list": item["instruction_id_list"],
                "kwargs": item.get("kwargs", []),
            }
        )
    _write_jsonl(out, rows)
    return out


# ---------------------------------------------------------------------------
# HumanEval (30 problems from openai/openai_humaneval)
# ---------------------------------------------------------------------------


def fetch_humaneval(n: int = 30, seed: int = 42) -> Path:
    from datasets import load_dataset  # type: ignore[import-untyped]

    out = DATASETS_DIR / "humaneval_subset.jsonl"
    if out.exists():
        return out

    ds = load_dataset("openai/openai_humaneval", split="test")
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    rows = []
    for i in indices[:n]:
        item = ds[i]
        rows.append(
            {
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "canonical_solution": item["canonical_solution"],
                "test": item["test"],
                "entry_point": item["entry_point"],
            }
        )
    _write_jsonl(out, rows)
    return out


# ---------------------------------------------------------------------------
# WikiText-103 sample (~1 MB for perplexity)
# ---------------------------------------------------------------------------


def fetch_wikitext(max_bytes: int = 1_000_000) -> Path:
    from datasets import load_dataset  # type: ignore[import-untyped]

    out = DATASETS_DIR / "wikitext-103-sample.txt"
    if out.exists():
        return out

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    parts: list[str] = []
    total = 0
    for item in ds:
        text = item.get("text", "").strip()
        if not text:
            continue
        parts.append(text)
        total += len(text) + 1
        if total >= max_bytes:
            break
    body = "\n\n".join(parts)
    _write_text(out, body)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Download eval dataset subsets.")
    parser.add_argument("--mmlu", type=int, default=200)
    parser.add_argument("--ifeval", type=int, default=100)
    parser.add_argument("--humaneval", type=int, default=30)
    parser.add_argument("--wikitext-mb", type=int, default=1)
    args = parser.parse_args()

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    paths = [
        fetch_mmlu(n=args.mmlu),
        fetch_ifeval(n=args.ifeval),
        fetch_humaneval(n=args.humaneval),
        fetch_wikitext(max_bytes=args.wikitext_mb * 1024 * 1024),
    ]

    print("Datasets ready:")
    for p in paths:
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  {p.relative_to(DATASETS_DIR.parent)}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
