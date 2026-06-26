# Contributing to edge-llm-benchmark

Thanks for taking the time to contribute. This document covers the most common
contribution types: adding a model to the matrix, fixing a benchmark bug, or
extending the report.

## Quick rules

- Open an issue first for non-trivial changes.
- All new code must have **unit tests** (`pytest tests/ -v`).
- All code must pass `ruff check .`, `ruff format --check .`, and `mypy src/`.
- Do **not** commit model artifacts (`.gguf`, `.safetensors`, `.bin`, …).
- Do **not** commit secrets (API keys, `HF_TOKEN`, …).

## Adding a model to the matrix

The matrix lives at `configs/matrix.yaml`. Open it and add an entry under
`models:`:

```yaml
  - id: org/model-name
    family: family-tag           # for plotly color grouping
    formats: [Q4_K_M, Q5_K_M]    # at least one format
    requires_ram_gb: 24          # conservative RAM at load (incl. KV cache)
```

A few rules of thumb:

- **Use the exact HuggingFace repo ID** under `id`. No nicknames.
- **Quantization formats** must be formats actually published by the upstream
  repo. If the repo only ships `Q4_K_M` and `Q5_K_M`, don't list `Q8_0`.
- **`requires_ram_gb`** is *available* RAM needed (model + KV cache + ~2 GB
  headroom). Use a conservative estimate — better to under-promote than to
  crash at load.
- If the model is **gated**, note it in the PR description. Users will need
  to run `huggingface-cli login` and accept the license.

After editing the matrix, validate it:

```bash
python -m edge_llm_bench.runner \
  --profile configs/profiles/<your-host>.yaml \
  --config configs/matrix.yaml \
  --dry-run
```

The dry-run prints the resolved (model × format) matrix and exits 0.

## Adding an eval suite

Eval suites live in `src/edge_llm_bench/quality_bench.py`. Each suite is a
function `run_<name>(backend, n=..., max_examples=None) -> QualityResult`.

To add a new suite:

1. Write the suite function. It must return a `QualityResult` with at least
   one of the canonical fields populated.
2. Add a fetch helper in `scripts/download_datasets.py` that produces a
   versioned JSONL in `datasets/`.
3. Wire it into `run_quality(suite="...")` orchestrator.
4. Add unit tests in `tests/test_quality_bench.py`.
5. Update `docs/runbook.md` if it requires extra setup (e.g. gated data).

## Adding an inference backend

1. Create `src/edge_llm_bench/inference/<name>_backend.py`.
2. Implement the `InferenceBackend` Protocol (see `base.py`).
3. Call `BackendRegistry.register("<name>", <YourClass>)` at module bottom.
4. Update `__init__.py` to import the new module.
5. Add a `pyproject.toml` extra so users can opt-in.
6. Write at least one integration-style test (mocked is fine).

## Running the test suite

```bash
pytest tests/ -v
ruff check .
ruff format --check .
mypy src/ --ignore-missing-imports
```

CI runs the same commands on every PR.

## Release process

1. Bump `version` in `pyproject.toml` and `src/edge_llm_bench/__init__.py`.
2. Update CHANGELOG (see `docs/CHANGELOG.md`).
3. Tag `git tag v0.X.Y && git push --tags`.
4. GitHub Actions builds + publishes to PyPI (manual for now).
