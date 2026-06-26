# Operational Runbook

> **Read this before running a benchmark.** Every failure mode you'll hit has a known fix, listed below.

This runbook covers three test levels (smoke / single-model / full), a troubleshooting matrix for the ~12 most common errors, and operational notes for keeping the project healthy.

---

## 1. Test Levels

### 1.1 Smoke test (≈ 5 min)

**Goal:** confirm the environment is healthy without downloading any model.

**Commands (PowerShell / Windows):**

```powershell
# Activate venv
.venv\Scripts\Activate.ps1

# Verify package is importable
python -c "import edge_llm_bench; print(edge_llm_bench.__version__)"

# Run all unit tests
pytest tests/ -v

# Profile your machine
python -c "from edge_llm_bench.hardware_profiler import profile_hardware; print(profile_hardware())"

# Dry-run: prints resolved matrix without running anything
python -m edge_llm_bench.runner --profile configs/profiles/<host>.yaml --dry-run

# Ollama sanity (skip if using llama.cpp only)
ollama --version
ollama list
```

**Commands (bash / macOS / Linux):**

```bash
source .venv/bin/activate
python -c "import edge_llm_bench; print(edge_llm_bench.__version__)"
pytest tests/ -v
python -c "from edge_llm_bench.hardware_profiler import profile_hardware; print(profile_hardware())"
python -m edge_llm_bench.runner --profile configs/profiles/<host>.yaml --dry-run
ollama --version
ollama list
```

**Pass criteria:**

- All pytest tests pass.
- `profile_hardware()` returns a populated `HardwareProfile` with realistic values.
- `--dry-run` exits 0 and prints a matrix table with ≥ 1 row.

---

### 1.2 Single-model smoke (≈ 30–60 min)

**Goal:** end-to-end with one small model (Phi-4 14B or smaller). Produces a real CSV row and HTML report.

**Commands:**

```powershell
# Activate venv
.venv\Scripts\Activate.ps1

# Pre-flight: dry-run with the host profile
python -m edge_llm_bench.runner --profile configs/profiles/<host>.yaml --dry-run

# Real run with a small eval subset (5 prompts per suite)
python -m edge_llm_bench.runner `
  --profile configs/profiles/<host>.yaml `
  --max-examples 5 `
  --output-dir results/smoke

# Inspect the CSV (should have 1 row, status=ok)
Get-Content results\smoke\results.csv

# Generate the report
python -m edge_llm_bench.report --csv results\smoke\results.csv --out docs\report-smoke.html

# Open it
start docs\report-smoke.html
```

**Pass criteria:**

- `results/smoke/results.csv` exists and has at least 1 row.
- `status=ok` (or `partial` if e.g. quality suite timed out).
- TTFT < 5 s, tokens/s > 5 (numbers should be plausible).
- HTML opens with at least 1 chart populated.

---

### 1.3 Full benchmark (≈ 2–6 hours)

**Goal:** full matrix, full eval suites, publishable report.

**Commands:**

```powershell
# Full run on a fresh output dir
python -m edge_llm_bench.runner `
  --profile configs/profiles/<host>.yaml `
  --output-dir results\2026-06-23

# If interrupted (Ctrl-C), resume:
python -m edge_llm_bench.runner `
  --profile configs/profiles/<host>.yaml `
  --output-dir results\2026-06-23 `
  --resume

# Generate the latest report
python -m edge_llm_bench.report --latest

# Get recommendations for this hardware
python -m edge_llm_bench.decision_tree `
  --profile configs/profiles/<host>.yaml `
  --strategy balanced
```

**Pass criteria:**

- `results/<date>/results.csv` has ≥ 5 rows.
- `docs/report.html` renders all 5 charts with non-empty data.
- Decision tree returns ≥ 3 ranked recommendations with rationales.

---

## 2. Troubleshooting Matrix

| # | Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|---|
| 1 | `ImportError: No module named edge_llm_bench` | Not installed in editable mode | `pip show edge-llm-bench` | `pip install -e ".[dev]"` |
| 2 | `ollama: command not found` | Ollama not installed | `ollama --version` | Install from <https://ollama.com> |
| 3 | `ConnectionError: localhost:11434` | Ollama daemon not running | `curl http://localhost:11434/api/tags` | `ollama serve` in another terminal |
| 4 | `GatedRepoError: meta-llama/...` | `HF_TOKEN` missing | `echo $HF_TOKEN` (or `$env:HF_TOKEN`) | `huggingface-cli login` |
| 5 | `OSError: [Errno 28] No space left on device` | Disk full | `df -h` (Linux/Mac) / `Get-PSDrive` (Win) | Free 50+ GB; re-run with `--resume` |
| 6 | TTFT > 30 s on first run | Cold start (CUDA init, weight mapping) | inspect `state.json` `n_warmup` field | pass `--n-warmup 10` |
| 7 | RAM peak = 0.0 in CSV | psutil not tracking subprocess | `python -c "import psutil; print(psutil.Process().memory_info())"` | re-run with `--track-ram`; ensure psutil ≥ 6.0 |
| 8 | MMLU accuracy = 0.0 | Prompt template mismatch | run a single MMLU prompt manually | check `_format_mmlu_prompt()` in `quality_bench.py` |
| 9 | HumanEval always fails (0% pass@1) | Code extraction broken | inspect a single generation | fix `_extract_python_code()` in `quality_bench.py` |
| 10 | CSV empty after run | all configs errored | `Get-Content results\<run>\run.log \| jq 'select(.level==\"error\")'` | each error has structured `error_type` field; fix top error first |
| 11 | Report chart blank | all rows filtered out | `Select-String '\"data\":\[\]' docs\report.html` | check CSV has non-null `mmlu_acc` |
| 12 | Decision tree returns empty list | strategy too strict for this profile | try `--strategy max-quality` | relax thresholds in `decision_tree.py` |
| 13 | Plotly CDN blocked (offline) | network restrictions | `view-source:docs/report.html` | download `plotly.min.js` locally, edit `report.py` template |
| 14 | `pynvml.NVML_ERROR_DRIVER_NOT_LOADED` | NVIDIA driver outdated | `nvidia-smi` | update NVIDIA driver ≥ 525 |
| 15 | `llama-quantize: command not found` | llama.cpp not built | `which llama-quantize` | `git clone https://github.com/ggerganov/llama.cpp && make` |

---

## 3. Operational Notes

### 3.1 Before every commit

```powershell
pytest tests/ -v
ruff check .
ruff format --check .
mypy src/ --ignore-missing-imports
```

CI also runs these; failing locally means CI will fail too.

### 3.2 CSV inspection habits

After every `--max-examples 5` smoke test, **read the CSV with your eyes.** Numbers should be plausible:

- TTFT (time-to-first-token) should be 100 ms – 5 s for warm models.
- tokens/s should be > 5 for any modern model on its target hardware.
- RAM peak should be within 1.5× of expected model size (e.g. 14B Q4 ≈ 8–10 GB).

If something looks off, suspect: wrong format, wrong ctx_size, cold cache, or quantization bug.

### 3.3 Log analysis with `jq`

`run.log` is **line-delimited JSON** (one JSON object per line). Pipe through `jq`:

```powershell
# All errors
Get-Content results\<run>\run.log | jq 'select(.level=="error")'

# All OOM events
Get-Content results\<run>\run.log | jq 'select(.event=="oom")'

# Slow generations (> 30 s)
Get-Content results\<run>\run.log | jq 'select(.total_ms > 30000)'

# Per-model summary
Get-Content results\<run>\run.log | jq 'select(.event=="row_complete") | {model_id, ttft_ms_p50, tokens_per_s_p50}'
```

### 3.4 Disk usage

**Disk fill is the #1 cause of mid-run failures.** Pre-flight check runs at runner start:

- < 50 GB free → warning printed.
- < 10 GB free → runner aborts before any model download.

Monitor with:

```powershell
Get-PSDrive C | Select-Object Used, Free
```

### 3.5 Resume semantics

The `--resume` flag reads `output-dir/state.json` and skips any config already marked `status=ok` there. **Partial runs get retried** — by design, since transient OOM/timeout shouldn't poison the final dataset.

To force a fresh start: delete the `state.json` and `results.csv` files in the output dir, or pass a new `--output-dir`.

### 3.6 SIGINT (Ctrl-C) handling

The runner installs a SIGINT handler that:

1. Finishes the current generation cleanly (no half-written CSV row).
2. Writes `state.json` with all completed configs.
3. Exits with code 130.

Subsequent runs with `--resume` pick up from `state.json`.

### 3.7 Hardware profile privacy

`HardwareProfile` is stored in CSV without the hostname. If you customize `configs/profiles/<host>.yaml` with personal notes, those notes ARE stored in CSV — keep them generic.

### 3.8 When to add a model

Before adding a model to `configs/matrix.yaml`:

1. Check the model fits the **hardware filter** (run `--dry-run` and confirm it appears).
2. Check the **GatedRepoError** path: if the model is gated, document the `HF_TOKEN` requirement in the matrix comment.
3. Pick formats that are **actually published** on HuggingFace (don't invent `Q4_K_S` if the upstream only ships `Q4_K_M`).

See `CONTRIBUTING.md` for the full procedure.

### 3.9 Version pinning

Always pin dependency versions in `requirements.txt` for reproducibility. The `pyproject.toml` uses `>=` for minimum versions (so consumers can upgrade), but CI installs the pinned `requirements.txt`.
