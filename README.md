# edge-llm-benchmark

> **Quality × Speed × Memory trade-offs for open-weight LLMs — measured on *your* hardware.**

[![CI](https://github.com/matheus/edge-llm-benchmark/actions/workflows/bench-linux-cpu.yml/badge.svg)](.github/workflows/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)

Stop guessing whether `Q4_K_M` is fast enough on your MacBook. Stop trusting
benchmarks published on H100s that don't translate to consumer hardware.
`edge-llm-benchmark` measures *your* deployment reality — RAM, TTFT, throughput
*and* quality (MMLU, IFEval, HumanEval, perplexity) — and produces a
self-contained HTML report you can open offline.

**Zero paid-API cost.** Runs entirely on local inference (Ollama or llama.cpp).

---

## Why this exists

The local-LLM ecosystem has a measurement crisis:

- ~600 base models × ~10 quantization formats = thousands of (model, format) combinations.
- Vendors publish numbers measured on H100s that don't translate to a 32 GB MacBook.
- Community knowledge is scattered across Reddit threads, Twitter screenshots, and Discord.
- Choosing *"the right model for my hardware"* requires weeks of trial-and-error.

`edge-llm-benchmark` is the first tool with **first-class concern for deployment reality**
(RAM, TTFT, throughput) **combined with systematic quality measurement**,
on consumer hardware, with zero API cost and reproducible CSV results.

---

## Quickstart

```bash
git clone https://github.com/matheus/edge-llm-benchmark
cd edge-llm-benchmark
bash scripts/setup.sh                     # one-time, < 5 min

# Dry-run: see what would be benchmarked on your machine
python -m edge_llm_bench.runner \
  --profile configs/profiles/macbook-m3-36gb.yaml \
  --dry-run

# Real run (a few hours; resumable with --resume)
python -m edge_llm_bench.runner \
  --profile configs/profiles/macbook-m3-36gb.yaml \
  --output-dir results/2026-06-23

# Generate the HTML report
python -m edge_llm_bench.report --latest

# Get recommendations for your hardware
python -m edge_llm_bench.decision_tree \
  --profile configs/profiles/macbook-m3-36gb.yaml \
  --strategy balanced
```

See [`docs/runbook.md`](docs/runbook.md) for the full workflow, troubleshooting
matrix, and operational notes.

---

## What it measures

| Dimension       | Metrics                                                       |
|-----------------|---------------------------------------------------------------|
| **Performance** | TTFT (p50, p95), tokens/s (p50, p95), prefill tokens/s, RAM peak, VRAM peak |
| **Quality**     | MMLU (200 q), IFEval strict (100 p), HumanEval pass@1 (30 q), WikiText-103 perplexity |
| **Resource**    | Disk size, RAM peak, GPU utilization (NVIDIA)                 |

Output: a single self-contained `docs/report.html` with **5 interactive Plotly charts**:

1. **Quality vs Time-to-First-Token** (bubble size = RAM).
2. **Quality vs Memory** (color = quantization format).
3. **Per-model metrics bar chart** by quantization format.
4. **Pareto frontier** on (MMLU, tokens/s).
5. **Quality degradation curve** across quantization levels.

---

## Decision tree

Given a hardware profile, the decision tree returns ranked recommendations
across 5 strategies:

| Strategy       | What it picks                                       |
|----------------|-----------------------------------------------------|
| `max-quality`  | Highest MMLU in runnable set                        |
| `max-speed`    | Fastest tok/s with TTFT < 500 ms                    |
| `min-ram`      | Lowest RAM that still scores MMLU ≥ 0.70            |
| `balanced`     | Pareto-optimal on (MMLU, tokens/s)                  |
| `code`         | Best HumanEval pass@1                               |

Example:

```text
Recommendations (strategy=balanced, profile=macbook-m3-pro-36gb):

  #1  Qwen/Qwen3-32B-Instruct  [Q5_K_M]
      Pareto-optimal on (MMLU=0.78, tok/s=24.3); RAM=22.1 GB.
  #2  mistralai/Mistral-Small-24B-Instruct-2501  [Q5_K_M] ⭐ Pareto
      Pareto-optimal on (MMLU=0.75, tok/s=26.8); RAM=20.4 GB.
  #3  microsoft/Phi-4-14B-Instruct  [Q8_0] ⭐ Pareto
      Pareto-optimal on (MMLU=0.71, tok/s=42.1); RAM=16.8 GB.
```

---

## Architecture

```
Layer 4 — Reporting         : report.py · decision_tree.py · GitHub Pages
Layer 3 — Orchestration     : runner.py · CLI · resume · SIGINT
Layer 2 — Benchmarks        : perf_bench.py · quality_bench.py
Layer 1 — Foundation        : hardware_profiler.py · model_fetcher.py ·
                               quantizer.py · inference/ (ollama, llama-cpp)
Data: configs/ + datasets/ → results/ → docs/
```

Each layer is independently testable. The inference engine is **pluggable**:
Ollama (default, best DX) and llama.cpp (more control). All outputs are static
files — no database, no server. Trivial to publish on GitHub Pages and to
version-control.

See [`plan.md`](plan.md) for the full design.

---

## How to add a model

Open `configs/matrix.yaml` and add:

```yaml
  - id: org/new-model-name
    family: my-family
    formats: [Q4_K_M, Q5_K_M]
    requires_ram_gb: 24
```

Then:

```bash
python -m edge_llm_bench.runner \
  --profile configs/profiles/<your-host>.yaml \
  --config configs/matrix.yaml \
  --dry-run
```

If the model appears in the resolved matrix, you're set. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for full details and
`docs/runbook.md` for the troubleshooting matrix.

---

## Project layout

```
edge-llm-benchmark/
├── configs/
│   ├── matrix.yaml                # (model × format) × backend matrix
│   └── profiles/                  # hardware profiles (yaml)
├── datasets/                      # versioned eval subsets (committed)
├── src/edge_llm_bench/            # all Python source
├── tests/                         # pytest suite
├── scripts/                       # setup.sh / setup.ps1 / dataset downloader
├── docs/                          # runbook, changelog, report output
├── .github/workflows/             # CI: Linux CPU, macOS ARM, Pages deploy
├── results/                       # CSV outputs (gitignored)
└── pyproject.toml
```

---

## Contributing

We welcome PRs for new models, new eval suites, new backends, and bug fixes.
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the workflow and code style.

---

## Acknowledgments

- **[Ollama](https://ollama.com)** — local inference daemon.
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — GGUF + quantization.
- **[lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)** — IFEval + HumanEval.
- **Plotly** — interactive HTML charts.

## License

[MIT](LICENSE)
