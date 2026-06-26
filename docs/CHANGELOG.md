# Changelog

All notable changes to edge-llm-benchmark are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial scaffold from `plan.md` + `prompt.md`.
- Hardware profiler (cross-platform RAM / CPU / GPU detection).
- Model fetcher with HuggingFace gated-repo support.
- Quantizer (llama.cpp / AWQ / GPTQ / MLX dispatch).
- Inference engine abstraction with Ollama + llama.cpp backends.
- Performance benchmark (TTFT, tokens/s, RAM, prefill cost).
- Quality benchmark (MMLU, IFEval, HumanEval, perplexity).
- Aggregator with atomic CSV writes.
- CLI runner with dry-run, resume, SIGINT handling, JSON logs.
- Report generator (5 Plotly charts, sortable HTML table).
- Decision tree (5 strategies: max-quality, max-speed, min-ram, balanced, code).
- 3 hardware profiles (macbook-m3-36gb, rtx-4090-24gb, threadripper-128gb).
- Unit tests for all core modules.
- CI workflows for Linux CPU, macOS ARM, GitHub Pages deploy.
