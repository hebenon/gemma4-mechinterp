# Gemma 4 Mechanistic Interpretability

Two interlinked projects:

1. **TransformerLens support for Gemma 4** — extending Gemma 3 support (PR #1149) to cover Gemma 4's novel elements: Per-Layer Embeddings (PLE) and shared KV cache. Scope: E2B text-only, PLE as explicit hook points. Note: Gemma 3n is a related but distinct model; issue #953 targets Gemma 3n. We will open a new TL issue for Gemma 4.

2. **Mechinterp investigations** — using the new tooling to run interpretability research on Gemma 4. Initial investigation: multi-turn sycophancy dynamics (does accumulated social pressure degrade internal knowledge representations?). PLE diagnostic as a secondary contribution: what is PLE doing at each layer?

## Why Gemma 4

- Apache 2.0 license — fully open for research and reuse
- Novel architecture (PLE, shared KV) not yet supported by any mechinterp tooling
- Strong benchmarks across sizes — E2B fits on T4 GPU (free Colab tier)
- Gemma 3 already in TL (PR #1149) — our work extends it rather than starting from scratch
- The PLE diagnostic is itself a publishable finding about a novel architectural element

## Structure

```
gemma4-mechinterp/
├── notes/
│   ├── architecture.md       # Gemma 4 architecture map (module names, layer types)
│   ├── tl_adapter_notes.md   # TransformerLens implementation notes
│   └── ple_analysis.md       # PLE contribution findings (filled as experiments run)
├── notebooks/                # Investigation notebooks (Phase 2+)
├── PLAN.md                   # Detailed phased plan
└── README.md                 # This file
```

## Repositories

- **This repo**: Research notes, investigation notebooks, findings
- **TransformerLens fork**: Implementation work — fork `TransformerLensOrg/TransformerLens`

## Status

Phase 1: Environment + architecture mapping — **starting 2026-04-20**

See [PLAN.md](PLAN.md) for full phased plan.
