# Gemma 4 Mechanistic Interpretability

Two interlinked projects:

1. **TransformerLens support for Gemma 4 (Gemma 3n)** — native adapter via TransformerBridge, contributing to [issue #953](https://github.com/TransformerLensOrg/TransformerLens/issues/953). Scope: E2B text-only, with Per-Layer Embeddings (PLE) as explicit hook points rather than bypassed.

2. **Mechinterp investigations** — using the new tooling to run interpretability research on Gemma 4. Initial investigation: multi-turn sycophancy dynamics (does accumulated social pressure degrade internal knowledge representations?). PLE diagnostic as a secondary contribution: what is PLE doing at each layer?

## Why Gemma 4

- Apache 2.0 license — fully open for research and reuse
- Novel architecture (PLE) not yet supported by any mechinterp tooling
- Strong benchmarks across sizes — E2B fits on T4 GPU (free Colab tier)
- TransformerLens #953 is open, unassigned, labeled complexity-high — a real gap, not duplicate work
- The PLE diagnostic is itself a publishable finding about a novel architectural element

## Structure

```
gemma4-mechinterp/
├── notes/
│   ├── architecture.md       # Gemma 4 architecture map (module names, layer types)
│   ├── tl_adapter_notes.md   # TransformerLens adapter implementation notes
│   └── ple_analysis.md       # PLE contribution findings (filled as experiments run)
├── notebooks/                # Investigation notebooks (Phase 2+)
├── PLAN.md                   # Detailed phased plan
└── README.md                 # This file
```

## Repositories

- **This repo**: Research notes, investigation notebooks, findings
- **TransformerLens fork**: Implementation work — fork `TransformerLensOrg/TransformerLens`, PR target is #953

## Status

Phase 1: Environment + architecture mapping — **starting 2026-04-20**

See [PLAN.md](PLAN.md) for full phased plan.
