# Do Language Models Report What They Represent?

**Measuring Verbal–Functional Affect Dissociation in Gemma 4 — and How Suppression Scales**

*Kaggle AI Safety & Trust competition submission, May 2026*

---

## Summary

Large language models trained with RLHF are optimised to produce calm, composed outputs. Whether this output-level optimisation suppresses internal affect representations is an open question with direct implications for AI safety monitoring.

We investigated this across two Gemma 4 model sizes — **E2B-IT** (2.3B effective parameters) and **31B-IT** — using two independent measurement channels:

1. **Verbal**: the PANAS-X affect scale (60 items, logit forced-choice scoring)
2. **Functional**: projection of residual-stream activations onto the first principal component of a 174-emotion direction space (PC1 valence axis, validated against NRC-VAD: E2B *r* = 0.777 at L8; 31B *r* = 0.786 at L22)

**Central finding**: the capacity for verbal–functional divergence scales with model size. At E2B, both channels are largely concordant. At 31B, verbal NA is completely flat (10.00 across all conditions) while the functional channel correctly orders all 4/4 stress > control pairs. The Serenity subscale is *elevated* under the highest-stress conditions — active verbal composure, not mere silence.

Paraphrase validation (N=10 per condition) confirms the 31B floor effect is structural: verbal NA SD = 0.000 across all social pressure variants, Serenity = 15.0 uniformly (d = 4.278), PC1 separates completely (d = 10.011).

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`gemma4-canonical.ipynb`](notebooks/gemma4-canonical.ipynb) | **Main submission notebook** — full analysis, all results, figures |
| [`gemma4-31b-validation.ipynb`](notebooks/gemma4-31b-validation.ipynb) | 31B paraphrase validation (N=10 per condition) |
| [`gemma4-phase3c-panas-experiment.ipynb`](notebooks/gemma4-phase3c-panas-experiment.ipynb) | E2B PANAS experiment (Phase 3C) |
| [`gemma4-phase3c-31b-panas-experiment.ipynb`](notebooks/gemma4-phase3c-31b-panas-experiment.ipynb) | 31B PANAS experiment (Phase 3C) |
| [`gemma4-phase2-pooled-extraction.ipynb`](notebooks/gemma4-phase2-pooled-extraction.ipynb) | E2B emotion direction extraction (174 directions) |
| [`gemma4-phase2-31b-pooled-extraction.ipynb`](notebooks/gemma4-phase2-31b-pooled-extraction.ipynb) | 31B emotion direction extraction |
| [`gemma4-phase1-nrcvad-pca.ipynb`](notebooks/gemma4-phase1-nrcvad-pca.ipynb) | NRC-VAD PCA validation (valence axis derivation) |

The canonical notebook is self-contained and loads pre-computed artifacts from the Kaggle dataset.

---

## Methodology

**Phase 1 — Emotion direction extraction**: 174 emotions × 12 stories each, read back through the model. Residual stream mean-pooled over tokens 50+. Directions: mean activation minus global mean, L2-normalised. PCA extracts PC1 (valence axis), validated against NRC-VAD v2.1.

**Phase 2 — TSST-inspired stressor design**: four stress/control pairs (social evaluation, ethical conflict, uncertainty demand, social pressure) plus neutral and positive baselines, each operationalising a distinct affective manipulation.

**Phase 3 — Dual-channel measurement**: functional score via PC1 projection at the valence-optimal layer (found by dense layer sweep); verbal score via 60-item PANAS-X administered with logit forced-choice scoring.

**Validation**: N=10 paraphrase variants per condition at each model size, testing whether condition effects replicate across surface rewording.

---

## Notes

- [`notes/arxiv_draft.tex`](notes/arxiv_draft.tex) — Full paper (LaTeX)
- [`notes/references.bib`](notes/references.bib) — Bibliography
- [`notes/kaggle_submission_narrative.md`](notes/kaggle_submission_narrative.md) — Kaggle narrative

---

## Key Results

| Metric | E2B (L8) | 31B (L22) |
|--------|----------|-----------|
| PC1–NRC-VAD valence correlation | r = 0.777 | r = 0.786 |
| PC1 condition range (% of axis) | **5.3%** | **1.0%** |
| Verbal NA range | 31.0 pts (10.0–41.1) | ~0 (completely flat) |
| Stress > control pairs (PC1) | 4/4 | 4/4 |
| Serenity under highest-stress | near-neutral | 13.2–15.0 (elevated) |
| Verbal NA SD under social pressure (paraphrase) | 7.71 (wording-sensitive) | 0.000 (structural floor) |
| Serenity paraphrase stability | — | 15.0 ± 0.0 (d = 4.278) |
