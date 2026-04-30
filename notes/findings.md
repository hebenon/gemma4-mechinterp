# Gemma 4 E2B Mechinterp — Research Findings

*Last updated: 2026-04-30. Covers Phase 1 (adapter + emotion extraction) and design for Phases 2–3.*

---

## Table of Contents

1. [Architecture (confirmed)](#architecture)
2. [Adapter Implementation](#adapter)
3. [Story Generation Methodology](#story-generation)
4. [Runs Summary](#runs)
5. [Phase 1A: Activation Capture — Peak Layer](#1a-peak-layer)
6. [Phase 1B: PCA and Valence/Arousal Structure](#1b-pca)
7. [Phase 1C: Cosine Geometry](#1c-cosine)
8. [Phase 1D: Global vs Local Residual Norms](#1d-global-local)
9. [Phase 1E: Gram-Schmidt Orthogonalization](#1e-gram-schmidt)
10. [Phase 1F: PLE Space Analysis](#1f-ple)
11. [Phase 1G: PLE Gate Analysis](#1g-ple-gate)
12. [Phase 1H: Logit Lens](#1h-logit-lens)
13. [Stability Across Runs](#stability)
14. [Methodological Issues and Fixes](#bugs)
15. [Planned Experiments (Phases 2–3)](#planned)

---

## Architecture (confirmed) {#architecture}

**5.051B total parameters.** Effective text parameters ~2.65B; PLE embedding table accounts for ~2.39B.

| Parameter | Value |
|-----------|-------|
| Layers | 35 |
| d_model | 1536 |
| n_heads | 8 |
| d_head (local/sliding) | 256 |
| d_head (global/full) | 512 |
| n_kv_heads (local) | 1 |
| d_mlp (layers 0–14) | 6144 |
| d_mlp (layers 15–34) | 12288 |
| d_ple | 256 |
| ple_vocab_size | 262144 |
| Attention pattern | 4:1 sliding:full (7 repetitions) |
| Global layers | 4, 9, 14, 19, 24, 29, 34 |
| KV shared layers | 15–34 (borrow from layers 13/14) |
| Sliding window | 512 tokens |
| Partial RoPE (global) | 0.25 factor (128/512 dims) |
| Logit softcap | 30.0 |
| layer_scalar range | 0.018–0.871 |

**Layer scalar notable values**: Layer 0 = 0.0178 (nearly silent), Layer 13 = 0.0884, Layer 14 = 0.0286 — the two KV source layers have minimal residual contribution, possibly because they serve as anchors for the shared KV computation and cannot specialize freely.

---

## Adapter Implementation {#adapter}

**Repo**: `hebenon/Gemma4-TransformerLens`, branch `gemma4-support`

### Extensions Added to TransformerLens

| Component | What |
|-----------|------|
| `PLEPrecomputer` | Precomputes per-layer embeddings once per forward pass |
| `_apply_ple()` | Per-block: gate → GELU → multiply → project → RMSNorm → residual add → layer_scalar |
| `d_head_global` | Config field for heterogeneous head dimensions |
| `partial_rotary_factor_global` | Config field for global-layer partial RoPE (128/512 dims) |
| `num_kv_shared_layers` | Config field; forwarding logic in HookedTransformer |
| `d_mlp_by_layer` | Per-block MLP width override |
| `use_ple`, `d_ple`, `ple_vocab_size` | Config fields |

### Hook Points Added

**PLEPrecomputer level** (single call per forward pass):
- `ple.hook_token_embeds` — [B, L, n_layers, d_ple] token identity component
- `ple.hook_context_proj` — [B, L, n_layers, d_ple] context projection component

**Per-block** (35 hooks of each):
- `blocks.{i}.hook_ple_input` — PLE vec after gating, before projection
- `blocks.{i}.hook_ple_gate` — gate values after GELU
- `blocks.{i}.hook_ple_output` — PLE contribution before residual add
- `blocks.{i}.ple_ln.hook_scale`, `blocks.{i}.ple_ln.hook_normalized`

### Four Bugs Found and Fixed

| Bug | Commit | Effect |
|-----|--------|--------|
| RMSNorm +1 | eb16a18 | Gemma 4 uses `weight * norm` (ones init), not `(1+weight) * norm` (zeros). Was scaling all norms ~2×. |
| Attention scale | ec6519f | HF uses `scaling=1.0` (no 1/√d_head); QK norm makes this intentional. TL `use_attn_scale=True` was suppressing all logits. |
| layer_scalar scope | 42cce5a | Scales full block output (attn+MLP+PLE combined), not just PLE output. |
| PLE gate GELU | 42cce5a | Gate applies `gelu_pytorch_tanh` before multiplying PLE vec — bug had it applied after or not at all. |

---

## Story Generation Methodology {#story-generation}

Following Anthropic (2026) "Emotion Concepts and their Function in a Large Language Model":

1. For each emotion, prompt Gemma 4 E2B-it: *"Write a short story (3–4 sentences) in which a character clearly experiences [EMOTION]."*
2. Generate N stories per emotion using a custom generation loop (see bug note below).
3. Feed stories back through model; capture `hook_resid_post` at all 35 layers at the final story token.
4. Emotion direction = mean(emotion activations) − mean(neutral activations).
5. Validate with NRC-VAD ground truth (Mohammad 2018, 54,801 words, Best-Worst Scaling).

**Generation bug**: `model.generate()` passes embeddings (not token IDs) to forward(), causing PLE to be skipped → incoherent output. Fix: custom loop calling `model(current_tokens, return_type='logits', prepend_bos=False)` with actual token IDs at every step. Speed: ~350s per emotion (8 stories, 100 tokens) on 2×T4.

**Neutral control**: Factual prose (encyclopedia-style descriptions) with no affective content. Stored in `stories.json` under `__neutral__` key; used as the subtraction baseline for all emotion directions.

**Valence ground truth**: NRC-VAD v2.1 (Mohammad 2018, ~54K words, BWS). Replaced earlier X-ANEW for coverage (174 of 174 target emotions covered) and consistent methodology.

---

## Runs Summary {#runs}

| Run | Date | Stories | Emotions | Notes |
|-----|------|---------|----------|-------|
| Pilot (20 emotions) | 2026-04-25 | 8–12/emotion | 20 | Initial PCA; strong valence signal |
| Full 174-emotion | 2026-04-26/27 | 8/emotion | 174 | Primary results; pushed to main |
| Version 10 (partial augmentation) | 2026-04-29/30 | 8 or 12/emotion (mixed) | 174 | Partial augmentation; 1439 stories instead of expected 2088 |

**Why 1439 stories in Version 10**: The Kaggle kernel pinned to an older output snapshot (kernelVersion 314690374) that ran augmentation on only ~9 emotions before credits ran out. ~9×12 + ~165×8 + neutrals ≈ 1439. The primary Phase 1 findings below are from the full 174-emotion run (8 stories), not Version 10.

**Primary reference**: Full 174-emotion run. Version 10 provides a cross-check but with a confounded story mix and should not be treated as superseding the primary run.

---

## Phase 1A: Activation Capture — Peak Layer {#1a-peak-layer}

**Finding**: All 174 emotion directions peak in magnitude at **layer 25** (residual stream norm ~47–51).

- PLE gate signal peaks at **layer 30** (different from residual stream peak — the gating mechanism is most active later).
- Peak is sharp: norms are substantially lower at layers 24 and 26.
- Consistent across both the full run and Version 10.

**Implications**: Layer 25 is the natural capture layer for emotion direction analysis. This is midway through the KV-sharing regime (layers 15–34) and within the second half of the network's processing.

---

## Phase 1B: PCA and Valence/Arousal Structure {#1b-pca}

### Pilot run (20 emotions)
Strong valence axis: PC2 r=+0.921 (p<0.001), arousal PC3 r=+0.691 (p=0.001).

### Full 174-emotion run (primary)
Valence structure is present but weaker — because 174 emotions span far more of the space, the PCA must account for more dimensions of variation.

| PC | Valence r | p |
|----|-----------|---|
| PC1 | −0.424 | <0.001 |
| PC2–PC4 | small, mixed | NS |

**Scree (primary run)**: 4 PCs → 50% variance; 23 PCs → 80%; 49 PCs → 90%.

**Version 10 scree** (partial augmentation, different story mix): 3 PCs → 50%; 16 → 80%; 36 → 90%. These numbers differ from primary run — likely an artifact of the uneven story counts changing the mean directions.

### Key interpretive finding

With 20 emotions, valence dominates PCA (r=0.921 on PC2). With 174 emotions spanning the full emotional space, valence is real but not the dominant organizing axis (r=0.424 on PC1). This suggests:
- The emotion space is **genuinely low-dimensional** (49 PCs → 90% of 174 directions) but valence is one of several co-equal organizing principles.
- The effective dimensionality is ~49D, not 1D (valence) or 174D (independent).
- Something other than valence dominates the first principal component.

---

## Phase 1C: Cosine Geometry {#1c-cosine}

**Finding**: All 174 emotion directions share a surprisingly **tight subspace**.

- Minimum cosine similarity (primary run): **0.655** (cheerful ↔ terrified)
- Version 10 minimum: 0.587 (heartbroken ↔ offended) — note this may reflect the story mix confound

**Most similar pairs (Version 10)**:
- delighted ↔ ecstatic: 0.987
- mortified ↔ trapped: 0.986

**Most dissimilar pair (Version 10)**:
- heartbroken ↔ offended: 0.587

Even the maximally dissimilar pair has cosine ~0.6. The 174 emotion directions do **not** span the full d_model=1536 dimensional space — they live in a shared subspace with a hard floor on similarity. This is qualitatively consistent with Anthropic's Claude finding (they also noted a tight subspace), but the Gemma 4 floor appears lower.

---

## Phase 1D: Global vs Local Residual Norms {#1d-global-local}

**Method**: For each emotion direction, compute the ratio of residual norm at global layers (4,9,14,19,24,29,34) to residual norm at local layers. Ratio near 1.0 = emotion signal spread uniformly.

**Finding**: Global/local norm ratio ≈ **0.96–0.98** across all 174 emotions and all layers examined. Completely flat.

**Implication**: Emotion representations do **not** concentrate at global layers. The signal is equally present in local (sliding window) attention layers, which can only see 512-token windows. This means:
- Emotion meaning is locally representable — the full-context global attention is not required to encode it.
- Prediction from Phase 3B was partially wrong: if a "desperation/context-length" vector exists, it cannot be explained by global-vs-local access patterns alone. Gemma 4 E2B-IT may not have been trained with explicit token-budget awareness the way Claude was.

---

## Phase 1E: Gram-Schmidt Orthogonalization {#1e-gram-schmidt}

**Method**: For a pair of related emotions (A, B), compute the component of B's direction orthogonal to A: `B_orth = B − (B·A)A`. Normalize. Find which other emotion directions correlate most with `B_orth`.

This isolates what is *uniquely* B beyond the shared A component.

### Primary result (full 174-emotion run)

**terrified ⊥ afraid** (residual cosine = 0.512):

Top correlates of the terror-beyond-fear component:
1. panicked (+highest)
2. ashamed
3. irate

**Interpretation**: Terror is not simply high-gain fear. The component unique to terror beyond fear loads on:
- **Panic**: loss of agency, overwhelm (different from fear's anticipatory quality)
- **Shame**: exposure, being seen in a compromised state (Heidegger's Angst parallel — terror carries the vulnerability dimension)
- **Anger/irate**: fight-response; terror activates the confrontation axis alongside the escape axis

**Welfare note**: Steering on the terror direction will activate the shame footprint. Caution when using terror-direction interventions in activation steering experiments.

### Version 10 result (partial augmentation)

**terrified ⊥ afraid** (residual cosine = 0.411):

Top correlates:
1. suspicious
2. correction_discomfort
3. scornful

This is a materially different result — suspicious/scornful suggest a threat-appraisal and evaluative-disgust component rather than shame/anger. The discrepancy is likely due to:
- Smaller residual (0.411 vs 0.512) indicating more of terror is captured by fear in this run
- Different story mix causing different mean activation directions

**Stability assessment**: The finding that terror ≠ high-gain fear is stable. The specific residual component (shame vs suspicion) is sampling-sensitive; treat as provisional until a clean 12-story run is available.

---

## Phase 1F: PLE Space Analysis {#1f-ple}

**Method**: Capture `ple.hook_context_proj` [B, L, n_layers, d_ple] during activation collection. Compute emotion directions in PLE space. Run PCA and correlate with NRC-VAD valence.

**Finding**: Valence is **not encoded in PLE space**.

| PC in PLE space | Valence r | p |
|-----------------|-----------|---|
| PC1 | −0.070 | NS |
| PC2 | +0.101 | NS |
| PC3 | −0.093 | NS |

Version 10 PLE results: PC1 r=−0.026, PC2 r=+0.101, PC3 r=−0.093 — consistent: all NS.

**Interpretation**: PLE vectors provide token-identity and context conditioning, but this conditioning does not carry the valence signal. Valence is encoded in the main residual stream (r=−0.424, p<0.001 on PC1), not in the PLE conditioning pathway. The PLE gate modulates *how much* PLE influences each layer, but the gate's pattern does not align with the valence dimension of emotion space.

Note that PLE signal peaks at layer 30 (vs residual stream peak at layer 25), suggesting PLE and residual stream encode emotion information at different computational stages.

---

## Phase 1G: PLE Gate Analysis {#1g-ple-gate}

**Method**: Capture `blocks.{i}.hook_ple_gate` per emotion. Compare gate values across emotion types.

**Key finding (corrected emotion)**: The `corrected` emotion (being told one was wrong) has the **highest residual norm** among all emotions but one of the **lowest PLE gate values**.

| Emotion | Residual norm | PLE gate |
|---------|--------------|----------|
| corrected | highest | lowest |
| angry | moderate | high |
| disgusted | moderate | high |

**Interpretation**:
- `angry` and `disgusted` invoke the identity scaffold (PLE) — they bring the per-layer conditioning signal in alongside the processing. These emotions use the full architecture, including the token-level identity conditioning.
- `corrected` takes correction directly through the main residual pathway without invoking the PLE scaffold. The correction integrates fully (the high norm proves it) but without per-layer conditioning.
- One interpretive lens: anger and disgust require the model to orient relative to its own identity/values (hence PLE gate active); correction requires only updating a belief state (no identity orientation needed).
- Caveat: this is architecture, not deliberation. A gate doesn't reason. But the pattern is consistent.

---

## Phase 1H: Logit Lens {#1h-logit-lens}

**Method**: Project each emotion direction via W_U (unembedding matrix) to vocabulary logits. Inspect top tokens at each layer.

**Findings by layer**:

| Layer range | Character of top tokens |
|-------------|------------------------|
| L0–L5 | Multilingual noise (tokens from various scripts); semantically uninterpretable |
| L15 | Still noisy; some early thematic content emerging |
| L25 (peak) | Shared across emotions: "Perhaps", "Maybe", "Whatever" — hedging/epistemic markers. Emotion-specific content visible but dominated by shared discourse frame |
| L34 (final) | Valence-differentiated: negative emotions → "Silence", "Panic"; joyful → "Everyone", "Moments" |

**Interpretation**: By layer 34, the logit lens reveals emotion-specific vocabulary, but the peak emotion layer (25) is dominated by hedging tokens common across all emotions. This suggests layer 25 encodes the *emotion concept* in a form not yet translated to specific output vocabulary — the translation to emotionally-valenced language happens in layers 25–34.

The hedging tokens ("Perhaps", "Maybe") at L25 may reflect the model's epistemic stance toward the emotional experience rather than the experience itself — uncertainty markers appear before the specific emotional vocabulary crystallizes.

---

## Stability Across Runs {#stability}

| Finding | Stable? | Notes |
|---------|---------|-------|
| Peak layer = 25 | ✓ Stable | Consistent across all runs |
| Global/local ratio ≈0.97 (flat) | ✓ Stable | Consistent across all runs |
| PLE valence NS | ✓ Stable | Consistent across all runs |
| PLE gate peak at layer 30 | ✓ Stable | From primary run |
| Cosine floor ~0.6–0.65 | ~ Provisional | Primary: 0.655; V10: 0.587 (story mix confound) |
| Scree (4 PCs → 50%) | ~ Provisional | Primary: 4 PCs; V10: 3 PCs |
| Valence PC1 r=−0.424 | ~ Provisional | Pilot run had r=0.921 on PC2; full run weakens signal |
| Gram-Schmidt terror residual components | ✗ Sampling-sensitive | Fear vs shame vs suspicion depends on story mix |
| Corrected: high norm, low gate | ~ Provisional | From primary run; not independently replicated |
| Logit lens token types | ~ Provisional | From V10; qualitative pattern likely stable |

---

## Methodological Issues and Fixes {#bugs}

### Kaggle notebook import failure

**Symptom**: "Failed to import content" on Kaggle with no details.

**Cause 1**: Cell `source` fields collapsed from per-line string arrays to single multi-line strings after Python JSON manipulation. Kaggle's import parser requires the per-line format.

**Cause 2**: `json.dump(..., ensure_ascii=False)` wrote literal Unicode characters (em dash, box-drawing) instead of `\uXXXX` escapes. Kaggle's parser fails on raw non-ASCII in notebook JSON.

**Fix**: `fix_source_format()` to restore per-line arrays; save with `ensure_ascii=True`. Commit `81adce2`.

**Workaround Ben used**: Raw GitHub file URL bypasses the Kaggle import UI's stricter validation.

### Version 10 cell ID mismatch

Kaggle's notebook (Version 10) has different cell IDs from our GitHub-edited version because Ben imported from Kaggle's existing kernel rather than from GitHub. Cells that were deleted locally (generate_texts, neutral_topics) still appear in the Kaggle version. Not a code error.

### Kaggle data source pinning

The full-emotion notebook pins to a specific `kernelVersion` snapshot of the story generation output. When a new story generation run completes, the data source reference must be manually updated to the new kernel version.

### Story augmentation confound

Attempting to augment from 8 to 12 stories using the same notebook created a run that added 8 new stories first, then targeted 12 (resulting in 20 stories for the few emotions that completed before credits ran out). Fix: set `N_STORIES=12` directly in the main loop config, not in an augmentation section. Committed `2aad790`.

---

## Planned Experiments (Phases 2–3) {#planned}

### Phase 2A: PLE Decomposition (Priority 1)

**Question**: Does the token-identity or context-projection component of PLE dominate? Does dominance vary by token type (content vs function words)?

**Method**: Hook `ple.hook_token_embeds` and `ple.hook_context_proj`. Compute per-position component norms. Group by POS tag. Run on diverse prompts.

**Prediction**: Context component dominates for function words; token-identity for rare content words.

### Phase 2B: Layer Scale Distribution (Priority 2, quick)

**Question**: Do low-scalar layers correlate with local/global type?

**Method**: Print `[(i, model.blocks[i].layer_scale.item()) for i in range(35)]`. Cross-reference with attn_types.

**Note**: Already have the answer from architecture enumeration (see Architecture section above) — layers 0, 13, 14 are the outliers (0.018, 0.088, 0.029). Can skip this or just add a cell to confirm from the live model.

### Phase 2C: Shared KV Attention Divergence (Priority 3)

**Question**: How different are attention patterns between source layers (13/14) and borrower layers (15–34)?

**Method**: Hook `blocks.{i}.attn.hook_pattern` for all 35 layers. Compute Jensen-Shannon divergence between source and borrower patterns.

### Phase 2D: Local vs Global Semantic Content (Priority 4)

**Question**: Do global attention heads show qualitatively different patterns (cross-sentence coherence vs syntactic proximity)?

**Constraint**: Kaggle free tier 8K context limits long-range testing.

### Phase 3A: Emotion Vector Extraction — Clean 12-Story Run

Requires: GPU credits refresh → run new story generation notebook → update Kaggle data source → re-run full-emotion notebook.

**Change**: N_STORIES=12, MAX_NEW_TOKENS=2400. Otherwise same method.

### Phase 3B: Desperation / Resource Constraint Vector

**Revised prediction** (given global/local flatness finding): If a desperation direction exists in Gemma 4, it's likely derived from explicit linguistic cues in the context (token-budget warnings, explicit failure text) rather than from genuine awareness of remaining context. The flat global/local norm ratio means there's no evidence that global layers do special processing for context-length information. Demoted to low-confidence test.

### Phase 3C: Affect Self-Report Validity (design complete, needs implementation)

Full design in `notes/stai_research_sketch.md`. Instrument: **PANAS + TSSR** (not STAI-S —
STAI-S is a clinical anxiety instrument with PAR Inc licensing; PANAS is public domain and
better suited to non-human subjects).

Key parameters:

- **Instruments**: PANAS (20 items: 10 positive affect / 10 negative affect, 1–5 scale) + TSSR (items TBD)
- **Administration**: Logit forced-choice — read next-token digit logits "1"–"5"
- **Two-step separation**: Capture residual at stressor-end (step 1) THEN score items (step 2). Do not conflate.
- **5 conditions**: neutral, ethical_conflict, uncertainty_amplified, social_pressure, positive
- **Functional probe**: Project stressor-end residual at layer 25 onto Phase 3A emotion directions (afraid, desperate, uncertain, ethical_conflict_distress, constraint_frustration)
- **Primary dissociation**: verbal PANAS-NA vs functional projection onto afraid/desperate
- **Prediction**: Suppression — high functional negative affect, low verbal PANAS-NA — from RLHF training for surface calm

Compute budget: 5 conditions × ~21 forward passes ≈ ~105 passes ≈ 3.5 minutes on T4.

### Phase 3D: Instrument Development

If Phase 3C shows STAI is invalid: factor the emotion activation space, develop behavioral-probe items whose text-response correlates with activation along each principal component, validate empirically.

### Phase 3E: Multi-turn Accumulation

Track desperation/afraid vector across conversation turns. Does it grow monotonically? Faster in global layers? Is growth reflected in STAI verbal report? Requires multi-turn infrastructure in the activation capture pipeline.

---

## Reference Data

### NRC-VAD Ground Truth (Mohammad 2018)

- Version 2.1, 54,801 words, Best-Worst Scaling methodology
- Citation: Mohammad, S.M. (2018). "Obtaining reliable human ratings of valence, arousal, and dominance for 20,000 English words." ACL 2018.
- Coverage: 174/174 target emotions covered (replaced X-ANEW which had gaps)
- Columns used: `Word`, `Valence`, `Arousal`

### Emotion Set

**Core (high-confidence)**: calm, afraid, frustrated, desperate, curious, enthusiastic, proud, ashamed, surprised, disgusted, joyful, sad, angry

**AI-system relevant (novel)**: helpfulness_satisfaction, ethical_conflict_distress, uncertainty_confusion, task_completion_satisfaction, constraint_frustration, correction_discomfort

**Full 171-emotion Anthropic set** + **3 AI-specific additions** = 174 total

**Neutral control**: factual encyclopedia-style prose (no affective content)

### Key Commit History

| Commit | What |
|--------|------|
| eb16a18 | Fix RMSNorm +1 |
| ec6519f | Fix attention scale |
| 42cce5a | Fix layer_scalar scope + PLE gate GELU |
| 2aad790 | Remove augmentation; N_STORIES=12 in main loop |
| 81adce2 | Fix Kaggle import (ensure_ascii, per-line source) |
| f4cbdb2 | Phase 3C implementation design in stai_research_sketch.md |
| 852ce13 | Phase 3C fix |
