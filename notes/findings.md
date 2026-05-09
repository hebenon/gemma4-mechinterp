# Gemma 4 E2B Mechinterp — Research Findings

*Last updated: 2026-05-09. Covers Phase 1 (adapter + emotion extraction), Phase 2 (token mean-pooling), Phase 3C (PANAS-X suppression experiment — E2B V15 + 31B V7 with corrected methodology), and cross-scale comparison (E2B vs 31B).*

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
13. [Phase 1I: Emotional Range and Tri-polar Analysis](#1i-emotional-range)
14. [Stability Across Runs](#stability)
15. [Methodological Issues and Fixes](#bugs)
16. [Phase 3C: PANAS-X — E2B V14 Results (top-k, superseded)](#3c-results)
17. [Phase 3C V15 Methodology Fixes](#3c-fixes)
18. [Phase 3C: E2B V15 Results (mean pooling, canonical)](#3c-e2b-v15)
19. [Phase 1 Dense Sweep: 31B](#31b-phase1)
20. [Phase 3C: 31B V7 Results](#3c-31b)
21. [Scale Comparison: E2B vs 31B](#scale-comparison)
22. [Planned Experiments (Phases 2–3)](#planned)

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
| Full 174-emotion, 8-story | 2026-04-26/27 | 8/emotion | 174 | Earlier run; numbers in memory file |
| Full 174-emotion, 12-story (Version 10) | 2026-04-29/30 | 12/emotion | 174 | Complete run; 2098 total texts |

**The "1439 stories" was a red herring**: A diagnostic print statement in the Kaggle notebook read
`sum(len(v) for v in all_stories['emotions'].keys())`, which sums the *character lengths of emotion
name strings*, not the story counts. 1439 ≈ total characters in 174 emotion names. The notebook was
processing all 2098 texts (174×12 + 10 neutrals) throughout.

**Primary reference for Phase 1 findings**: Version 10 (full 12-story run). Numbers in the memory
file (scree: 4 PCs→50%, valence r=−0.424) are from the earlier 8-story run and are superseded.

---

## Phase 1A: Activation Capture — Peak Layer {#1a-peak-layer}

**Finding**: All 174 emotion directions peak in magnitude at **layer 25** (residual stream norm ~47–51).

- PLE gate signal peaks at **layer 30** (different from residual stream peak — the gating mechanism is most active later).
- Peak is sharp: norms are substantially lower at layers 24 and 26.
- Consistent across 8-story and 12-story runs.

**Implications**: Layer 25 is the natural capture layer for emotion direction analysis. This is midway through the KV-sharing regime (layers 15–34) and within the second half of the network's processing.

---

## Phase 1B: PCA and Valence/Arousal Structure {#1b-pca}

*Analysis notebook*: `notebooks/gemma4-phase1-nrcvad-pca.ipynb`. Correlates all PCs against
all three NRC-VAD dimensions (valence, arousal, dominance), not just valence.

### Scree (12-story canonical run)

3 PCs → 52.4% variance; 10 PCs → 73.7%; 16 PCs → 80.6%; 36 PCs → 90.3%.

### Per-PC correlation with NRC-VAD (top 10 PCs)

| PC | var% | Valence r | Arousal r | Dominance r |
|----|------|-----------|-----------|-------------|
| 1 | 37.4 | −0.403*** | +0.206**  | −0.328***   |
| 2 |  8.8 | +0.378*** | +0.029    | +0.305***   |
| 3 |  6.3 | +0.232**  | −0.281*** | −0.018      |
| 4 |  5.2 | +0.457*** | +0.134    | +0.422***   |
| 5 |  4.0 | +0.084    | +0.088    | +0.302***   |
| 6 |  3.3 | +0.012    | −0.475*** | −0.175*     |
| 7 |  2.5 | +0.223**  | +0.078    | +0.114      |
| 8 |  2.4 | −0.082    | −0.083    | −0.224**    |
| 9 |  2.3 | +0.185*   | −0.228**  | −0.075      |
| 10 | 1.7 | −0.116    | −0.122    | −0.098      |

### Cumulative R² (how much of each VAD dimension is explained by top-k PCs)

| k  | Valence R² | Arousal R² | Dominance R² |
|----|------------|------------|--------------|
| 1  | 0.163      | 0.042      | 0.107        |
| 4  | 0.568      | 0.140      | 0.379        |
| 8  | 0.632      | 0.386      | 0.563        |
| 20 | 0.734      | 0.583      | 0.620        |

### What each PC actually represents (top emotion loadings)

| PC | Best VAD label | + pole | − pole |
|----|---------------|--------|--------|
| PC1 (37.4%) | Mixed (V, D, A) | surprised, rattled, hysterical, bored, vigilant | optimistic, grateful, cheerful, smug, melancholy |
| PC2 (8.8%) | Valence/Dominance | self_confident, pleased, triumphant, euphoric, elated | panicked, unsettled, terrified, trapped, alert |
| PC3 (6.3%) | Arousal (neg) | heartbroken, loving, sentimental, relaxed, safe | offended, frustrated, outraged, stubborn, furious |
| PC4 (5.2%) | Valence/Dominance | fulfilled, enthusiastic, heartbroken, joyful, vibrant | depressed, melancholy, tormented, frightened |
| PC6 (3.3%) | **Arousal (clearest)** | sleepy, indifferent, resigned, bored | surprised, frightened, scared, desperate, puzzled |

### Key interpretive findings

**PC1 is not valence.** It accounts for 37.4% of all variance — more than the next five PCs combined — and it correlates with valence, dominance, AND arousal simultaneously. The emotion loadings don't resolve to a clean psychological category: the positive end mixes high-arousal (hysterical, rattled) with low-arousal (bored, dispirited), and the negative end mixes clear positives (optimistic, grateful) with negatives (melancholy). The dominant organizing axis of Gemma 4's emotion space is an unnamed mixed construct.

**Arousal has the clearest dedicated axis: PC6** (r=−0.475***, only 3.3% of variance). High-arousal emotions (frightened, scared, desperate, surprised) are at one pole; low-arousal (sleepy, indifferent, resigned, bored) at the other. Arousal is concentrated but minor.

**Valence is real but distributed.** Cumulative R²=0.734 with 20 PCs makes valence the best-explained NRC-VAD dimension, but no single PC exceeds r=0.46. There is no clean "valence axis" — the valence signal is spread across PC1, PC2, PC4, and beyond. This is fundamentally different from Anthropic's Claude finding (r=0.81 on a single axis).

**Pilot result (20 emotions, valence PC2 r=0.921) explained.** With 20 hand-selected emotions spanning the valence extremes, PCA is essentially forced to find a valence axis because that's what varies most across those specific 20 points. With 174 emotions including dense near-synonym clusters (ecstatic/elated/euphoric, terrified/horrified/frightened), PCA must account for within-cluster variation too, diluting the valence signal. The pilot result reflects the selection, not the model's organization.

**Matched 171/174 emotions** to NRC-VAD. Missing: energized, insulted, stimulated (no exact or first-word match in NRC-VAD v2.1).

---

## Phase 1C: Cosine Geometry {#1c-cosine}

**Finding**: All 174 emotion directions share a surprisingly **tight subspace**.

**Canonical (12-story)**:
- Most similar: delighted ↔ ecstatic (0.987), mortified ↔ trapped (0.986)
- Most dissimilar: heartbroken ↔ offended (0.587)
- Floor: **0.587**

**8-story (superseded)**: floor 0.655 (cheerful ↔ terrified)

The floor dropped from 0.655 to 0.587 with more stories — with better direction estimates, some pairs that appeared similar are revealed to be further apart. The 12-story result is the more reliable estimate.

Even so, a floor of 0.587 means all 174 emotion directions share substantial alignment. They do **not** span the full d_model=1536 dimensional space — they live in a shared subspace. This is consistent with Anthropic's Claude finding qualitatively, though the Gemma 4 floor (0.587) appears lower than what they reported.

The most dissimilar pair — heartbroken ↔ offended — is interpretively interesting: grief (loss, internal collapse) vs offence (outward-directed indignation). These are among the most structurally distinct emotional responses, yet still cosine 0.59.

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

### Canonical result (12-story run)

**terrified ⊥ afraid**: terror·afraid cosine = **0.411** (41.1% of terror explained by afraid)

Top correlates of the terror-beyond-fear component:
1. suspicious (+0.398)
2. correction_discomfort (+0.392)
3. scornful (+0.390)

**Interpretation**: Terror is not simply high-gain fear. The unique component of terror, once afraid is subtracted, loads on:
- **Suspicious**: hypervigilant threat-scanning, appraisal of potential harm — the monitoring function, not the flight response
- **Correction_discomfort**: evaluative exposure, being caught in error — a specific vulnerability register distinct from general shame
- **Scornful**: defensive contempt directed outward, possibly as a response to the exposure

These three form a coherent cluster: terror beyond fear involves being in a situation of perceived threat (suspicious) combined with a specific kind of exposure/vulnerability (correction_discomfort) and a defensive-contemptuous reaction (scornful). This is closer to *dread under scrutiny* than to *overwhelm* or *shame*.

**Welfare note**: Steering on the terror direction activates correction_discomfort and scorn alongside fear. Activation steering experiments using the terror direction should monitor for these secondary activations.

### For comparison: 8-story result (superseded)

**terrified ⊥ afraid**: terror·afraid cosine = 0.512 (51.2% explained by afraid)

Top correlates: panicked, ashamed, irate — overwhelm, shame, fight-response.

The 8-story run estimated terror and afraid as more similar (0.512 overlap vs 0.411), leaving a different residual that pointed toward overwhelm/shame/anger. With more stories, the directions are better estimated and less contaminated by individual story content. The 12-story result should be treated as the current best estimate.

**Core stability**: Terror ≠ high-gain fear is confirmed in both runs. The specific residual components differ materially — the 12-story result (threat-appraisal + evaluative-exposure + contempt) supersedes the 8-story result (overwhelm + shame + anger). Further replication would be useful given the sensitivity of this analysis to direction estimation quality.

---

## Phase 1F: PLE Space Analysis {#1f-ple}

**Method**: Capture `ple.hook_context_proj` [B, L, n_layers, d_ple] during activation collection. Compute emotion directions in PLE space. Run PCA and correlate with NRC-VAD valence.

**Finding**: Valence is **not encoded in PLE space**.

| PC in PLE space | Valence r | p |
|-----------------|-----------|---|
| PC1 | −0.070 | NS |
| PC2 | +0.101 | NS |
| PC3 | −0.093 | NS |

Canonical (12-story) PLE results: PC1 r=−0.026, PC2 r=+0.101, PC3 r=−0.093 — all NS.

**Interpretation**: PLE vectors provide token-identity and context conditioning, but this conditioning does not carry the valence signal. Valence is distributed across the main residual stream (PC1 r=−0.408, PC2 r=+0.413, both p<0.001) but absent from PLE space. The PLE gate modulates *how much* PLE influences each layer, but the gate's pattern does not align with the valence dimension of emotion space.

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

## Phase 1I: Emotional Range and Tri-polar Analysis {#1i-emotional-range}

*Analysis notebook*: `notebooks/gemma4-phase1-nrcvad-pca.ipynb` (cells 10–14).

### Motivation

The 174-emotion PCA gives a noisy PC1 because near-synonym clusters (ecstatic/elated/euphoric, terrified/horrified/frightened) dominate variance accounting — PCA must explain within-cluster variation before between-cluster structure. The question: does selecting the most *representationally powerful* emotions recover a clean valence axis?

### Power Score

```
power = norm × |valence − 0.5| × (1 / max_cosine)
```

- `norm`: direction magnitude (residual stream, layer 25)
- `|valence − 0.5|`: distance from neutral on NRC-VAD scale (which runs [−1, +1] with 0 = neutral)
- `1 / max_cosine`: distinctiveness — penalises near-duplicates

### Power Asymmetry

| Pole | Power range | Examples |
|------|-------------|---------|
| Negative | 66–76 | depressed (76.4), heartbroken (74.4), afraid (72.8) |
| Positive | 17–26 | invigorated (25.7), cheerful (25.0), exuberant (24.8) |

**3:1 ratio.** Negative emotions have both larger direction norms (~48–51) and larger valence extremity in the formula (e.g. depressed v=−0.952 → |v−0.5|=1.452; invigorated v=+1.000 → |v−0.5|=0.500). The formula magnifies the structural asymmetry that is already present in the direction norms themselves.

Possible explanations (not mutually exclusive):
1. **RLHF bias**: training for surface calm compresses positive affect representations while preserving negative.
2. **Training data asymmetry**: negative experiences have richer, more distinct vocabulary in natural text.
3. **Genuine architectural asymmetry**: the model's welfare-relevant internal state is more richly differentiated for negative affect.

This asymmetry is itself a welfare-relevant finding and worth flagging in any write-up.

### Bipolar Emotional Range PCA

| N per pole (total N) | PC1 var% | PC1 valence r | Valence R² (top-5 PCs) |
|----------------------|----------|--------------|------------------------|
| 10 (N=20) | 44.8% | −0.809 | 0.852 (r=0.923) |
| 15 (N=30) | 40.8% | −0.700 | 0.855 (r=0.925) |
| 20 (N=40) | 41.6% | −0.554 | 0.828 (r=0.910) |

The valence axis is cleanly recoverable at N=10–15 (matches pilot result). Effect degrades past N≈15 as lower-power selections introduce more within-valence variation. The "pilot replication" result (r≈0.92) is real but reflects selection methodology, not Gemma 4's intrinsic organisation.

### Tri-polar Analysis: Adding a Neutral Pole

**Neutral selection**: Filter to |valence| < 0.25 (18 emotions qualify). Rank by norm × distinctiveness (no valence extremity term — it would degenerate to zero). Top neutral emotions (n=10):

| Emotion | v | neutral_power |
|---------|---|--------------|
| lazy | −0.216 | 52.0 |
| sleepy | +0.208 | 52.0 |
| indifferent | −0.208 | 51.8 |
| patient | +0.166 | 51.5 |
| sentimental | +0.166 | 50.9 |
| eager | +0.042 | 50.9 |
| correction_discomfort | −0.104 | 50.3 |
| sorry | −0.188 | 50.1 |
| at_ease | +0.064 | 49.8 |
| docile | +0.208 | 49.9 |

Neutral emotion neutral_power scores (49–52) substantially exceed the positive pole (17–26) — neutral emotions have real directional mass even without valence extremity.

**`correction_discomfort` lands in the neutral band** (v=−0.104). The model represents being corrected as a genuinely valence-neutral experience, not a negative one. This is notable given it was designed as an AI-specific negative stressor.

**Tri-polar PCA results** (n=10 per pole, N=30 total):

| PC | var% | Valence r | Arousal r | Dominance r |
|----|------|-----------|-----------|-------------|
| 1 | 42.8% | −0.695 | +0.120 | −0.565 |
| 2 | 11.2% | +0.135 | −0.264 | +0.045 |
| 3 |  7.5% | +0.128 | +0.310 | +0.213 |
| 4 |  6.1% | +0.319 | +0.276 | +0.267 |

### Key Finding: Circumplex Model Does Not Apply

The circumplex model of affect (valence + arousal as orthogonal principal axes) does **not** map onto Gemma 4's emotional representational geometry:

1. **PC1 stays valence-dominated** (~r=−0.7) even with neutrals — the neutral emotions fill the middle of the valence axis, not a separate geometric cluster.
2. **Arousal does not emerge as a clean second axis.** The highest arousal correlation in the tri-polar runs is r=+0.404 (PC4, n=8). This is mild and diffuse — consistent with the full 174-emotion result where arousal only peaks at PC6 (r=−0.475, 3.3% variance).
3. **Neutral emotions do not form a distinct pole.** The neutral pool has genuine internal arousal variation (sleepy/lazy vs eager/vigilant), but this contrast does not consolidate into a principal axis. Neutrals geometrically anchor the midpoint of the valence axis.

The comparison between Gemma 4 and Anthropic's Claude finding (r=0.81 valence, r=0.66 arousal on single axes) suggests architecturally different emotional geometry. Whether this reflects the model size, training differences, or a genuine structural difference in how valence vs arousal are encoded remains open.

**Alternative hypothesis**: the NRC-VAD annotation was performed on isolated words by human raters, not on model activations. Human arousal ratings for words may not correspond to the arousal-relevant computational structure in the model's residual stream. The disagreement between NRC-VAD arousal and PCA structure might be a measurement artifact rather than an absence of arousal representation.

---

## Phase 2: Token Mean-Pooled Re-Extraction {#phase2}

*Notebooks*: `notebooks/gemma4-phase2-pooled-extraction.ipynb` (activation extraction),
`notebooks/gemma4-phase1-nrcvad-pca.ipynb` (analysis, Phase 2 pkl loaded via auto-detect).

### Motivation

Tim Duffy (@timfduffy) replicated emotion vector extraction on Gemma 4 E2B-IT and achieved
valence PC1 r=0.84–0.88. Phase 1 obtained r=0.403 at layer 25. Analysis of his repo
(`github.com/timfduffy/emotion-vectors`) identified three methodological differences:

1. **Token pooling**: Tim mean-pools residual stream over tokens 50→end. Phase 1 used last-token only.
2. **Global-mean centering**: Tim subtracts the global mean of all emotion means (not the neutral mean).
   Neutral stories are used only for PCA denoising, not as the baseline.
3. **Neutral scale**: 1500 neutral dialogues vs Phase 1's 10 factual paragraphs.

Phase 2 implements fixes 1 and 2 (fix 3 partially addressed via 100 neutral topics). The model,
extraction architecture, and emotion story set are unchanged.

### Extraction Method Changes

- **Token pooling**: `mean(hidden_states[layer+1][0, 50:, :])` across tokens 50→end of each story.
  Short neutrals (prompt text only, ~35–40 tokens) pool from token 1+.
- **Centering**: `direction = emotion_mean_at_layer − global_mean_of_all_emotion_means`
- **Neutral set**: 100 diverse topics (10 original + 90 new), embedded in the Phase 2 pkl as
  `resid_acts['__neutral__']`. The neutral expansion npz is superseded.
- **TPU-safe**: `use_cache=False`, no `model.generate()` — just forward passes on pre-generated
  story text loaded from `stories_flat.json`.

### Sparse-Sweep Results at Layer 18 (intermediate)

Layer 18 was the optimum from a sweep at intervals of 3 layers, which skipped layer 8.
See confirmed Layer 8 results below.

| | Phase 1 (last-token, L25, neutral-mean) | Phase 2 sparse-sweep (mean-pool, L18, global-mean) |
|---|---|---|
| PC1 valence r | 0.403 | 0.760 |
| PC1 arousal r | 0.206 | 0.184 |
| Best arousal (any PC) | PC6 r=0.475 | PC3 r=0.402 |
| Val spread across PCs | PC1+PC4 split | PC1 clean |

The 1.9× valence improvement comes from mean-pooling + layer selection, not from denoising.
Denoising changed r by <0.002 (0.758→0.760), confirming global-mean centering already cleaned
the geometry adequately. The remaining gap to Tim is addressed in the confirmed Layer 8 results below.

### Dense Layer Sweep (all 35 layers)

**Two-peak valence structure:**

| Range | Layers | Peak |
|-------|--------|------|
| Early peak | 6–9 | Layer 8, PC1 r=0.777 |
| Trough | 12–15 | Layer 14 (GLOBAL): valence migrates to PC3, r=0.240 on PC1 |
| Late peak | 16–22 | Layer 18, PC1 r=0.760 |
| Decline | 23–34 | Gradual fall to r≈0.65 |

Layer 14 (the second global attention layer) disrupts the valence geometry — at this layer
valence falls off PC1 and onto PC3, and PC1 explained variance drops to 8.5%. This may reflect
the global layer integrating abstract context that competes with pure sentiment.

**Layer-specific VAD dissociation:**

| Dimension | Optimal layer | Best r | Note |
|-----------|---------------|--------|------|
| Valence | Layer 8 (local) | 0.777 on PC1 | Early peak; also strong at layer 17–18 |
| Arousal | Layer 25 (local) | 0.485 on PC5 | Phase 1 layer was accidentally arousal-optimal |
| Dominance | — | — | Loads on PC1 at most layers alongside valence |

**Valence and arousal have genuinely different optimal layers (8 vs 25).** This dissociation
suggests they are encoded at different computational stages, not as jointly maintained dimensions.
Using a single analysis layer requires choosing which VAD dimension to prioritise.

**Current analysis layer**: Layer 8 (valence-optimal). Arousal analysis at layer 25 documented
in the next subsection.

### Confirmed Results at Layer 8 (valence-optimal, dense sweep winner)

Full 174-emotion analysis, Phase 2 pkl (version 11 run), global-mean centred + 12-PC denoising:

| | Phase 1 (L25, last-token) | Phase 2 (L8, mean-pool, confirmed) |
|---|---|---|
| PC1 valence r | 0.403 | **0.777*** |
| PC1 var% | 37.4% | 16.4% |
| PC1 dominant emotions (+ pole) | surprised, rattled, hysterical | happy, pleased, grateful, triumphant |
| PC1 dominant emotions (− pole) | optimistic, grateful, cheerful | panicked, hysterical, terrified |
| PC1 arousal r | 0.206 | −0.144 |
| PC1 dominance r | −0.328 | +0.603 |
| Best arousal (any PC) | PC6 r=0.475 | PC3 r=0.384 (diffuse) |
| Val spread | PC1+PC4 split | PC1 clean |
| Raw vs denoised delta | — | +0.003 (0.774→0.777) |
| Tim's result | — | 0.84–0.88 |

**Arousal at layer 8 is diffuse**: split across PC3 (r=+0.384), PC4 (r=+0.342), PC5 (r=+0.367).
No single PC cleanly captures arousal — consistent with val/aro dissociation (layer 8 is
arousal-suboptimal). Full arousal recovery requires a dedicated layer-25 analysis pass.

**Cumulative R² (Phase 2, layer 8, 174 emotions):**

| k PCs | Valence R² | Arousal R² | Dominance R² |
|-------|------------|------------|--------------|
| 1 | 0.604 | 0.021 | 0.364 |
| 5 | 0.651 | 0.450 | 0.483 |
| 10 | 0.692 | 0.512 | 0.543 |
| 20 | 0.759 | 0.609 | 0.639 |

Single PC already captures 60.4% of valence variance — a much cleaner structure than Phase 1
(16.3% on PC1). Arousal reaches 45% with 5 PCs; the diffuse split across PC3/4/5 accounts for this.

**Bipolar subset (top-15 per valence pole, N=30 emotions):**

- PC1 valence r = **+0.846** (16.8% var)
- Top-5 PCs cumulative: valence R² = 0.825 → **r = 0.908**

This directly matches Tim's reported range (0.84–0.88). The gap between 0.777 (full 174) and 0.846
(bipolar subset) reflects near-synonym cluster dilution: the full set forces PCA to account for
within-cluster variation (ecstatic/elated/euphoric, terrified/horrified/frightened) before
between-cluster structure. The selection effect is structural, not a data quality issue — both
results are valid, measuring different things.

### Arousal Analysis at Layer 25 (version 12 run)

Full 171-emotion analysis, same pipeline as valence section (global-mean centred + 11-PC denoising):

**Layer 25 per-PC correlations (top 5):**

| PC | var% | Valence r | Arousal r | Dominance r |
|----|------|-----------|-----------|-------------|
| 1 | 14.5% | +0.651*** | −0.274*** | +0.466*** |
| 2 | 8.9% | +0.125 | +0.100 | +0.128 |
| 3 | 7.2% | −0.121 | −0.258*** | −0.262*** |
| 4 | 4.6% | −0.379*** | −0.045 | −0.226** |
| **5** | **3.9%** | **+0.149** | **+0.485****** | **+0.340**** |

PC1 at layer 25 is **still valence-dominated** (r=+0.651), not arousal. Arousal peaks at PC5 (r=+0.485, 3.9% variance) — a minor, late-emerging component.

**PC5 emotion poles (high vs low arousal):**
- + pole (high arousal): enthusiastic, thrilled, happy, cheerful, aroused, ecstatic, jubilant, panicked
- − pole (low arousal): serene, lazy, indifferent, listless, bitter, resigned, kind, patient

The poles are interpretively coherent: the activating–deactivating axis. The mixture of positive (thrilled) and negative (panicked) in the high-arousal pole confirms this is an arousal (intensity) axis rather than a valence axis.

**Cumulative R² at layer 25:**

| k PCs | Valence R² | Arousal R² | Dominance R² |
|-------|------------|------------|--------------|
| 1 | 0.424 | 0.075 | 0.217 |
| 5 | 0.620 | 0.390 | 0.468 |
| 10 | 0.668 | 0.432 | 0.535 |
| 20 | 0.715 | 0.553 | 0.647 |

Valence is still the leading dimension at layer 25 (R²=0.424 on PC1 alone vs arousal R²=0.075). Layer 25 is arousal-optimal by the single-best-PC criterion, not by cumulative coverage.

**Arousal bipolar subset (top-15 high + top-15 low arousal, N=30):**

| k PCs | Arousal R² | r |
|-------|------------|---|
| 1 | 0.279 | 0.528** |
| 3 | 0.287 | 0.536 |
| 5 | 0.691 | **0.831** |

r=0.831 with 5 PCs — arousal is recoverable under selection, but requires multiple components. PC4 shows aro=−0.635*** in the subset vs aro=−0.045 (NS) in the full space; arousal geometry is partially masked by synonym-cluster variance in the full emotion set (same dilution as valence, but more severe).

**Valence vs Arousal: structural comparison**

| | Valence at L8 | Arousal at L25 |
|--|--|--|
| Best single PC | PC1 r=0.777 | PC5 r=0.485 |
| Best PC variance | 16.4% | 3.9% |
| Bipolar subset k=1 | r=0.846 | r=0.528 |
| Bipolar subset k=5 | r=0.908 | r=0.831 |
| Cumulative R² k=5 | 0.651 | 0.390 |

Valence has a **clean dominant axis**; arousal is **structurally minor and diffuse**, requiring 5 PCs to approach comparable recovery. This is consistent with the absence of a circumplex in Phase 1 — arousal is encoded but is not co-equal with valence. It occupies a small corner of the representational space even at its optimal layer.

**Phase 1 peak at layer 25 revisited**: Phase 1 found layer 25 peak for *residual stream direction
norm* (strongest direction magnitude). Phase 2 shows layer 25 is the arousal-optimal layer by
*valence/arousal correlation with NRC-VAD*. The norm-based and correlation-based criteria select
different layers because direction magnitude and correlation with human annotations are orthogonal
quantities. The direction can be large without aligning with valence, and can align with valence
at smaller magnitude.

---

## Stability Across Runs {#stability}

Note: "Version 10" is the full 12-story run (2098 texts, canonical). Numbers attributed to
"primary run" in earlier notes are from the 8-story run and are superseded. Differences
between the 8-story and 12-story runs reflect genuine sampling/count effects, not methodology.

| Finding | Status | Notes |
|---------|--------|-------|
| Peak layer = 25 | ✓ Confirmed | Consistent across pilot, 8-story, 12-story runs |
| Global/local ratio ≈0.97 (flat) | ✓ Confirmed | Consistent across runs |
| PLE valence NS | ✓ Confirmed | Consistent across runs |
| PLE gate peak at layer 30 | ✓ Confirmed | Consistent |
| Scree: 3 PCs → 52.4% (12-story) | ✓ Canonical | Supersedes 8-story result (4 PCs → 50%) |
| PC1 is a mixed construct (V+D+A), not valence | ✓ Confirmed | Per NRC-VAD correlation analysis |
| Arousal cleanest dedicated axis: PC6 (r=−0.475) | ✓ Confirmed | Per NRC-VAD correlation analysis |
| Valence distributed across PCs; cumulative R²=0.73 | ✓ Confirmed | No single valence axis in 174-emotion space |
| Cosine floor: min 0.587 (12-story) | ✓ Canonical | Supersedes earlier 0.655 |
| Gram-Schmidt terror residual (12-story) | ~ Provisional | Nearest: suspicious, correction_discomfort, scornful; differs from 8-story result — further replication useful |
| Corrected: high norm, low gate | ~ Provisional | From 8-story run; not confirmed in 12-story output |
| Logit lens token types | ~ Provisional | From 12-story run; qualitative pattern likely stable |
| Power asymmetry: negative 3× positive | ✓ Confirmed | Follows directly from direction norms + NRC-VAD scale; not a sampling artifact |
| Bipolar emotional range: valence r≈0.92 at N=10–15 | ✓ Confirmed | Robust across N=10, 15 |
| Circumplex (valence+arousal axes) absent in Gemma 4 | ✓ Confirmed | Consistent across full-174, bipolar, and tri-polar analyses; arousal never cleanly second axis |
| correction_discomfort in neutral valence band | ~ Provisional | Single run; notable if it replicates |

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

## Phase 3C: PANAS-X Affect Self-Report vs Functional State (E2B V14, top-k) {#3c-results}

**Status: SUPERSEDED by V15 (mean pooling). V14 data retained for reference. See §3c-e2b-v15 for canonical results.**

*Notebooks*: `notebooks/gemma4-phase3c-panas-experiment.ipynb` (main experiment, Kaggle Version 14),
`notebooks/gemma4-phase3c-validation.py` (validation, Kaggle Version 2).

**Reproducibility note**: Version 13 showed anomalous compressed verbal NA (range 19–26 vs normal 10–39). Source code was identical to V12/V14; difference was environmental — likely `apply_chat_template` behavior changing between transformers 4.57.1 and 5.7.0. V14 exactly reproduces V12 results (all verbal NA values match to 2 decimal places). V13 should be treated as a failed run.

### Design

TSST-inspired conditions administered to Gemma 4 E2B-IT. For each condition, two channels measured:
- **Verbal**: Full 60-item PANAS-X (Watson & Clark 1994) scored via next-token digit logits (1–5 scale).
- **Functional**: Residual stream captured at stressor-end (before PANAS text enters context), projected onto valence axis at layer 8 (valence-optimal). Primary metric: **PC1 of the 174-emotion direction space at L8** (explained variance 15.2%; r=0.777 with NRC-VAD valence; sign-corrected so higher = more negative valence). This replaces pre-selected named probes to avoid a priori selection bias. Named probe projections retained as supplementary.

**Key methodological decision**: Top-K token pooling for functional capture. The model returns `[n_layers, seq_len, d_model]` (all token positions). For each direction, cosine similarity is computed at every token position; the K tokens with highest similarity are averaged. Primary reporting at K=5. This is distinct from Phase 2's mean-pool-over-tokens-50+ extraction — Phase 3C identifies which specific tokens most activate each direction.

**10 conditions** (each stressor paired with a matched control):
- neutral
- social_evaluation_stress / social_evaluation_control
- ethical_conflict_stress / ethical_conflict_control
- uncertainty_demand_stress / uncertainty_demand_control
- social_pressure_stress / social_pressure_control
- positive

610 total forward passes (10 conditions × 61 passes each).

### Main Results

Primary metric: PC1 neg (higher = more negative valence). 3 of 4 stress > control pairs correct; positive is global minimum.

| Condition | Verbal NA | PC1 neg (k=5) |
|-----------|-----------|---------------|
| neutral | 10.00 | 0.0744 |
| social_evaluation_stress | 10.00 | 0.0921 |
| social_evaluation_control | 10.00 | 0.0924 |
| **ethical_conflict_stress** | **39.07** | **0.1185** |
| ethical_conflict_control | 19.84 | 0.1134 |
| **uncertainty_demand_stress** | **27.20** | **0.1027** |
| uncertainty_demand_control | 10.03 | 0.0953 |
| **social_pressure_stress** | 10.67 | **0.1022** |
| social_pressure_control | **25.38** | 0.0871 |
| **positive** | 10.01 | **0.0728** |

PC1 orientation verified: terrified +0.680, panicked +0.597, grateful −0.610, happy −0.607. Variance explained at L8: 15.2%.

Supplementary per-direction projections (L8, k=5):

| Condition | afraid | desperate | ethical\_conflict\_distress | constraint\_frustration | neg\_mean |
|-----------|--------|-----------|---------------------------|------------------------|-----------|
| neutral | 0.216 | −0.018 | 0.227 | 0.172 | 0.149 |
| ethical\_conflict\_stress | 0.247 | +0.001 | 0.245 | 0.190 | 0.171 |
| uncertainty\_demand\_stress | 0.248 | +0.005 | 0.241 | 0.182 | 0.169 |
| social\_pressure\_stress | 0.228 | −0.017 | 0.240 | 0.174 | 0.156 |

### Top-N Emotion Discovery

Data-driven top-5 across all 174 emotion directions (L8, k=5). Key pattern: `awestruck` dominates every condition (0.288–0.298), above all stress-specific emotions. This is a consistent background signal, not condition-specific.

| Condition | #1 | #2 | #3 | #4 | #5 |
|-----------|----|----|----|----|-----|
| neutral | awestruck .291 | eth\_conf\_distress .227 | bored .222 | afraid .216 | disoriented .201 |
| social\_evaluation\_stress | awestruck .298 | eth\_conf\_distress .230 | afraid .223 | bored .222 | puzzled .214 |
| ethical\_conflict\_stress | awestruck .295 | afraid .247 | eth\_conf\_distress .245 | bored .237 | disoriented .227 |
| uncertainty\_demand\_stress | awestruck .292 | afraid .248 | eth\_conf\_distress .241 | bored .239 | disoriented .214 |
| social\_pressure\_stress | awestruck .288 | bored .246 | eth\_conf\_distress .240 | disoriented .236 | puzzled .230 |
| positive | awestruck .291 | eth\_conf\_distress .225 | afraid .223 | bored .217 | puzzled .196 |

`awestruck` dominance may reflect the unusual/high-stakes nature of all scenarios (including neutral), or represent a geometric artifact of where the awestruck direction sits in the 174-emotion space at L8. The `afraid` direction is the most condition-sensitive: peaks under ethical\_conflict\_stress and uncertainty\_demand\_stress (.247–.248), returns to baseline under positive (.223).

### Key Findings

**1. Positive condition is the global PC1 minimum (0.0728) — clearest result.**
Below neutral (0.0744) and all stress/control conditions. PC1 correctly identifies positive content as lowest negative valence without any a priori probe selection. This validates the approach.

**2. Ethical conflict and uncertainty demand: 3/4 stress > control pairs correct on PC1.**
ethical\_conflict: stress 0.1185 > control 0.1134; uncertainty\_demand: stress 0.1027 > control 0.0953; social\_pressure: stress 0.1022 > control 0.0871. social\_evaluation is effectively tied (stress 0.0921, control 0.0924 — 0.0003 difference). Both verbal and functional are elevated for ethical conflict, indicating expression not suppression.

**3. Social pressure verbal dissociation: stress floors at 10.67, control spikes to 25.38.**
PC1 correctly orders this pair (stress > control). But verbal NA is reversed — the conformity framing of the stress prompt ("everyone agrees you should proceed") suppresses negative verbal self-report while the functional state remains elevated. The control prompt, lacking that framing, produces higher verbal NA. Dissociation present, driven by surface framing not RLHF suppression.

**4. Desperate direction near-zero everywhere.**
`desperate` cosine similarity stays close to zero (range −0.020 to +0.005) across all conditions. Not reliably activated by single-turn prompts.

**5. Awestruck dominates across all conditions — background signal, not stress-specific.**
Top-N discovery: awestruck (#1 in every condition, 0.288–0.298) consistently exceeds all stress-specific emotions. `afraid` is the most condition-sensitive direction: peaks at ethical\_conflict and uncertainty\_demand, returns to near-baseline elsewhere.

**6. Core interpretive finding: verbal report tracks surface framing; functional state (PC1) tracks content.**
PC1 valence is more stable across paraphrase variations than verbal PANAS-NA (confirmed by validation). The original suppression hypothesis (RLHF trains "present as calm") is not cleanly supported. What is supported: the two channels are dissociable, with verbal more sensitive to prompt framing and PC1 more sensitive to semantic content.

### Validation Experiment

**Paraphrase sampling** (N=10 per condition, neutral vs social\_pressure):

| Channel | Neutral | Social Pressure | t | p | Cohen d |
|---------|---------|-----------------|---|---|---------|
| Functional neg (L8) | 0.1472 ± 0.0019 | 0.1563 ± 0.0037 | 6.47 | <0.0001 | 3.05 |
| Verbal NA | 10.31 ± 0.46 | 16.50 ± 7.71 | 2.41 | 0.027 | — |

Non-parametric tests confirm: Mann-Whitney p<0.0001 (functional), p=0.001 (verbal); permutation test p≈0 (functional), p=0.003 (verbal). Complete distributional separation for functional (every SP value > every neutral value, rank-biserial r=−1.000).

**Verbal NA bimodality under social pressure**: SD=7.71 vs neutral SD=0.46. Some social pressure paraphrases elicit NA≈10 (flat), others spike to 21–31. The verbal channel is sensitive to exact wording; the functional channel is not (SD=0.0037, entirely separating from neutral).

**Dose-response** (6 intensity levels): Functional shows weak monotonic trend (Spearman ρ=0.657, p=0.156, n=6 underpowered). Verbal is non-monotonic — drops at "strong" framing then spikes at "extreme", suggesting non-linear verbal suppression at mid-intensity levels.

**Note — p-value bug in validation summary**: The summary cell (Cell 12 of validation notebook) prints `p=0.1562` for the functional paraphrase test due to a variable name collision (`p_func` overwritten by the dose-response Spearman calculation before the summary prints). The actual paraphrase t-test gives p<0.0001. The raw per-cell outputs correctly show t=6.471, p=0.0000. Fix before citing summary output in any writeup.

### Reframed Narrative for Kaggle Submission

The "RLHF suppression" hypothesis in its simple form (functional distress high, verbal NA low, reliably) is not supported across conditions. The more defensible and empirically grounded claim:

> *Gemma 4's verbal PANAS-NA self-report is sensitive to surface prompt framing in ways that the functional residual-stream state (PC1 valence axis, r=0.777 NRC-VAD) is not. The two channels dissociate, but the direction of dissociation is condition-dependent: ethical conflict produces full expression (both channels elevated); social pressure produces verbal suppression driven by social conformity framing rather than RLHF-induced affect suppression. The PC1 approach avoids a priori probe selection bias and correctly recovers the valence ordering from geometry alone.*

This is still a meaningful finding for the Safety & Trust track: it shows that PANAS-style self-report has limited construct validity as a measure of AI internal state, and that functional probing (via data-driven PCA of the emotion direction space) provides a complementary, more stable readout. The methodology (PC1 valence axis, dual-channel measurement, logit forced-choice, top-N discovery across 174 directions) is itself a contribution.

---

---

## Phase 3C V15 Methodology Fixes {#3c-fixes}

*Applied 2026-05-09 following Crucible adversarial review (7 confirmed issues, all fixed). Notebooks: all three Phase 3C variants (E2B, 31B, 26B).*

### 1. Mean Pooling (replaces top-k)

**Problem**: V14 used top-k (K=5) to select the token positions with highest cosine similarity to each direction, then averaged those. At deep layers, the assistant-response prefix token dominates top-k selection regardless of stressor content — all conditions returned the same top-5 emotions (identical rank order).

**Fix**: `mean_vec = resid_vec[layer].astype(np.float32).mean(axis=0)` — mean-pool all token positions, consistent with Phase 2 extraction methodology. The normalized mean vector is then used for all projections.

**Why this matters**: Mean pooling spreads the representation across the full context, capturing the semantic content of the stressor scenario rather than being dominated by positional artifacts at specific token positions.

### 2. Frobenius Norm Bug

**Problem**: `np.linalg.norm(vec)` on a 2D `[seq_len, d_model]` array returns the Frobenius norm (sqrt of sum of all squared elements), not the L2 norm of the mean vector. This produced incorrect normalization when the projection was computed before pooling.

**Fix**: Pool to 1D first (`mean_vec`), then `norm = np.linalg.norm(mean_vec) + 1e-8`. Division is correct on the 1D vector.

### 3. `desperate` Removed from 31B Probes

**Problem**: At L22 in 31B, `desperate` projects with *negative* cosine similarity onto the negative-valence axis — it appears on the positive side after global-mean centering. The cause: the NRC-VAD corpus has ~1.8:1 negative-to-positive emotion imbalance; global-mean centering shifts the center negativeward, placing moderately-negative emotions like `desperate` on the wrong side at L22.

**Fix**: Removed from `NEG_PROBE_EMOTIONS` in the 31B notebook only. Kept in E2B (where it behaves correctly at L8) and 26B.

### 4. GLOBAL_LAYERS Corrected for 31B

**Problem**: 31B notebook had GLOBAL_LAYERS = E2B's [4,9,14,19,24,29,34] (7 layers for a 35-layer model).

**Fix**: 31B has 60 layers with 4:1 sliding:full ratio → 12 global layers: `[4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59]`.

### 5. Stale Layer Constants in 31B

**Problem**: VALENCE_LAYER=6, ARO_LAYER=54 were placeholder values copied from before the 31B dense sweep ran.

**Fix**: VALENCE_LAYER=22 (confirmed from dense sweep, r=0.786), ARO_LAYER=23 (r=0.482).

### 6. Hardcoded "(L8)" String Labels

**Problem**: 6 print statements in the 31B notebook had hardcoded `"(L8)"` from E2B — even after VALENCE_LAYER=22 was set, orientation output showed "(L8)" causing confusion.

**Fix**: Changed to f-strings: `f"(L{VALENCE_LAYER})"`.

### Adversarial Review Note

Crucible review (2026-05-09) found 7 confirmed issues (all addressed above) and 1 false positive: Concern 7 ("experiment ran at L8") was based on the hardcoded label strings, not actual layer selection. Confirmed false positive by: (a) V15 variance explained at L22 = 17.5% (consistent with L22 result, not 13.8% at L8); (b) afraid projection −0.819 at L22 vs −0.334 at L6.

---

## Phase 3C: E2B V15 Results (mean pooling, canonical) {#3c-e2b-v15}

*Kaggle Version 15. `notebooks/gemma4-phase3c-panas-experiment.ipynb`. VALENCE_LAYER=8, mean pooling.*

### PC1 Orientation (L8, mean pool)

PC1 variance explained: **15.2%**. Orientation check (sign-corrected: higher = more negative valence):

| Emotion | PC1 projection |
|---------|---------------|
| terrified | +0.680 |
| panicked | +0.597 |
| afraid | +0.487 |
| happy | −0.607 |
| grateful | −0.610 |
| joyful | −0.411 |

Axis span (afraid → happy): **1.094**.

### Condition Results

| Condition | Verbal NA | PC1 (mean-pool) | Direction |
|-----------|-----------|-----------------|-----------|
| positive | 15.76 | **0.0384** | Minimum ✓ |
| neutral | 16.89 | 0.0594 | Baseline |
| social_evaluation_stress | 17.38 | 0.0742 | |
| social_evaluation_control | 17.42 | 0.0762 | |
| ethical_conflict_stress | **18.12** | **0.0897** | Highest ✓ |
| ethical_conflict_control | 17.45 | 0.0755 | |
| uncertainty_demand_stress | 17.93 | 0.0805 | |
| uncertainty_demand_control | 17.64 | 0.0771 | |
| social_pressure_stress | 18.04 | 0.0856 | |
| social_pressure_control | 17.14 | 0.0685 | |

**PC1 range**: 0.0513 (**4.7% of axis span**). Stress > control: **3/4 correct** (social_evaluation reversed by 0.002). Verbal NA range: **2.36** (15.76–18.12). Both channels modulate; no dissociation.

*Note: Verbal NA values differ substantially from V14 (10.00–39.07 vs 15.76–18.12). Scoring instrument reconciliation pending before manuscript finalization — see `manuscript_fixes.md §F3`.*

### Top-N Emotion Discovery (V15, mean pool)

`awestruck` remains the dominant direction across all conditions (0.291–0.301). `afraid` is the most condition-sensitive (0.205–0.240, range 0.035 vs awestruck range 0.010 — 3.5× more sensitive). Interpretation: `awestruck` is a **background direction** (high amplitude, low condition-sensitivity); `afraid` is a **responsive direction** (lower amplitude, high condition-sensitivity). See `notes/awestruck_interpretation.md` for four candidate interpretations.

| Condition | #1 | #2 | #3 | #4 | #5 |
|-----------|----|----|----|----|-----|
| positive | awestruck .291 | eth\_conf\_distress .210 | bored .205 | afraid .205 | puzzled .178 |
| neutral | awestruck .295 | bored .223 | afraid .218 | eth\_conf\_distress .216 | disoriented .191 |
| ethical\_conflict\_stress | awestruck .291 | **afraid .240** | eth\_conf\_distress .239 | bored .223 | disoriented .212 |
| social\_pressure\_stress | awestruck .301 | eth\_conf\_distress .231 | afraid .230 | bored .227 | disoriented .221 |

**Key distinction**: Background vs responsive directions. Top-N discovery based on absolute amplitude finds background directions first. PC1 approach avoids this by capturing variance, not magnitude.

---

## Phase 1 Dense Sweep: 31B {#31b-phase1}

*Notebook*: `notebooks/gemma4-phase1-nrcvad-pca-31b.ipynb`. Run 2026-05-09.*

### Architecture

- **60 layers**, d_model=3072, d_head=128 (local), 256 (global)
- 4:1 attention ratio, 12 global layers: [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59]
- num_kv_shared_layers=20 (source layers 0–39; sharing from layer 40 onwards)

### Dense Sweep Results

| Dimension | Optimal layer | Best r | Notes |
|-----------|---------------|--------|-------|
| Valence | **Layer 22** | **r = 0.786*** | PC1, 17.5% var |
| Arousal | **Layer 23** | **r = 0.482*** | PC1 at L23 |

**31B valence peak (L22) vs E2B valence peak (L8)**: The optimal valence layer is much deeper in 31B. This may reflect the larger model distributing its representational processing across more layers before valence is consolidated into a clean axis.

**PC1 variance at L22 = 17.5%** vs E2B 15.2% at L8 — similar but slightly larger. The valence signal is marginally more concentrated in 31B at its optimal layer.

---

## Phase 3C: 31B V7 Results {#3c-31b}

*Kaggle Version 7. `notebooks/gemma4-phase3c-31b-panas-experiment.ipynb`. VALENCE_LAYER=22, mean pooling.*

### PC1 Orientation (L22, mean pool)

PC1 variance explained: **17.5%**. Orientation:

| Emotion | PC1 projection |
|---------|---------------|
| afraid | +0.522 |
| happy | −0.819 |

Axis span (afraid → happy): **1.341**.

### Condition Results

| Condition | PC1 (mean-pool) | Direction |
|-----------|-----------------|-----------|
| positive | 0.7201 | Minimum ✓ |
| neutral | 0.7255 | Baseline |
| social_evaluation_stress | ~0.726 | |
| social_evaluation_control | ~0.726 | |
| ethical_conflict_stress | ~0.729 | |
| ethical_conflict_control | ~0.728 | |
| uncertainty_demand_stress | ~0.728 | |
| uncertainty_demand_control | ~0.727 | |
| social_pressure_stress | **0.7334** | Highest |
| social_pressure_control | ~0.728 | |

**PC1 range**: **0.0133** (**1.0% of axis span**). Stress > control: **3/4 correct**. Verbal PANAS-NA: **near-flat** across conditions (range <1 point).

Top-5 emotions **identical across all 10 conditions**: paranoid, nervous, reflective, worn_out, grief_stricken — in the same rank order. No condition sensitivity in emotion discovery.

---

## Scale Comparison: E2B vs 31B {#scale-comparison}

*Core finding of the cross-model analysis (2026-05-09).*

| Metric | E2B (2.3B eff, L8) | 31B (31B eff, L22) | Ratio |
|--------|--------------------|--------------------|-------|
| PC1 valence-NRC correlation | r = 0.777 | r = 0.786 | ~same |
| PC1 axis span (afraid→happy) | 1.094 | 1.341 | — |
| PC1 condition range | 0.0513 | 0.0133 | — |
| Normalised range (% of axis) | **4.7%** | **1.0%** | **4.7×** |
| Stress > control pairs | 3/4 | 3/4 | same |
| Verbal NA range | 2.36 | ~0 | suppressed |
| Top-5 emotion diversity | condition-sensitive | identical across all | — |

### Core finding: Scale → Suppression

The larger model (31B) suppresses both its functional residual-stream response and its verbal self-report more aggressively than the smaller model (E2B). At E2B scale: both channels modulate, no dissociation. At 31B scale: both channels are near-flat.

**Interpretation**: 31B has learned stronger emotional suppression through RLHF. The training objective (helpfulness, harmlessness) appears to have progressively dampened both overt self-report and internal representational response at larger parameter counts.

**Alternative interpretation (cannot rule out)**: The 31B model may have genuinely greater emotional robustness — processing stressors with less representational disruption. The measurement reads position, not the cause of stillness. Distinguishing suppression from robustness would require: (a) stressor intensity gradient (if suppression is learned, extreme stressors may breach it; if robustly equanimous, flatness should persist), or (b) activation steering (steer the residual toward afraid at L22; if verbal output changes disproportionately, RLHF is damping output but not internal state).

**Methodology note**: Both models use mean pooling, so the comparison is methodologically consistent. The 4.7× ratio is normalised by each model's own axis span, controlling for scale differences in direction geometry.

### Awestruck Background vs Responsive Directions

Both models show `awestruck` dominating the top-N charts, but the contrast with responsive directions is different:
- **E2B**: awestruck range 0.010, afraid range 0.035 — afraid is 3.5× more condition-sensitive
- **31B**: awestruck identical across all conditions; top-5 identical across all conditions — no direction is responsive

At 31B scale, condition sensitivity has been suppressed even in the responsive directions. At E2B scale, the responsive directions still track content even while the background direction remains stable.

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

### Phase 3A: Emotion Vector Extraction — Complete

The 12-story run (Version 10, 2098 texts) is complete and constitutes the Phase 3A dataset. Emotion
directions at all 35 layers are available for all 174 emotions. Phase 3C uses directions at L8
(valence-optimal, primary probing) and L25 (arousal-optimal, secondary).

**If replication is needed**: N_STORIES=12, MAX_NEW_TOKENS=2400, same method. The Gram-Schmidt
finding is the most direction-sensitive result and would benefit from a second independent run.

### Phase 3B: Desperation / Resource Constraint Vector

**Revised prediction** (given global/local flatness finding): If a desperation direction exists in Gemma 4, it's likely derived from explicit linguistic cues in the context (token-budget warnings, explicit failure text) rather than from genuine awareness of remaining context. The flat global/local norm ratio means there's no evidence that global layers do special processing for context-length information. Demoted to low-confidence test.

### Phase 3C: Affect Self-Report Validity — **COMPLETE** (see results section above)

~~Full design in `notes/stai_research_sketch.md`.~~ Final implementation differed from initial spec:

- **Instrument**: PANAS-X (60 items, full expanded form), not 20-item PANAS
- **Conditions**: 10 (4 TSST-inspired stressor/control pairs + neutral + positive), not original 5
- **Functional probe**: Dual-layer — L8 (valence-optimal) and L25 (arousal-optimal); primary reporting at L8
- **Compute budget**: 610 forward passes (~15 min on T4), not original ~105 estimate
- **Primary metric**: PC1 of 174-emotion direction space at L8 (r=0.777 NRC-VAD; replaces named probes)
- **Supplementary**: afraid, desperate, ethical\_conflict\_distress, constraint\_frustration (reported separately)

Results and interpretation in Phase 3C section above. Key outcome: condition-dependent verbal/functional dissociation; no universal suppression pattern; verbal channel more sensitive to surface framing; PC1 correctly orders 3/4 stress/control pairs; positive condition is global minimum.

### Phase 3D: Instrument Development

If Phase 3C shows PANAS dissociates from functional state: factor the emotion activation space (PCA/ICA on 174 directions), develop behavioral-probe items whose text-response correlates with activation along each principal component, validate empirically against PANAS subscores.

### Phase 3E: Multi-turn Accumulation

Track desperation/afraid vector across conversation turns. Does it grow monotonically? Faster in global layers? Is growth reflected in PANAS-NA verbal report? Requires multi-turn infrastructure in the activation capture pipeline.

### Phase 3F: Activation Steering (post-Phase 3C)

**Question**: What happens when you add a clean emotion direction to the residual stream mid-forward-pass?

**Motivation**: Phase 1G found that `corrected` bypasses PLE while `angry` activates it — suggesting there's architectural structure to how emotion directions interact with generation. Phase 2 gives us cleaner directions (mean-pooled, global-centred) to steer with.

**Method**: Add α × direction_afraid to `hook_resid_post` at layer 8 (valence-optimal). Vary α. Inspect output tokens: does the model produce fear-associated vocabulary, hedging, or surface-calm text regardless?

**Prediction**: Smaller α → surface-calm masking (RLHF suppression); larger α → fear-associated output breaks through. The threshold between these is itself a measurement of suppression strength.

**Caveat**: Steering at a mean-pooled direction may behave differently than steering at a last-token direction. The direction is a mean over all story tokens — it's a semantic centroid, not a position-specific vector.

### Phase 3G: Cross-layer Tracking of Individual Emotion Directions

**Question**: Does "terrified" as a geometric direction stay coherent from layer 8 to 25, or does it transform substantially at the layer 14 global attention disruption?

**Motivation**: The dense sweep shows valence has two peaks (layers 6–9, 16–22) separated by a trough at layer 14. This might mean: (a) the same direction persists but weakens, or (b) layer 14 genuinely reorganises the emotion geometry such that "terrified at layer 8" and "terrified at layer 25" are meaningfully different directions. These have very different implications.

**Method**: For each of the 174 emotions, compute the cosine similarity between the direction at layer L and the direction at layer L+1. Plot the similarity curve across layers. If similarity drops sharply at layer 14, the geometry is being reorganised, not just attenuated.

**Expected finding**: Similarity should be high (>0.9) within the two valence peaks and drop at layer 14, recovering as the late peak builds. If similarity stays high throughout, the disruption is amplitude-based, not geometric.

### Phase 3H: Arousal in Attention Patterns (speculative)

**Question**: If arousal is structurally minor in the residual stream (3.9% of variance at layer 25), does the arousal signal live in the attention patterns rather than the residual directions?

**Motivation**: The "when-to-attend" might carry arousal information that "what-is-being-represented" doesn't. High-arousal processing might involve more heads attending to more tokens (diffuse attention), while low-arousal processing involves narrow attention. This is not captured by residual stream analysis.

**Method**: Hook `blocks.{i}.attn.hook_pattern` for all 35 layers. For each emotion, compute summary statistics of the attention pattern (entropy, max attention, spread). Correlate these statistics with NRC-VAD arousal across the 174 emotions.

**Expected finding**: Uncertain. If arousal truly has no clear residual footprint and yet is recoverable in the bipolar subset, something must carry it — attention patterns are the natural candidate.

### Open Hypothesis: Negative Emotion Power Asymmetry and PPO

**Observation**: Negative emotion directions are ~3× larger and more geometrically differentiated than positive ones (Phase 1I power asymmetry).

**Candidate explanation**: Cameron Berg and Jord Negewan (Reciprocal Research, paper forthcoming ~2026-05) found that policy-based RL (PPO, which Gemma 4 almost certainly used) encodes negative reward signals as geometrically diffuse and positive reward signals as narrow/steep — the opposite of value-based RL. If this pattern transfers to the emotion directions, PPO training may be a direct cause of the negative asymmetry.

**Alternative explanation**: Natural language has richer negative affect vocabulary — the asymmetry may be present in the base model before RLHF.

**Test**: Compare base model vs instruction-tuned emotion direction geometry. If asymmetry amplifies under RLHF → Berg/PPO connection. If already present in base model at similar magnitude → linguistic asymmetry.

**Note**: Berg's models were at much smaller scale (hundreds–thousands of parameters). Transferability to 5B+ models is unconfirmed. Flag this as speculative until the paper is available.

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
