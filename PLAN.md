# Project Plan: Gemma 4 TransformerLens Support + Mechinterp

**Started**: 2026-04-20  
**Target**: Working PR against TransformerLens #953 + initial PLE diagnostic findings  
**Estimated duration**: 2.5–3 weeks  

---

## Context

TransformerLens 3.0 (milestone 91% complete) uses TransformerBridge as the recommended path for adding new model architectures. Issue #953 requests Gemma 3n (Gemma 4) support — open, unassigned, labeled complexity-high. No prior work exists.

Gemma 4 introduces **Per-Layer Embeddings (PLE)** — a parallel conditioning pathway that modulates hidden states at every layer. This is the primary novel element and the primary interpretability challenge. Our approach: expose PLE as explicit hook points rather than bypass it. The PLE characterization (what does it contribute at each layer?) is itself a research contribution.

---

## Phase 1: Environment and Architecture Mapping
**Duration**: 2–3 days  
**Goal**: Understand the implementation surface before writing code

### Tasks

1. **Fork TransformerLens** and set up development environment
   - Clone fork locally
   - Read an existing adapter in `transformer_lens/model_bridge/supported_architectures/` (Llama or Gemma 3)
   - Understand the pattern: adapter class, weight mapping, generalized component usage

2. **Load Gemma 4 E2B via nnsight** as a stopgap
   - Enumerate all named modules via `model.named_modules()`
   - Identify: attention layers (which are local? which global?), MLP layers, layernorm layers, PLE modules, shared KV layers
   - Note the HuggingFace class names for each

3. **Document architecture map** (`notes/architecture.md`)
   - Full module hierarchy for E2B
   - Per-layer type table: layer index → {attention type (local/global), shares KV?, has PLE?}
   - PLE module interface: inputs, outputs, parameter shapes

### Deliverables
- `notes/architecture.md` — complete E2B module map
- Development environment set up in TL fork

---

## Phase 2: Core Adapter
**Duration**: 5–7 days  
**Goal**: Working TransformerBridge adapter for Gemma 4 E2B (without PLE hooks yet)

### Tasks

1. **Create `Gemma4Adapter`** in `transformer_lens/model_bridge/supported_architectures/gemma4.py`
   - Extend the appropriate base adapter class
   - Map HuggingFace attention → TL generalized attention component
   - Map HuggingFace MLP → TL generalized MLP component
   - Map layernorm (RMSNorm) → TL generalized layernorm

2. **Handle alternating attention**
   - E2B alternates local (sliding window, 512 tokens) and global (full context) layers
   - Need per-layer attention type metadata in the adapter
   - Check if TL generalized attention already supports sliding window (likely does from Gemma 3 support)

3. **Handle shared KV cache**
   - Last `num_kv_shared_layers` layers reuse K/V from the last non-shared layer of the same attention type
   - Need to track which layers source their KV from which other layer
   - May require a custom component or adapter hook

4. **Register in model registry**
   - Add Gemma 4 E2B checkpoint name(s) to TL model registry
   - Handle text-only mode: bypass vision/audio token paths

5. **Basic validation**
   - Forward pass: compare output logits to HuggingFace baseline
   - Target: < 1e-3 mean absolute difference on token logits

### Deliverables
- `gemma4.py` adapter (without PLE)
- Forward pass validation passing

---

## Phase 3: PLE Hook Points
**Duration**: 3–5 days  
**Goal**: PLE exposed as first-class hook points; PLE diagnostic tool

### Background

PLE adds a parallel conditioning pathway at every transformer block:
```
h_layer_out = standard_transformer_output(h) + PLE_residual(ple_vector_for_layer)
```

Where `ple_vector` is computed from: token-identity embedding lookup + learned projection of main embeddings, then cached per layer.

### Tasks

1. **Add PLE hook points** at each layer:
   - `hook_ple_input` — hidden state before PLE modulation
   - `hook_ple_vector` — the PLE conditioning vector for this layer
   - `hook_ple_output` — hidden state after PLE modulation

2. **PLE ablation support**
   - Allow PLE to be zeroed out at any subset of layers via hook intervention
   - Useful for: measuring PLE contribution, isolating standard transformer processing

3. **PLE diagnostic tool** (`notes/ple_analysis.md`)
   - For each layer: compute `||PLE_residual|| / ||total_layer_output||`
   - This answers: what fraction of each layer's output change is attributable to PLE vs. standard processing?
   - Run on a representative text sample; plot per-layer contribution
   - Hypothesis: PLE contribution is larger in earlier layers (where token identity is more salient) and smaller in later layers (where contextual processing dominates)

### Deliverables
- PLE hook points in adapter
- Diagnostic script + initial findings
- `notes/ple_analysis.md` with results

---

## Phase 4: Weight Loading and Validation
**Duration**: 2–3 days  
**Goal**: Robust weight loading; full validation suite

### Tasks

1. **Weight conversion rules**
   - Map all HuggingFace parameter names → TL standardised naming
   - Handle PLE weight shapes (likely different from standard embedding dimensions)
   - Handle shared KV: ensure shared layers point to the correct source parameters

2. **Validation suite**
   - Text-only forward pass (numerical equivalence with HF)
   - Hook point smoke test: confirm hooks fire at correct layers
   - PLE ablation test: zeroing PLE at all layers should degrade but not crash

3. **E4B compatibility check**
   - E4B is a separate, larger variant — check whether the adapter generalises or needs a second config
   - Document scope boundary: "E2B validated; E4B support TBD"

### Deliverables
- Full validation suite passing
- Weight conversion documented

---

## Phase 5: PR and Writeup
**Duration**: 2–3 days  
**Goal**: Open PR; publish findings

### Tasks

1. **PR against #953**
   - Clean branch from TL main
   - Tests for E2B forward pass and hook points
   - Documentation: E2B usage example, PLE hook point API
   - Reference issue #953, note text-only scope

2. **Mensmachina post: "What PLE does — mechanistic interpretability on Gemma 4"**
   - Motivation: novel architecture, no prior mechinterp tooling
   - Method: PLE diagnostic (Phase 3)
   - Findings: per-layer PLE contribution, ablation results
   - Conclusion: what PLE is and isn't doing; implications for using Gemma 4 for interpretability research

3. **Connect to sycophancy investigation**
   - Once adapter is stable, run the multi-turn sycophancy dynamics experiment on Gemma 4 E2B
   - See `notes/sycophancy_brief.md` (cross-reference with mensmachina-web research brief)

### Deliverables
- PR open against #953
- Mensmachina post published or drafted

---

## Open Questions

- Does TL's generalized attention component already handle sliding window (from Gemma 3)? If yes, Phase 2 is faster.
- Are PLE parameters entirely separate from the main embedding table, or are they a projection of it? (affects weight loading complexity)
- How does the shared KV cache interact with activation patching? (patching at a shared-KV layer vs. the source layer are different operations)

---

## Success Criteria

1. `HookedTransformer.from_pretrained("google/gemma-3n-E2B-it")` works with text-only input
2. All standard TL hook points available (residual stream, attention, MLP)
3. PLE hook points available and documented
4. PLE diagnostic published (per-layer contribution figures)
5. PR accepted or under review at #953

---

## Resources

- TransformerLens issue: https://github.com/TransformerLensOrg/TransformerLens/issues/953
- Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4
- Gemma 4 HuggingFace: https://huggingface.co/blog/gemma4
- Sycophancy dynamics research brief: `~/projects/mensmachina-web/src/content/research/sycophancy-dynamics-brief.md`
