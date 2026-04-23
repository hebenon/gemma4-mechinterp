# Project Plan: Gemma 4 TransformerLens Support + Mechinterp

**Started**: 2026-04-20  
**Target**: New PR against TransformerLens (Gemma 4 support + PLE hook points) + initial PLE diagnostic findings  
**Estimated duration**: 1.5–2 weeks  

---

## Context

Gemma 3 was added to TransformerLens in PR #1149 (merged 2026-01-15). That PR introduced the key machinery: alternating local/global attention, per-layer RoPE bases, Q/K normalization, and text-only weight extraction for multimodal models. The implementation follows an architecture-detection pattern (HF class name → config and weight conversion logic) rather than a full adapter class.

Gemma 4 is a separate, newer model family (E2B 2.3B effective, E4B 4.5B effective, 31B dense, 26B MoE). It shares Gemma 3's attention pattern but adds two novel elements not present in Gemma 3:

1. **Per-Layer Embeddings (PLE)** — parallel conditioning pathway at every transformer block: `h_out = standard_output + PLE_residual(ple_vec)`. Not in any TL model.
2. **Shared KV cache** — last N layers reuse K/V from an earlier source layer. Not in TL.

Our contribution framing: **Gemma 4 support + PLE hook points as a new TL hook mechanism**. The PLE hook points are a new TL capability, not just model support — worth calling out in the PR.

Issue #953 is for Gemma 3n (an older related model). We will open a new issue for Gemma 4 and reference #953 as prior context.

---

## Implementation Pattern (from PR #1149)

Follow this sequence — same as #1149:

1. **Config params first** — add new fields to `HookedTransformerConfig.py` (e.g. `use_ple`, `num_kv_shared_layers`)
2. **Weight conversion** — extend `weight_conversions/gemma.py` with Gemma 4 detection and PLE weight mappings
3. **Architecture detection** — update `loading_from_pretrained.py` to detect `Gemma4ForCausalLM` / `Gemma4ForConditionalGeneration`
4. **Component changes** — PLE in a new component or in `abstract_attention.py`; shared KV routing
5. **Tests** — unit tests for config, forward pass equivalence, hook point smoke test

Keep commits logically separated by concern (config / weight conversion / component / tests). #1149 used 46 commits over 37 days; we should be leaner since we're extending existing code.

---

## Phase 1: Environment and Architecture Mapping
**Duration**: 2–3 days  
**Goal**: Understand the implementation surface before writing code

### Tasks

1. **Fork TransformerLens** and set up development environment
   - Clone fork locally
   - Read `weight_conversions/gemma.py` and `loading_from_pretrained.py` as they stand post-#1149
   - Understand the Gemma 3 detection and weight mapping pattern — this is what we extend
   - Study TL's hook mechanism (`HookPoint`, how hooks are registered per component) to assess the most idiomatic pattern for PLE: first-class hook category or model-specific hooks

2. **Load Gemma 4 E2B via nnsight** to enumerate module structure
   - `model.named_modules()` — get full module tree
   - Identify: attention layers (which are local? which global? what pattern?), MLP layers, layernorm layers, PLE modules, shared KV layers
   - Note exact HuggingFace class names for Gemma4 (needed for detection logic)

3. **Document architecture map** (`notes/architecture.md`)
   - Full module hierarchy for E2B
   - Per-layer type table: layer index → {attention type (local/global), shares KV?, PLE present?}
   - PLE module interface: inputs, outputs, parameter shapes, parameter names in HF state dict
   - Shared KV config: which layers are source, which borrow, and what the config field is called

### Deliverables
- `notes/architecture.md` — complete E2B module map
- TL fork set up locally

---

## Phase 2: Core Extension (without PLE)
**Duration**: 2–3 days (reduced from 5–7: we extend gemma.py, not build from scratch)  
**Goal**: Forward pass working for Gemma 4 E2B, matching HF output

### Tasks

1. **Extend `weight_conversions/gemma.py`**
   - Add `Gemma4ForCausalLM` / `Gemma4ForConditionalGeneration` detection
   - Add weight name mappings for Gemma 4 (most will be identical to Gemma 3; note differences)
   - Text-only extraction logic already exists — adapt for Gemma 4 multimodal weight layout

2. **Config entries** in `loading_from_pretrained.py` and `HookedTransformerConfig.py`
   - Add Gemma 4 E2B checkpoint name(s)
   - New config params: `num_kv_shared_layers` (shared KV), `use_ple` (flag)
   - Attention pattern: confirm local/global ratio for E2B (different from Gemma 3's 5:1)

3. **Shared KV cache**
   - Last `num_kv_shared_layers` layers reuse K/V from source layer of same attention type
   - Implement routing logic — shared layers should not compute their own K/V, just reference source
   - This is novel to TL; may require a small addition to attention component

4. **Basic validation** (run on Colab T4 — Pi cannot load E2B)
   - Forward pass: compare output logits to HuggingFace baseline (text-only)
   - Target: < 1e-3 logit MAE; also check < 1e-5 on residual stream tensors at selected layers
   - Confirm tokenizer text-only mode does not accidentally trigger multimodal paths via reserved token IDs

### Deliverables
- Extended `gemma.py` with Gemma 4 detection and weight mapping
- Forward pass validation passing (PLE not yet included)

---

## Phase 3: PLE Hook Points
**Duration**: 3–5 days  
**Goal**: PLE exposed as first-class hook points; PLE diagnostic

### Background

PLE adds a parallel conditioning pathway at every transformer block:
```
h_layer_out = standard_transformer_output(h) + PLE_residual(ple_vector_for_layer)
```

Where `ple_vector` is computed from: token-identity embedding lookup + learned projection, cached per layer. This is not present in any existing TL model — it is a new hook mechanism, not just a new model.

### Tasks

1. **Add PLE hook points** at each layer:
   - `hook_ple_input` — hidden state before PLE modulation
   - `hook_ple_vector` — the PLE conditioning vector for this layer
   - `hook_ple_output` — hidden state after PLE modulation

2. **PLE ablation support**
   - Zero-out PLE at any subset of layers via hook intervention
   - Useful for: measuring PLE contribution, isolating standard transformer processing

### Deliverables
- PLE hook points implemented and documented
- Tests: hooks fire at correct layers, ablation (zero PLE at all layers) degrades but does not crash

---

## Phase 4: Validation
**Duration**: 1–2 days (lighter: weight loading pattern already established in Phase 2)  
**Goal**: Full validation suite; E4B compatibility check

### Tasks

1. **Validation suite**
   - Text-only forward pass (numerical equivalence with HF, with PLE included)
   - Hook point smoke test: confirm all hooks fire at correct layers
   - PLE ablation test: zeroing PLE at all layers degrades but does not crash
   - Shared KV test: K/V tensors at shared layers match source layer

2. **E4B compatibility check**
   - Check whether config generalises to E4B or needs a second entry
   - Document scope: "E2B validated; E4B support follows same pattern"

### Deliverables
- Full validation suite passing
- E4B compatibility documented

---

## Phase 5: PR and Writeup
**Duration**: 2–3 days  
**Goal**: Open PR; publish findings

### Tasks

1. **Open new TL issue** for Gemma 4 support (reference #953 as prior context for Gemma 3n)

2. **PR against TransformerLens main**
   - Clean branch from TL main
   - Commit structure mirrors #1149: config → weight conversion → component → tests
   - Framing: "Gemma 4 support + PLE hook points" — two contributions, not one
   - Tests: config, forward pass equivalence, hook point smoke test, PLE ablation
   - Documentation: E2B usage example, PLE hook point API reference

3. **PLE diagnostic** (`notes/ple_analysis.md`) — separate from hook implementation
   - Per layer: compute `||PLE_residual|| / ||total_layer_output||`
   - Run on representative text; plot per-layer contribution
   - Hypothesis: PLE contribution larger in early layers (token identity), smaller in later (context dominates)

4. **Mensmachina post: "What PLE does — mechanistic interpretability on Gemma 4"**
   - Method: PLE diagnostic (above)
   - Findings: per-layer PLE contribution, ablation results
   - Conclusion: what PLE is and isn't doing; implications for Gemma 4 interpretability research

### Deliverables
- TL issue open for Gemma 4
- PR open against TL main
- Mensmachina post published or drafted

---

## Open Questions

**Resolved from HF source reading (2026-04-20):**
- ~~HF class names~~ → `Gemma4ForCausalLM` (text-only), `Gemma4TextConfig`
- ~~Attention pattern~~ → 5:1 sliding/full, same as Gemma 3; `config.layer_types` list; existing TL machinery handles it
- ~~Shared KV: weight vs activation sharing~~ → **activation sharing** confirmed. Layers past `first_kv_shared_layer_idx` skip K/V computation and borrow from `kv_shared_layer_index`
- ~~PLE hook shape~~ → gated bottleneck: Linear(2304→256) × ple_vec → Linear(256→2304). NOT a simple additive residual.
- ~~TL hook mechanism~~ → `HookPoint()` as class attribute, `setup()` auto-discovers. Pattern settled in notes.

**Still open (need enumeration notebook):**
- What is `num_kv_shared_layers` for E2B specifically? (config default is 0 but E2B likely ~17)
- What is the exact `attn_types` list for E2B? TL field is `attn_types` (["local","global"]), not `layer_types`. Gemma 3 uses 5:1; DeepMind JAX says 4:1 for Gemma 4. Confirm from enumeration.
- ~~What activation function is used in `per_layer_input_gate`?~~ **GELU confirmed** from DeepMind JAX: `gelu_pytorch_tanh`
- Minimum `transformers` version for `Gemma4ForCausalLM` to exist
- Exact `rotary_base_local` for E2B (expect 10000, same as Gemma 3 — confirm from enumeration RoPE cell)
- How does PLE precomputation device placement interact with TL's hook tensor device assumptions?
- How does shared KV interact with activation patching? (patching a shared-KV layer vs. source layer — document clearly)
- MatFormer: does HF export a pre-sliced checkpoint for E2B, or must we slice the full 5.1B at load time?

## Known Risks

- **PLE hook shape**: ~~Resolved~~ — PLE is a gated bottleneck (Linear 1536→256, gate × ple_vec, Linear 256→1536). Hook design: `hook_ple_input` [B,L,256] and `hook_ple_output` [B,L,1536]. See `notes/tl_adapter_notes.md`.
- **Two head dimensions**: Local attention uses `d_head=256`, full/global attention uses `d_head=512`. TL's config has a single `d_head` field. Options: (a) extend config with `d_head_global`; (b) always use the larger and pad; (c) handle in weight conversion per-layer. This needs a design decision before Phase 2.
- **Partial RoPE on global attention**: Full attention layers use `partial_rotary_factor=0.25` (only 128/512 dims get RoPE, `rope_type="proportional"`). TL's current RoPE implementation applies rotary to the full head. New per-attention-type RoPE handling required.
- **v_norm (new QKV norm)**: Gemma 4 adds `v_norm` alongside `q_norm`/`k_norm`. Confirmed from named_modules. TL's attention component only has q_norm and k_norm — needs extension. Low complexity but a PR change.
- **`final_logit_softcapping=30.0`**: Logits are capped via `tanh(logit/30)*30` before softmax. Not in TL. Needs a new config field and hook in the unembed path. Precedent: Gemma 2 may have this (check PR #1149).
- **Execution environment**: E2B has 5.1B raw parameters. Validation cannot run on the Pi — use Kaggle T4 (free tier, no HF token required) for all Phase 2+ runs.
- **Numerical threshold**: 1e-3 logit MAE is adequate for generation parity but insufficient for mechinterp. Target 1e-5 on residual stream tensors in addition to the logit check.
- **Upstream coordination**: Adding PLE as a new hook mechanism is an API surface change. We will implement first in the most idiomatic way we can assess, prove it works, then engage maintainers. Accepted risk of post-implementation redesign feedback.
- **Timeline**: 1.5–2 weeks is optimistic given two-head-dim problem and partial RoPE. 3–4 weeks is a safer estimate to a mergeable PR.

---

## Success Criteria

1. `HookedTransformer.from_pretrained("google/gemma-4-E2B-it")` works with text-only input
2. All standard TL hook points available (residual stream, attention, MLP)
3. PLE hook points available and documented
4. Shared KV cache handled correctly
5. PLE diagnostic published (per-layer contribution figures)
6. PR open or merged against TL main

---

## Resources

- PR #1149 (Gemma 3 support — our template): https://github.com/TransformerLensOrg/TransformerLens/pull/1149
- Issue #953 (Gemma 3n — prior context): https://github.com/TransformerLensOrg/TransformerLens/issues/953
- Gemma 4 HuggingFace blog: https://huggingface.co/blog/gemma4
- Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4
- Sycophancy dynamics research brief: `~/projects/mensmachina-web/src/content/research/sycophancy-dynamics-brief.md`
