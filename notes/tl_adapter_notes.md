# TransformerLens Adapter Notes — Gemma 4

## TL Hook Mechanism (from source reading)

`HookPoint` is an `nn.Module` identity wrapper placed at key computation points. Adding new hooks is simple: instantiate `HookPoint()` as a class attribute in `__init__`. The `setup()` method on `HookedRootModule` auto-discovers all `HookPoint` instances by scanning named modules — no explicit registration needed.

Hook naming convention: `hook_` prefix + stage name (e.g. `hook_resid_pre`, `hook_attn_out`, `hook_ple_vector`).

## TransformerBlock Residual Flow (from source)

```
hook_resid_pre
  → ln1 → attn → hook_attn_out
  → + resid_pre → hook_resid_mid        (sequential mode)
  → ln2 → mlp → hook_mlp_out
  → + resid_mid → hook_resid_post
  → return resid_post
```

Parallel mode: both attn and mlp read from `resid_pre`, outputs summed into `hook_resid_post`.

## PLE — What It Actually Is (from HF source)

**Two-stage computation in `Gemma4TextModel`:**

**Stage 1 — Token identity component** (computed once, shared across all layers):
```python
embed_tokens_per_layer: Embedding(vocab_size_per_layer_input=262144, num_layers * d_ple = 30*256)
per_layer_inputs = embed_tokens_per_layer(input_ids).reshape(B, L, num_layers, 256)
```

**Stage 2 — Context-aware component** (also computed once from initial embeddings):
```python
per_layer_model_projection: Linear(d_model=2304, num_layers * d_ple = 7680, bias=False)
context_proj = per_layer_model_projection(inputs_embeds) * hidden_size**-0.5
context_proj = context_proj.reshape(B, L, num_layers, 256)
context_proj = per_layer_projection_norm(context_proj)  # RMSNorm

ple_vec = (context_proj + token_identity) * (2**-0.5)  # combined: [B, L, num_layers, 256]
```

**Per decoder layer — gated bottleneck (NOT simple additive residual):**
```python
residual = hidden_states
gate = act_fn(per_layer_input_gate(hidden_states))  # Linear(2304, 256): [B, L, 256]
gated = gate * ple_vec[layer_idx]                   # element-wise: [B, L, 256]
out = per_layer_projection(gated)                    # Linear(256, 2304): [B, L, 2304]
out = post_per_layer_input_norm(out)                 # RMSNorm
hidden_states = residual + out
hidden_states *= layer_scalar                        # learned per-layer scalar
```

**Key insight**: PLE is a **PLE-conditioned bottleneck projection** of the residual stream — not an additive signal. The PLE vector acts as a gate multiplied against hidden_states projected down to d=256. The residual stream modulates what gets gated.

**Parameter accounting**: E2B (30 layers, d=2304, d_ple=256, vocab_ple=262144):
- `embed_tokens_per_layer`: 262144 × 7680 ≈ 2.0B params
- `per_layer_model_projection`: 2304 × 7680 ≈ 0.018B
- Per-layer PLE weights (30 layers): `per_layer_input_gate` (2304×256) + `per_layer_projection` (256×2304) ≈ 2×0.59M × 30 ≈ 0.35B
- Total PLE: ~2.4B → consistent with 5.1B - 2.3B = 2.8B gap

**Mechinterp implication**: PLE's contribution depends on BOTH the PLE vector AND the current hidden state (via the gate projection). This is richer than "token identity signal added per layer" — it's "token identity modulates how much of each dimension of the residual passes through the bottleneck."

**Ablation design revisions**:
- Zero `hook_ple_vector` → removes PLE conditioning entirely (gate × 0 = 0, bottleneck outputs 0)
- Zero `hook_ple_gate` → same effect from the hidden-states side
- Vocabulary-mean `hook_ple_vector` → removes token specificity, preserves scale

## PLE Integration Design (revised)

PLE is a gated bottleneck applied AFTER the standard MLP output, before `hook_resid_post`. `hook_resid_post` continues to be the final layer output (including PLE) — consistent with all TL models.

```
... mlp_out
→ resid_standard = resid_mid + mlp_out
→ ple_vec = hook_ple_input(ple_vecs[layer])            # [B, L, 256] — intervene here to ablate
→ gate = hook_ple_gate(act_fn(W_gate(resid_standard))) # [B, L, 256] — gate activations
→ ple_out = hook_ple_output(W_up(gate * ple_vec))      # [B, L, 2304] — the bottleneck output
→ resid_post = resid_standard + LayerNorm(ple_out)
→ resid_post = hook_resid_post(resid_post * layer_scalar)
```

New HookPoints in `TransformerBlock` when `cfg.use_ple=True`:
- `hook_ple_input` — the PLE conditioning vector for this layer `[batch, seq, d_ple]` (d_ple=256)
- `hook_ple_gate` — the gate activations after act_fn, before elementwise multiply `[batch, seq, d_ple]`
- `hook_ple_output` — the bottleneck output before adding to residual `[batch, seq, d_model]`

New HookPoints in `HookedTransformer.forward()` precomputation when `cfg.use_ple=True`:
- `hook_ple_token_embeds` — token identity component `[batch, seq, n_layers, d_ple]` (from `embed_tokens_per_layer`)
- `hook_ple_context_proj` — context component `[batch, seq, n_layers, d_ple]` (from `per_layer_model_projection` + RMSNorm)

These model-level hooks enable decomposing ple_vec into its two constituents for ablation and attribution experiments.

**Ablation via hooks**:
- Zero `hook_ple_input` at layer L → removes PLE conditioning for that layer (gate × 0 = 0)
- Zero `hook_ple_token_embeds` → replaces token identity with zero, context only
- Zero `hook_ple_context_proj` → replaces context projection with zero, token identity only
- Zero `hook_ple_output` → removes PLE contribution after computation

**PLE precomputation**: PLE vectors are computed ONCE in the model forward pass (Stage 1 + Stage 2 from both `embed_tokens_per_layer` and `per_layer_model_projection`), then passed per-layer. In TL, this precomputation runs before the layer loop — we pass `ple_vecs[layer_idx]` into each `TransformerBlock`.

**Config additions needed**:
- `use_ple: bool = False`
- `d_ple: int = 0` (= hidden_size_per_layer_input)
- `ple_vocab_size: int = 0` (= vocab_size_per_layer_input, 262144)
- `num_kv_shared_layers: int = 0`
- `layer_scalar: bool = False` (per-layer learned scale, new in Gemma 4)

**Attention pattern**: config.layer_types list → "sliding_attention" / "full_attention" with 5:1 ratio. Same as Gemma 3 — existing TL machinery from PR #1149 handles this unchanged.

## Gemma 3 Weight Conversion Pattern (from `gemma.py`)

- Multimodal detection: `hasattr(gemma, "language_model")`
- Weight name mapping: systematic HF→TL rename + `einops` reshape for head dimensions
- Gemma 3 additions: `use_qk_norm` (loads `q_norm.w`, `k_norm.w`), `use_normalization_before_and_after` for pre/post layernorms
- RMSNorm: all norm weights have 1 added pre-computation (GemmaRMSNorm quirk)

Our additions for Gemma 4:
- Detection: `Gemma4ForCausalLM` / `Gemma4ForConditionalGeneration` (exact names TBD from enumeration)
- PLE weight mapping: TBD from Phase 1 enumeration
- Shared KV: need to determine if activation-sharing (no extra weights) or weight-sharing (one projection, multiple layers) — different handling

## Shared KV — Implementation Design (confirmed: activation sharing)

Confirmed from HF source: shared layers skip K/V projection, borrow activations from `shared_kv_cache[kv_shared_layer_index]`. No weight changes needed for shared layers.

**TL implementation (Option A — dict passed through layer loop):**

```python
# HookedTransformer.forward():
shared_kv_cache = {}
for l, block in enumerate(self.blocks):
    resid = block(resid, ple_vec=ple_vecs[:, :, l, :], shared_kv_cache=shared_kv_cache)
    # block stores K/V to shared_kv_cache if it's a source layer
```

```python
# TransformerBlock.forward():
if self.is_kv_shared_layer:
    attn_out = self.attn(resid, shared_kv=shared_kv_cache[self.kv_shared_layer_index])
else:
    attn_out = self.attn(resid)
    if self.store_kv_for_sharing:
        shared_kv_cache[self.layer_idx] = self.attn.last_kv  # or return from attn
```

**Config/block flags needed:**
- `cfg.num_kv_shared_layers: int` — how many terminal layers share KV
- `cfg.first_kv_shared_layer_idx: int` — computed as `n_layers - num_kv_shared_layers`
- Per-block: `block.is_kv_shared_layer: bool`, `block.kv_shared_layer_index: int`, `block.store_kv_for_sharing: bool`

**Hook semantics at shared layers:**
- `hook_k` and `hook_v` at shared layers expose the *borrowed* tensors from the source layer
- Patching `hook_k` at a shared layer patches the copy being used for that layer's attention — does NOT affect the source layer or other shared layers using the same source
- Patching `hook_k` at the *source* layer propagates to all downstream shared layers
- Document this clearly in the API reference

**Files to modify:**
- `HookedTransformerConfig`: add `num_kv_shared_layers`, `first_kv_shared_layer_idx`
- `TransformerBlock.__init__`: set `is_kv_shared_layer`, `kv_shared_layer_index`, `store_kv_for_sharing` from cfg + layer_idx
- `AbstractAttention.forward()`: accept optional `shared_kv` param; when provided, skip K/V projection
- `HookedTransformer.forward()`: manage `shared_kv_cache` dict across blocks

## Files to Modify in TL Fork

| File | Change |
|------|--------|
| `transformer_lens/HookedTransformerConfig.py` | Add `use_ple`, `num_kv_shared_layers` params |
| `transformer_lens/pretrained/weight_conversions/gemma.py` | Add Gemma 4 detection + PLE weight mapping |
| `transformer_lens/loading_from_pretrained.py` | Add Gemma 4 model entries |
| `transformer_lens/components/transformer_block.py` | Add PLE residual + `hook_ple_vector` |
| `transformer_lens/components/abstract_attention.py` | Shared KV routing (if activation-sharing) |
| `tests/unit/test_gemma4_config.py` | New test file |
