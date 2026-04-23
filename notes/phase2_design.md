"""
Phase 2 Design Decisions
-------------------------
Written 2026-04-22 before implementation starts.
These are non-trivial choices that affect the TL API surface.
"""

# Phase 2 Design: Gemma 4 TL Integration

## Problem 1: Dual Head Dimensions

Gemma 4 E2B uses two different head dimensions:
- Local (sliding) attention: d_head = 256
- Global (full) attention:   d_head = 512

TL's `HookedTransformerConfig` has a single `d_head` field. The attention component
uses `cfg.d_head` for all weight reshape operations and forward pass computations.

### Options

**A. Add `d_head_global` to config (recommended)**
- New optional field: `d_head_global: Optional[int] = None`
- When set: local layers use `d_head`, global layers use `d_head_global`
- Weight conversion: einops rearrange uses per-layer head dim based on `layer_type`
- Attention component: check `cfg.d_head_global` if `is_global_attn` else `cfg.d_head`
- Pro: explicit, minimal, backward-compatible (None = same d_head everywhere)
- Con: attention component forward needs a branch

**B. Two attention subclasses: `GemmaLocalAttention` / `GemmaGlobalAttention`**
- Each embeds its d_head as a class attribute
- Pro: clean separation
- Con: larger TL API change; adds new hook points that need coordination

**C. d_head as list (one per layer)**
- `d_head: Union[int, List[int]]`
- Pro: fully general
- Con: significant TL change; breaks many assumptions downstream

**Decision: Option A** — add `d_head_global`. Minimal API surface change, explicit,
handles the Gemma 4 case cleanly. The attention forward branch is a ~3-line addition.

Implementation note: For Q projection at global layers:
  `einops.rearrange(W_Q, "(n h) m -> n m h", n=cfg.n_heads)`
  where h = d_head_global (512) not d_head (256).
  
  Q projection weight is [n_heads × d_head, d_model]:
  - Local:  [8×256, 1536] = [2048, 1536]
  - Global: [8×512, 1536] = [4096, 1536]

---

## Problem 2: Partial RoPE on Global Attention

Global (full) attention uses `partial_rotary_factor = 0.25`:
- Head dim = 512
- Rotary dims = 512 × 0.25 = 128 (positions 0–127 get rotary encoding)
- Non-rotary dims = 384 (positions 128–511 are not rotated)

TL's current RoPE applies rotation to all head dims. The RoPE module computes
cos/sin tensors of shape `[seq, d_head // 2]` and applies them to all dims.

### Options

**A. Add `partial_rotary_factor` / `partial_rotary_factor_global` to config (recommended)**
- New field: `partial_rotary_factor_global: float = 1.0` (default = full rotary)
- RoPE application: split head dims into rotary and non-rotary portions
  - Apply rotation to first `d_head_global × partial_rotary_factor` dims
  - Leave remaining dims unchanged
  - Concatenate
- Pro: explicit, Gemma 4 specific, extensible to future models
- Con: RoPE implementation change affects a core component

**B. Pre-compute different cos/sin for global layers**
- cos/sin shape: `[seq, rotary_dims // 2]` instead of `[seq, d_head // 2]`
- Only apply to the first N dims, leave rest unchanged
- This is essentially Option A with the detail worked out

**Decision: Option A** — add `partial_rotary_factor_global`. The rotary embedding
module (`Gemma4TextRotaryEmbedding`) already dispatches by layer_type (local/global);
TL needs to replicate this dispatch.

**TL implementation detail (confirmed from source):**

TL's `apply_rotary` already slices at `cfg.rotary_dim`:
```python
x_rot = x[..., :self.cfg.rotary_dim]
x_pass = x[..., self.cfg.rotary_dim:]
```
And `cfg.rotary_dim` defaults to `cfg.d_head` in `__post_init__`.

For Gemma 4 global layers, we need a parallel `rotary_dim_global`:
- `rotary_dim_global = round(d_head_global * partial_rotary_factor_global)` = 512 × 0.25 = 128
- Add to `__post_init__`: `if d_head_global and partial_rotary_factor_global: rotary_dim_global = round(d_head_global * partial_rotary_factor_global)`

The attention component for global layers then uses `rotary_dim_global` (128) in the slice
instead of `rotary_dim` (256). The sin/cos for global layers must also be precomputed at
128 dims using `rotary_base` (1e6), not `rotary_base_local` (1e4).

Two sin/cos buffers: `rotary_sin/cos` (local, base=1e4, dim=256) and
`rotary_sin_global/cos_global` (global, base=1e6, dim=128).

---

## Problem 3: Logit Softcapping

`final_logit_softcapping = 30.0`: applied as `logits = tanh(logits / 30.0) * 30.0` before softmax.

**RESOLVED — no new TL code needed.**

TL already has `output_logits_soft_cap: float = -1.0` in `HookedTransformerConfig`. Gemma 2
uses it with the exact same value: `"output_logits_soft_cap": 30.0`. The mechanism is
already wired in `HookedTransformer.forward`:

```python
if self.cfg.output_logits_soft_cap > 0.0:
    logits = self.cfg.output_logits_soft_cap * F.tanh(logits / self.cfg.output_logits_soft_cap)
```

**Action**: Add `"output_logits_soft_cap": 30.0` to `GEMMA4_E2B_CONFIG`. Done.

---

## Problem 4: v_norm

Gemma 4 adds `v_norm: Gemma4RMSNorm` to attention, alongside existing q_norm/k_norm.
TL's attention component already has optional q_norm and k_norm via `cfg.use_qk_norm`.

Extension: add `v_norm` weight loading conditional on `use_qk_norm` (or a new
`use_v_norm` flag). The forward change: apply v_norm to V before attention computation.

Low risk — same pattern as q_norm/k_norm, just one more norm.

---

## Problem 5: layer_scalar

`layer_scalar` is a plain `torch.Tensor` per layer (NOT `nn.Parameter`), applied as
`hidden_states *= layer_scalar` at the end of each decoder layer.

The value is NOT 1.0 — it's trained to values ranging from 0.018 to 0.87 across layers.

**Implementation**: TL's `TransformerBlock.forward` needs a new `layer_scale` optional
parameter, applied after `hook_resid_post` computation but before returning.

Wait — should `hook_resid_post` fire before or after `layer_scalar`?
- If before: `hook_resid_post` captures the pre-scaled residual → ablation at hook point
  doesn't account for scaling → slightly misleading
- If after: `hook_resid_post` captures the final layer output including scaling → consistent
  with TL's convention (hook_resid_post = what the next layer receives)

**Decision: apply layer_scalar BEFORE hook_resid_post.** This matches TL's semantics:
hook_resid_post captures the actual activation that flows to the next layer.

Weight loading: `state_dict[f"blocks.{l}.layer_scale"] = layer.layer_scalar`
(plain tensor, no +1 convention needed — not an RMSNorm weight)

---

## Problem 6: Shared KV Cache

Last 20 layers (15–34) borrow K/V from source layers instead of computing their own:
- Shared sliding layers borrow from **layer 13** (last sliding source)
- Shared full/global layers borrow from **layer 14** (last full source)

This is activation-level sharing — no weight changes, but the forward pass must route
K/V from source to borrowing layers.

### Options

**A. Model-level KV pass-through (recommended)**
- HookedTransformer.forward maintains a `kv_store: dict[int, tuple[K, V]]`
- At source layers (13, 14): store K/V after computation
- At borrowing layers (15–34): skip K/V computation, retrieve from store
- TransformerBlock.forward and AbstractAttention.forward both take `cached_kv=None`
- Pro: clean, explicit, no hidden state; cacheable; works with TL's existing KV cache API
- Con: API change to TransformerBlock.forward; HookedTransformer.forward needs to manage store

**B. Direct module reference**
- Borrowing attention component holds a reference to source attention component
- On forward: if `self.kv_source is not None`, call source's `calculate_kv()` instead
- Pro: self-contained per-block
- Con: circular module references; incompatible with TL's module structure; hooks at
  borrowing layer would capture stale K/V

**C. Config-driven source lookup in HookedTransformer**
- `cfg.kv_shared_layer_sources: dict[int, int]` maps borrowing → source
  e.g. {15: 13, 16: 13, ..., 19: 14, ...}
- HookedTransformer caches K/V at source layers and routes during forward
- Variant of Option A but the mapping is in config (not computed at runtime)
- This is cleaner than A: the routing table is explicit and inspectable

**Decision: Option C** — config-driven source lookup.

Implementation:
1. Add `kv_shared_layer_sources: Optional[dict[int, int]] = None` to config
   Built as: `{l: 13 for l in range(15, 35) if attn_types[l] == 'local'} | {l: 14 for l in range(15, 35) if attn_types[l] == 'global'}`
2. AbstractAttention.forward gets `cached_kv: Optional[tuple[K, V]] = None`
   If not None: skip K/V projection, use cached K/V directly
3. HookedTransformer.forward: track K/V at source layers, pass cached_kv to borrowing layers
4. Hook semantics: `hook_k` / `hook_v` at borrowing layers will fire with BORROWED values —
   this is correct behavior (it's what those layers actually use). Document clearly.

Note: `first_kv_shared_layer_idx = 15` and source lookup can be derived from `attn_types` + 
`num_kv_shared_layers`, so these don't need to be stored explicitly in config — compute in `__post_init__`.

---

## Implementation Order

Phase 2 (without PLE, establishing baseline):
1. Config: add `d_head_global`, `rotary_dim_global`, `kv_shared_layer_sources`
   (output_logits_soft_cap already exists — just set 30.0 in model config entry)
2. Weight conversion: Gemma 4 detection, per-layer d_head in biases, v_norm, layer_scalar
3. Attention: dual d_head (bias shape), partial RoPE (rotary_dim_global + new sin/cos buffers), v_norm, cached_kv
4. TransformerBlock: layer_scalar before hook_resid_post
5. HookedTransformer.forward: kv_store for shared KV routing
6. Validate: forward pass logit MAE < 1e-3 (generation) + residual stream < 1e-5

Phase 3 (PLE):
1. PLEPrecomputer module (model-level): embed_tokens_per_layer, per_layer_model_projection,
   per_layer_projection_norm, scale factors
2. Hook points: hook_ple_token_embeds, hook_ple_context_proj (PLEPrecomputer level)
3. TransformerBlock PLE: hook_ple_input, hook_ple_gate, hook_ple_output
4. PLE weight loading in convert_gemma4_weights
5. Validate: with PLE, residual stream matches HF
