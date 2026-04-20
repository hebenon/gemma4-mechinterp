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

## PLE Integration Design

**Approach**: PLE happens between the standard MLP output and `hook_resid_post`. This means `hook_resid_post` continues to represent the final layer output — consistent with all existing TL models. Non-PLE models are unaffected.

```
... mlp_out
→ resid_standard = resid_mid + mlp_out
→ ple_vec = PLE(resid_standard)           # hook_ple_vector applied here
→ resid_post = resid_standard + ple_vec
→ hook_resid_post(resid_post)
→ return resid_post
```

New HookPoints in `TransformerBlock` when `cfg.use_ple=True`:
- `hook_ple_vector` — the PLE conditioning vector `[batch, seq, d_model]`

No `hook_ple_input` needed (it's identical to `resid_standard`, reconstructible from existing hooks). No `hook_ple_output` needed (`hook_resid_post` serves this role).

**Ablation via hooks**: Zero out `hook_ple_vector` to remove PLE from any subset of layers — standard TL hook intervention, nothing special required.

**Config flag**: `use_ple: bool = False` in `HookedTransformerConfig`. Only Gemma 4 sets this True.

## Gemma 3 Weight Conversion Pattern (from `gemma.py`)

- Multimodal detection: `hasattr(gemma, "language_model")`
- Weight name mapping: systematic HF→TL rename + `einops` reshape for head dimensions
- Gemma 3 additions: `use_qk_norm` (loads `q_norm.w`, `k_norm.w`), `use_normalization_before_and_after` for pre/post layernorms
- RMSNorm: all norm weights have 1 added pre-computation (GemmaRMSNorm quirk)

Our additions for Gemma 4:
- Detection: `Gemma4ForCausalLM` / `Gemma4ForConditionalGeneration` (exact names TBD from enumeration)
- PLE weight mapping: TBD from Phase 1 enumeration
- Shared KV: need to determine if activation-sharing (no extra weights) or weight-sharing (one projection, multiple layers) — different handling

## Shared KV — Implementation Question

**Activation sharing** (most likely): shared layers compute Q normally but attend to K/V from a source layer's computation. Requires caching source K/V and routing shared layers to use them. Hook semantics: `hook_k` and `hook_v` at shared layers return the borrowed tensors, not locally computed ones.

**Weight sharing**: one K/V projection matrix referenced by multiple layer instances. Handled at weight loading time — no forward pass change.

Need Phase 1 enumeration to determine which applies.

## Files to Modify in TL Fork

| File | Change |
|------|--------|
| `transformer_lens/HookedTransformerConfig.py` | Add `use_ple`, `num_kv_shared_layers` params |
| `transformer_lens/pretrained/weight_conversions/gemma.py` | Add Gemma 4 detection + PLE weight mapping |
| `transformer_lens/loading_from_pretrained.py` | Add Gemma 4 model entries |
| `transformer_lens/components/transformer_block.py` | Add PLE residual + `hook_ple_vector` |
| `transformer_lens/components/abstract_attention.py` | Shared KV routing (if activation-sharing) |
| `tests/unit/test_gemma4_config.py` | New test file |
