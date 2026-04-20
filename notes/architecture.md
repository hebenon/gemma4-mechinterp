# Gemma 4 E2B Architecture Notes

*To be filled during Phase 1 — module enumeration via nnsight*

## Model Variants

| Variant | Effective Params | Total Params | Context | Notes |
|---------|-----------------|--------------|---------|-------|
| E2B | 2.3B | 5.1B (with embeddings) | 128K | PLE-heavy |
| E4B | 4.5B | 8B (with embeddings) | 128K | Separate variant |

## Key Architectural Elements

### Per-Layer Embeddings (PLE)
- Parallel conditioning pathway alongside main residual stream
- Each decoder layer receives: token-identity component (embedding lookup) + context-aware component (learned projection)
- Applied as residual block after attention and FFN: `h_out = h_standard + PLE_residual(ple_vec)`
- PLE parameters are partially CPU-offloaded in inference (memory management consideration)
- For multimodal inputs: PLE uses pad token ID before soft token merge

### Alternating Attention
- Alternates between local (sliding window) and global (full context) attention layers
- Local: 512-token sliding window
- Global: full context up to 128K
- Final layer: always global
- RoPE config differs: standard RoPE for local, pruned/proportional RoPE for global

### Shared KV Cache
- Last `num_kv_shared_layers` layers reuse K/V from last non-shared layer of same attention type
- Reduces compute and memory with minimal quality impact
- Need to track: which layers are "source" KV layers vs. "borrowing" layers

### Vocabulary
- Large vocabulary with reserved multimodal token IDs
- Text-only mode: bypass vision/audio token paths; use only text token IDs

## Module Map (E2B)

*Fill from `model.named_modules()` enumeration — Phase 1 task*

```
# Placeholder — to be filled
model.
  embed_tokens        → token embeddings
  layers[i].          → transformer block i
    self_attn         → attention (local or global depending on i)
    mlp               → feed-forward
    input_layernorm   → pre-attention RMSNorm
    post_feedforward_layernorm → post-FF RMSNorm
    ple_...           → PLE module (exact name TBD)
  norm                → final RMSNorm
  lm_head             → output projection
```

## Per-Layer Type Table

*Fill from architecture inspection — Phase 1 task*

| Layer | Attention Type | Shares KV? | Has PLE? |
|-------|---------------|------------|----------|
| 0 | local | no | yes |
| ... | ... | ... | ... |

## Open Questions

- What are the exact HuggingFace class names for PLE modules?
- Are PLE parameters in the same checkpoint file or separate?
- How is `num_kv_shared_layers` configured for E2B?
- Does the sliding window attention use a standard causal mask within the window?
