# Gemma 4 E2B Architecture Notes

*Partially filled from HuggingFace source reading (2026-04-20). Remaining gaps need enumeration notebook output.*

## Model Variants

| Variant | Effective Params | Total Params | Context | Notes |
|---------|-----------------|--------------|---------|-------|
| E2B | 2.3B | 5.1B (with embeddings) | 128K | 30L, d=2304, d_ple=256 |
| E4B | 4.5B | 8B (with embeddings) | 128K | Separate variant |

## HuggingFace Class Names (confirmed from source)

- Text-only: `Gemma4ForCausalLM`
- Multimodal: `Gemma4Model` (no LM head) / `Gemma4ForConditionalGeneration` (TBC)
- Config: `Gemma4TextConfig`
- Decoder layer: `Gemma4TextDecoderLayer`
- Attention: `Gemma4TextAttention`

## Configuration Defaults (from `configuration_gemma4.py`)

| Field | Value | Notes |
|-------|-------|-------|
| `num_hidden_layers` | 30 | E2B default |
| `hidden_size` | 2304 | d_model |
| `num_attention_heads` | 8 | |
| `num_key_value_heads` | 4 | GQA |
| `sliding_window` | 512 | local attention window |
| `hidden_size_per_layer_input` | 256 | d_ple |
| `vocab_size_per_layer_input` | 262144 | PLE vocab |
| `num_kv_shared_layers` | 0 | **E2B value TBD** |
| `layer_types` | auto-generated | 5:1 sliding/full ratio |

## Key Architectural Elements

### Per-Layer Embeddings (PLE)

**Precomputed once per forward pass:**

```
embed_tokens_per_layer: Embedding(262144, 30 × 256 = 7680)
  → lookup by input_ids → reshape [B, L, 30, 256]   (token identity)

per_layer_model_projection: Linear(2304, 7680, bias=False)
  → project inputs_embeds → scale × reshape → RMSNorm → [B, L, 30, 256]  (context)

ple_vecs = (context + token_identity) × (1/√2)        [B, L, 30, 256]
```

**Applied per decoder layer (gated bottleneck):**

```python
residual = hidden_states                              # [B, L, 2304]
gate = act_fn(per_layer_input_gate(hidden_states))    # Linear(2304→256)
gated = gate * ple_vecs[:, :, layer_idx, :]           # [B, L, 256]
out = per_layer_projection(gated)                     # Linear(256→2304)
out = post_per_layer_input_norm(out)                  # RMSNorm
hidden_states = residual + out
hidden_states *= layer_scalar                         # learned per-layer scale
```

PLE is NOT a simple additive residual. It's a gated bottleneck: the PLE vector gates the projection of the residual stream through a 256-dim bottleneck.

**Parameter budget (approx):**
- `embed_tokens_per_layer`: 262144 × 7680 ≈ 2.01B
- `per_layer_model_projection`: 2304 × 7680 ≈ 17.7M
- Per-layer (30×): `per_layer_input_gate` (2304×256) + `per_layer_projection` (256×2304) ≈ 35.4M
- Total PLE: ~2.06B (rest of the 2.8B gap is likely vision encoder + other embeddings)

### Alternating Attention

- Pattern: `config.layer_types` list, 5:1 sliding/full ratio by default
- `self.is_sliding = (layer_type == "sliding_attention")`
- Same pattern as Gemma 3 — TL machinery from PR #1149 already handles this
- Sliding window: 512 tokens
- Causal mask: `create_sliding_window_causal_mask` for sliding, `create_causal_mask` for full

### Shared KV Cache (activation sharing, not weight sharing)

```python
# Source layers store K/V:
if self.store_full_length_kv:
    shared_kv_states[self.layer_idx] = (key_states, value_states)

# Borrowing layers skip K/V computation:
if self.is_kv_shared_layer:
    key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
```

- `first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers`
- Layers `>= first_kv_shared_layer_idx` are borrowing layers
- `kv_shared_layer_index`: last non-shared layer of same attention type
- **E2B `num_kv_shared_layers` value: TBD from enumeration notebook**

### Layer Scalar

Each decoder layer has a learned scalar `layer_scalar` applied at the end:
`hidden_states *= layer_scalar`

This is new in Gemma 4. Needs to be included in weight loading.

## Module Map (E2B)

*Confirmed names from HF source; exact instance names need enumeration notebook*

```
Gemma4ForCausalLM
  model: Gemma4TextModel
    embed_tokens                     → standard token embedding
    embed_tokens_per_layer           → PLE token embedding [262144, 7680]
    per_layer_model_projection       → PLE context projection Linear(2304, 7680)
    per_layer_projection_norm        → PLE context RMSNorm(256)
    layers[i]: Gemma4TextDecoderLayer
      self_attn: Gemma4TextAttention  → sliding or full, per layer_types[i]
      mlp                             → SwiGLU or GeLU FFN (TBD)
      input_layernorm                 → RMSNorm pre-attention
      post_feedforward_layernorm      → RMSNorm post-FFN
      per_layer_input_gate            → Linear(2304, 256)
      per_layer_projection            → Linear(256, 2304)
      post_per_layer_input_norm       → RMSNorm(2304) post-PLE
      layer_scalar                    → learnable scalar
    norm                             → final RMSNorm
  lm_head                           → output projection
```

## Per-Layer Type Table

*layer_types values TBD — 5:1 sliding/full ratio, 30 layers total*

| Layer | Attention Type | is_kv_shared? | Notes |
|-------|---------------|---------------|-------|
| 0 | sliding | no | |
| ... | ... | ... | |
| 24+ | TBD | TBD | shared KV region |
| 29 | full | no | likely always global |

*Fill exact values from enumeration notebook.*

## Open Questions (remaining)

- What is the exact `num_kv_shared_layers` for E2B? (config default is 0 but E2B likely non-zero)
- What is the exact `layer_types` list for E2B? (5:1 ratio with 30 layers → need exact pattern)
- What activation function is used in `per_layer_input_gate`? (`act_fn` in the source)
- Are PLE parameters in the same checkpoint file as the main weights?
- What is `post_per_layer_input_norm` norming — d=2304 or d=256? (appears to be 2304 from position)
