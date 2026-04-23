# Gemma 4 E2B Architecture Notes

*Updated 2026-04-22 from config.json enumeration (Kaggle run).*

## Model Variants

| Variant | Effective Params | Total Params | Context | Notes |
|---------|-----------------|--------------|---------|-------|
| E2B | 2.3B | 5.1B (with embeddings) | 128K | **35L, d=1536**, d_ple=256 |
| E4B | 4.5B | 8B (with embeddings) | 128K | Separate variant |

## HuggingFace Class Names (confirmed from enumeration)

- **Kaggle model (multimodal)**: `Gemma4ForConditionalGeneration`
- **Config class**: `Gemma4Config` (outer) / text sub-config: `model_type: "gemma4_text"`
- Text-only variant may be `Gemma4ForCausalLM` — not confirmed in Kaggle model
- For TL weight extraction: text model is at `model.language_model.model` (same as Gemma 3 multimodal pattern)

## Configuration (from config.json enumeration 2026-04-22)

### Text Config (`text_config` sub-object)

| Field | Value | Notes |
|-------|-------|-------|
| `num_hidden_layers` | **35** | Was wrong in notes (had 30) |
| `hidden_size` | **1536** | d_model. Was wrong in notes (had 2304) |
| `head_dim` | **256** | Local attention head dim |
| `global_head_dim` | **512** | **Full attention uses DIFFERENT head dim** |
| `num_attention_heads` | 8 | |
| `num_key_value_heads` | **1** | Very aggressive GQA (1 KV head for local attn) |
| `num_global_key_value_heads` | null | Full attention KV config TBD from model load |
| `intermediate_size` | **6144** | d_mlp |
| `sliding_window` | 512 | |
| `hidden_size_per_layer_input` | 256 | d_ple ✓ |
| `vocab_size_per_layer_input` | 262144 | PLE vocab ✓ |
| `num_kv_shared_layers` | **20** | First 15 are source layers; last 20 share KV |
| `vocab_size` | 262144 | ✓ |
| `hidden_activation` | `gelu_pytorch_tanh` | MLP activation (gate + up) |
| `final_logit_softcapping` | 30.0 | **NEW** — logit capped at tanh(x/30)*30 |
| `tie_word_embeddings` | true | lm_head.weight = embed_tokens.weight |
| `use_double_wide_mlp` | true | d_mlp = 2 × standard SwiGLU size |
| `rms_norm_eps` | 1e-06 | |
| `max_position_embeddings` | 131072 | 128K context |

### RoPE Config (per attention type)

| Attention type | rope_theta | rope_type | partial_rotary_factor |
|---------------|-----------|-----------|----------------------|
| `sliding_attention` | 10000.0 | "default" | — (full rotary) |
| `full_attention` | 1000000.0 | "proportional" | **0.25** |

Full attention uses **partial RoPE**: only 25% of head dimensions (512 × 0.25 = 128 dims) get rotary encoding. This is new and not handled by TL's existing Gemma 3 machinery.

### Layer Types (35 layers, 4:1 sliding:full)

Pattern: 7 repetitions of [sliding×4, full×1].

```
sliding_attention  ×4  }
full_attention     ×1  } × 7  =  35 layers
```

Full attention (global) at indices: **4, 9, 14, 19, 24, 29, 34**
Sliding attention (local) at all other indices.

**4:1 ratio confirmed** — DeepMind JAX was right, HF config default (5:1) was for a different variant.

### Shared KV Cache

- `num_kv_shared_layers = 20`
- `first_kv_shared_layer_idx = 35 - 20 = 15`
- Layers 0–14: **source layers** (compute and store K/V)
- Layers 15–34: **borrowing layers** (reuse K/V from last source of same attention type)

Source layers by type:
- **Sliding source**: 0,1,2,3, 5,6,7,8, 10,11,12,13 → last = 13
- **Full source**: 4, 9, 14 → last = 14

All shared sliding layers (15–34, non-full) borrow from **layer 13**.
All shared full layers (19, 24, 29, 34) borrow from **layer 14**.

## Key Architectural Elements

### Per-Layer Embeddings (PLE)

**Precomputed once per forward pass:**

```
embed_tokens_per_layer: Embedding(262144, 35 × 256 = 8960)
  → lookup by input_ids → reshape [B, L, 35, 256]   (token identity)

per_layer_model_projection: Linear(1536, 8960, bias=False)
  → project inputs_embeds → scale × reshape → RMSNorm → [B, L, 35, 256]  (context)

ple_vecs = (context + token_identity) × (1/√2)        [B, L, 35, 256]
```

**Applied per decoder layer (gated bottleneck):**

```python
residual = hidden_states                              # [B, L, 1536]
gate = gelu(per_layer_input_gate(hidden_states))      # Linear(1536→256), GELU confirmed
gated = gate * ple_vecs[:, :, layer_idx, :]           # [B, L, 256]
out = per_layer_projection(gated)                     # Linear(256→1536)
out = post_per_layer_input_norm(out)                  # RMSNorm(1536)
hidden_states = residual + out
hidden_states *= layer_scalar                         # learned per-layer scale
```

**Parameter budget (updated):**
- `embed_tokens_per_layer`: 262144 × 8960 ≈ **2.35B** (bulk of total params)
- `per_layer_model_projection`: 1536 × 8960 ≈ 13.8M
- Per-layer (35×): `per_layer_input_gate` (1536×256) + `per_layer_projection` (256×1536) ≈ 27.5M
- Total PLE: ~2.39B

### Attention Architecture

**Two distinct head dimensions:**
- Local (sliding) attention: `d_head = 256`, `n_kv_heads = 1`
- Global (full) attention: `d_head = 512`, `n_kv_heads = ?` (null in config — TBD from model load)

Q projection shapes:
- Local: `[n_heads × d_head, d_model]` = `[8×256, 1536]` = `[2048, 1536]`
- Global: `[n_heads × global_head_dim, d_model]` = `[8×512, 1536]` = `[4096, 1536]`

**Partial RoPE on global attention**: `partial_rotary_factor=0.25` → only 128/512 dims get rotary encoding. This requires new handling in TL (Gemma 3 does not have partial RoPE on global layers).

Mechinterp note: 384/512 global head dims are position-free. At 131K context, global attention treats content identity as a stronger signal than location. The positional encoding is present but deliberately minor at full-context scale.

**Logit softcapping**: `final_logit_softcapping=30.0` — applied to final logits as `tanh(logit/30)*30`. TL already has `output_logits_soft_cap` field (used by Gemma 2 with same value); zero new code needed.

### Layer Scalar

Each decoder layer has a learned scalar applied at end: `hidden_states *= layer_scalar`.
Initialized to 1.0 during training. Introduces per-layer heterogeneity in output magnitude.

## Module Map (E2B — confirmed from named_modules + forward source 2026-04-22)

```
Gemma4ForConditionalGeneration           ← model
  model: Gemma4Model                     ← model.model
    language_model: Gemma4TextModel      ← model.model.language_model = base_model
      embed_tokens: Gemma4TextScaledWordEmbedding   → [262144, 1536], scaled by sqrt(d_model)
      embed_tokens_per_layer: Embedding             → [262144, 35×256 = 8960]  (PLE token identity)
      per_layer_model_projection: Linear            → Linear(1536, 8960, bias=False)  (PLE context)
      per_layer_projection_norm: Gemma4RMSNorm      → RMSNorm(256)  (applied after reshape to [B,L,35,256])
      rotary_emb: Gemma4TextRotaryEmbedding         → handles both local+global RoPE, dispatched by layer_type
      layers[i]: Gemma4TextDecoderLayer
        self_attn: Gemma4TextAttention
          q_proj, k_proj, v_proj, o_proj: Linear
          q_norm, k_norm, v_norm: Gemma4RMSNorm     ← v_norm new in Gemma 4 (not in Gemma 3)
        mlp: Gemma4TextMLP
          gate_proj, up_proj, down_proj: Linear
          act_fn: GELUTanh                          ← MLP activation
        input_layernorm: Gemma4RMSNorm              ← pre-attention
        post_attention_layernorm: Gemma4RMSNorm     ← post-attention (before residual add)
        pre_feedforward_layernorm: Gemma4RMSNorm    ← pre-FFN
        post_feedforward_layernorm: Gemma4RMSNorm   ← post-FFN (before residual add)
        act_fn: GELUTanh                            ← PLE gate activation (layer-level)
        per_layer_input_gate: Linear                → Linear(1536, 256)
        per_layer_projection: Linear                → Linear(256, 1536)
        post_per_layer_input_norm: Gemma4RMSNorm    → RMSNorm(1536) post-PLE
        layer_scalar: torch.Tensor                  ← plain tensor (NOT nn.Parameter), NOT in named_children
      norm: Gemma4RMSNorm                           → final RMSNorm
    lm_head                                         → tied to embed_tokens (tie_word_embeddings=True)
    vision_tower: Gemma4VisionModel                 (skip for text-only TL)
    embed_vision: Gemma4MultimodalEmbedder          (skip)
    audio_tower: Gemma4AudioModel                   (skip)
    embed_audio: Gemma4MultimodalEmbedder           (skip)
```

### PLE Precomputation (confirmed from forward source)

```python
# Step 1: get_per_layer_inputs — token identity component
per_layer_inputs = base_model.embed_tokens_per_layer(input_ids)
per_layer_inputs = per_layer_inputs.reshape(B, L, n_layers, d_ple)   # [B, L, 35, 256]

# Step 2: project_per_layer_inputs — adds context component
context = base_model.per_layer_model_projection(inputs_embeds)        # [B, L, 35*256]
context = context * base_model.per_layer_model_projection_scale       # scale (≈ hidden_size**-0.5)
context = context.reshape(B, L, n_layers, d_ple)                      # [B, L, 35, 256]
context = base_model.per_layer_projection_norm(context)               # RMSNorm(256)
per_layer_inputs = (context + per_layer_inputs) * base_model.per_layer_input_scale  # ≈ 2**-0.5
```

Two scale factors — **confirmed plain Python floats, not learned** (2026-04-22):
- `per_layer_model_projection_scale` = `0.02551551815399144` = `1/√1536` (hidden_size**-0.5)
- `per_layer_input_scale` = `0.7071067811865476` = `1/√2`

Bake as constants in TL PLEPrecomputer.forward — no weight loading needed.

### PLE Per-Layer Forward (confirmed from decoder layer forward source)

```python
residual = hidden_states
hidden_states = layer.per_layer_input_gate(hidden_states)   # [B, L, 1536] → [B, L, 256]
hidden_states = layer.act_fn(hidden_states)                  # GELUTanh
hidden_states = hidden_states * per_layer_input              # [B, L, 256] × per_layer_inputs[:,:,i,:]
hidden_states = layer.per_layer_projection(hidden_states)    # [B, L, 256] → [B, L, 1536]
hidden_states = layer.post_per_layer_input_norm(hidden_states)  # RMSNorm(1536)
hidden_states = residual + hidden_states
hidden_states *= layer.layer_scalar                          # learned scalar per layer
```

### Layer Scalar Values (E2B, bfloat16, 2026-04-22)

```
Layer  0:  0.0178  ← nearly zero (first layer barely contributes)
Layer  1:  0.2227
Layer  2:  0.7930
Layer  3:  0.2871
Layer  4:  0.4980  ← first global attention
Layer  5:  0.6367
Layer  6:  0.4980
Layer  7:  0.6094
Layer  8:  0.3770
Layer  9:  0.4648  ← global
Layer 10:  0.4434
Layer 11:  0.3691
Layer 12:  0.3242
Layer 13:  0.0884  ← last sliding source layer
Layer 14:  0.0286  ← last global source layer (tiny!)
Layer 15:  0.2539  ← first shared KV layer
Layer 16–29: 0.49–0.87 range
Layer 34:  0.1670  ← final layer, reduced
```

Mechanistic note: layers 0, 13, 14 have dramatically small scalars. These are the first layer and the two "last source" layers before KV sharing begins. The model learned to minimize their residual contribution, possibly because those positions are architecturally constrained (layer 0 = pure embedding processing; layers 13/14 = KV cache anchor layers that can't specialize freely).

## Per-Layer Type Table (35 layers)

| Layers | Attention Type | is_kv_shared? | d_head | KV source |
|--------|---------------|---------------|--------|-----------|
| 0–3 | sliding | no | 256 | — |
| 4 | **full** | no | 512 | — |
| 5–8 | sliding | no | 256 | — |
| 9 | **full** | no | 512 | — |
| 10–13 | sliding | no | 256 | — |
| 14 | **full** | no | 512 | — |
| 15–18 | sliding | **yes** | 256 | layer 13 |
| 19 | **full** | **yes** | 512 | layer 14 |
| 20–23 | sliding | **yes** | 256 | layer 13 |
| 24 | **full** | **yes** | 512 | layer 14 |
| 25–28 | sliding | **yes** | 256 | layer 13 |
| 29 | **full** | **yes** | 512 | layer 14 |
| 30–33 | sliding | **yes** | 256 | layer 13 |
| 34 | **full** | **yes** | 512 | layer 14 |

## Open Questions (remaining after config enumeration)

- What is `num_global_key_value_heads` for full attention layers? (null in config — need model load)
- Does the weight conversion need separate logic for local (d_head=256) vs global (d_head=512) layers?
- How does partial RoPE (`partial_rotary_factor=0.25`) interact with TL's RoPE implementation?
- How does `final_logit_softcapping` (30.0) integrate with TL's unembed/final logits?
- Exact module paths in `Gemma4ForConditionalGeneration` — TBD from full `named_modules()` output
- Is `pre_feedforward_layernorm` the attribute name (same as Gemma 3)?
- Does `use_double_wide_mlp` change the module structure or just the size?
