"""
Draft: Gemma 4 weight conversion extension for TransformerLens

This extends transformer_lens/pretrained/weight_conversions/gemma.py
to support Gemma4ForCausalLM.

Integration plan:
1. Add Gemma 4 detection at top of convert_gemma_weights()
2. Add PLE weight mappings (model-level + per-block)
3. Add layer_scalar per block
4. Handle post_feedforward_layernorm (same as Gemma 3 — use_normalization_before_and_after=True)

TL naming conventions used:
  Model-level PLE:
    ple.W_embed       — embed_tokens_per_layer [vocab_ple, n_layers * d_ple]
    ple.W_proj        — per_layer_model_projection [n_layers * d_ple, d_model]
    ple.ln.w          — per_layer_projection_norm [d_ple]
  Per-block PLE (when cfg.use_ple):
    blocks.{l}.ple_gate.W     — per_layer_input_gate [d_ple, d_model]
    blocks.{l}.ple_up.W       — per_layer_projection [d_model, d_ple]
    blocks.{l}.ple_ln.w       — post_per_layer_input_norm [d_model]
    blocks.{l}.layer_scale    — layer_scalar (scalar)

References:
  HF source: src/transformers/models/gemma4/modeling_gemma4.py
  architecture notes: notes/architecture.md
  adapter notes: notes/tl_adapter_notes.md
"""

import einops
import torch
from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig


# Detection helpers
GEMMA4_TEXT_ONLY_CLASSES = {"Gemma4ForCausalLM"}
GEMMA4_MULTIMODAL_CLASSES = {"Gemma4ForConditionalGeneration", "Gemma4Model"}
GEMMA4_CLASSES = GEMMA4_TEXT_ONLY_CLASSES | GEMMA4_MULTIMODAL_CLASSES


def is_gemma4(model) -> bool:
    return type(model).__name__ in GEMMA4_CLASSES


def get_gemma4_base_model(gemma):
    """Get the Gemma4TextModel from text-only or multimodal wrapper."""
    cls_name = type(gemma).__name__
    if cls_name in GEMMA4_TEXT_ONLY_CLASSES:
        # Gemma4ForCausalLM: gemma.model is Gemma4TextModel
        return gemma.model
    else:
        # Multimodal: structure TBD from enumeration — likely gemma.language_model.model
        if hasattr(gemma, "language_model"):
            return gemma.language_model.model if hasattr(gemma.language_model, "model") else gemma.language_model
        raise ValueError(f"Unknown Gemma 4 multimodal structure for {cls_name}")


def convert_gemma4_weights(gemma, cfg: HookedTransformerConfig) -> dict:
    """
    Convert Gemma 4 weights to TransformerLens format.

    Gemma 4 extends Gemma 3 with:
    - Per-Layer Embeddings (PLE): gated bottleneck at each decoder layer
    - layer_scalar: learned per-layer output scale
    - Shared KV cache: activation-level sharing, no weight changes needed

    The base transformer weights (attn, mlp, ln) follow Gemma 3 conventions
    with use_normalization_before_and_after=True.
    """
    assert cfg.n_key_value_heads is not None
    assert cfg.d_mlp is not None

    base_model = get_gemma4_base_model(gemma)
    state_dict = {}

    # === Embeddings ===
    # Same scaling as Gemma 3: embed × sqrt(d_model)
    state_dict["embed.W_E"] = base_model.embed_tokens.weight * torch.tensor(
        cfg.d_model ** 0.5, dtype=cfg.dtype
    )

    # === PLE model-level weights ===
    if cfg.use_ple:
        # embed_tokens_per_layer: [vocab_ple, n_layers * d_ple]
        state_dict["ple.W_embed"] = base_model.embed_tokens_per_layer.weight

        # per_layer_model_projection: Linear(d_model, n_layers * d_ple, bias=False)
        # Weight shape: [n_layers * d_ple, d_model] → transpose to [d_model, n_layers * d_ple]
        state_dict["ple.W_proj"] = base_model.per_layer_model_projection.weight.T

        # per_layer_projection_norm: RMSNorm(d_ple)
        # Same +1 trick as other Gemma RMSNorm weights
        state_dict["ple.ln.w"] = (
            base_model.per_layer_projection_norm.weight.float()
            + torch.ones_like(base_model.per_layer_projection_norm.weight, dtype=torch.float32)
        )

    # === Per-layer weights ===
    for l in range(cfg.n_layers):
        layer = base_model.layers[l]

        # Layer norms (same as Gemma 3 with use_normalization_before_and_after)
        # GemmaRMSNorm adds 1 in forward() — pre-add here
        state_dict[f"blocks.{l}.ln1.w"] = (
            layer.input_layernorm.weight.float()
            + torch.ones_like(layer.input_layernorm.weight, dtype=torch.float32)
        )
        # Gemma 3+ has post_attention_layernorm (applied before FFN)
        # TBD: confirm Gemma 4 uses same names as Gemma 3
        # For now assume: pre_feedforward_layernorm → ln2, post_feedforward_layernorm → ln2_post
        if hasattr(layer, "pre_feedforward_layernorm"):
            state_dict[f"blocks.{l}.ln2.w"] = (
                layer.pre_feedforward_layernorm.weight.float()
                + torch.ones_like(layer.pre_feedforward_layernorm.weight, dtype=torch.float32)
            )
        if hasattr(layer, "post_feedforward_layernorm"):
            state_dict[f"blocks.{l}.ln2_post.w"] = (
                layer.post_feedforward_layernorm.weight.float()
                + torch.ones_like(layer.post_feedforward_layernorm.weight, dtype=torch.float32)
            )
        if hasattr(layer, "post_attention_layernorm"):
            state_dict[f"blocks.{l}.ln1_post.w"] = (
                layer.post_attention_layernorm.weight.float()
                + torch.ones_like(layer.post_attention_layernorm.weight, dtype=torch.float32)
            )

        # Attention weights (same as Gemma 3)
        W_Q = einops.rearrange(layer.self_attn.q_proj.weight, "(n h) m -> n m h", n=cfg.n_heads)
        W_K = einops.rearrange(layer.self_attn.k_proj.weight, "(n h) m -> n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(layer.self_attn.v_proj.weight, "(n h) m -> n m h", n=cfg.n_key_value_heads)
        W_O = einops.rearrange(layer.self_attn.o_proj.weight, "m (n h) -> n h m", n=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        # Q/K norm (same as Gemma 3)
        if cfg.use_qk_norm:
            state_dict[f"blocks.{l}.attn.q_norm.w"] = (
                layer.self_attn.q_norm.weight.float()
                + torch.ones_like(layer.self_attn.q_norm.weight, dtype=torch.float32)
            )
            state_dict[f"blocks.{l}.attn.k_norm.w"] = (
                layer.self_attn.k_norm.weight.float()
                + torch.ones_like(layer.self_attn.k_norm.weight, dtype=torch.float32)
            )

        # MLP weights (same as Gemma 3)
        state_dict[f"blocks.{l}.mlp.W_in"] = layer.mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = layer.mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_out"] = layer.mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        # PLE per-block weights
        if cfg.use_ple:
            # per_layer_input_gate: Linear(d_model, d_ple, bias=False)
            # Weight [d_ple, d_model] — store transposed for matmul convention
            state_dict[f"blocks.{l}.ple_gate.W"] = layer.per_layer_input_gate.weight.T

            # per_layer_projection: Linear(d_ple, d_model, bias=False)
            # Weight [d_model, d_ple] — store transposed
            state_dict[f"blocks.{l}.ple_up.W"] = layer.per_layer_projection.weight.T

            # post_per_layer_input_norm: RMSNorm
            # TBD: is this norming d_ple or d_model? Likely d_model (applied after projection)
            state_dict[f"blocks.{l}.ple_ln.w"] = (
                layer.post_per_layer_input_norm.weight.float()
                + torch.ones_like(layer.post_per_layer_input_norm.weight, dtype=torch.float32)
            )

            # layer_scalar: learned scalar, applied as hidden_states *= layer_scalar
            # Store as-is (scalar tensor)
            state_dict[f"blocks.{l}.layer_scale"] = layer.layer_scalar

    # Final norm and unembedding
    state_dict["ln_final.w"] = (
        base_model.norm.weight.float()
        + torch.ones_like(base_model.norm.weight, dtype=torch.float32)
    )

    if hasattr(gemma, "lm_head"):
        state_dict["unembed.W_U"] = gemma.lm_head.weight.T
        state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)
    else:
        state_dict["unembed.W_U"] = base_model.embed_tokens.weight.T
        state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict


# === Config entries to add to loading_from_pretrained.py ===
# (This is a reference for what entries to add, not actual code)
GEMMA4_E2B_CONFIG = {
    "model_name": "google/gemma-4-E2B-it",
    "architecture": "Gemma4ForCausalLM",
    "d_model": 2304,
    "d_head": 288,        # d_model / n_heads = 2304 / 8
    "n_heads": 8,
    "n_key_value_heads": 4,
    "d_mlp": None,        # TBD from enumeration
    "n_layers": 30,
    "n_ctx": 131072,      # 128K
    "d_vocab": None,      # TBD from enumeration (standard Gemma vocab or different)
    "use_qk_norm": True,
    "use_normalization_before_and_after": True,
    "use_ple": True,
    "d_ple": 256,
    "ple_vocab_size": 262144,
    "num_kv_shared_layers": None,   # TBD from enumeration
    "layer_types": None,            # TBD from enumeration (5:1 sliding/full)
    "sliding_window": 512,
    "rotary_base_local": 10000,     # TBD: confirm per-layer RoPE bases
    "rotary_base_global": 1000000,  # TBD: confirm
    "tokenizer_name": "google/gemma-4-E2B-it",
    "dtype": torch.bfloat16,
    "weight_conversion": "gemma4",
}
