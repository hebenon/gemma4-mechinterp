"""
Draft: PLE additions to TransformerLens TransformerBlock for Gemma 4.

This shows the modifications needed to transformer_lens/components/transformer_block.py.
The actual file is in the TL fork — this draft captures the logic before the fork exists.

PLE sits after the complete standard transformer output (attn + MLP), before hook_resid_post.

Flow (sequential mode, use_normalization_before_and_after=True, use_ple=True):
    hook_resid_pre
      → ln1 → attn → hook_attn_out
      → + resid_pre → hook_resid_mid
      → ln2 → mlp → hook_mlp_out
      → + resid_mid → resid_standard         ← standard output
      → hook_ple_input(ple_vec)              ← PLE conditioning vector [B, L, d_ple]
      → gate = hook_ple_gate(gelu(W_gate(resid_standard)))  [B, L, d_ple]
      → ple_out = hook_ple_output(W_up(gate * ple_vec))     [B, L, d_model]
      → resid_post = resid_standard + ple_ln(ple_out)
      → resid_post = resid_post * layer_scale  (if cfg.layer_scalar)
      → hook_resid_post(resid_post)

References:
  tl_adapter_notes.md — hook design and semantics
  architecture.md — PLE mechanics from HF source
  weight_conversion_draft.py — weight name conventions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens.hook_points import HookPoint
# TL's RMSNorm — exact import path TBD from fork inspection
# from transformer_lens.components import RMSNorm  (or LayerNorm depending on cfg)


# ─── New attributes to add to TransformerBlock.__init__ ──────────────────────

def _init_ple_components(self):
    """Call from TransformerBlock.__init__ after standard component init."""
    if not self.cfg.use_ple:
        return

    d_model = self.cfg.d_model
    d_ple = self.cfg.d_ple

    # HookPoints — auto-discovered by setup() via nn.Module scan
    self.hook_ple_input = HookPoint()   # [B, L, d_ple] — PLE conditioning vector
    self.hook_ple_gate = HookPoint()    # [B, L, d_ple] — gate activations post-GELU
    self.hook_ple_output = HookPoint()  # [B, L, d_model] — bottleneck output pre-residual

    # Weight matrices
    self.ple_gate = nn.Linear(d_model, d_ple, bias=False)
    self.ple_up = nn.Linear(d_ple, d_model, bias=False)

    # Post-PLE norm (applied to ple_out before residual add, normalizes d_model)
    # TL uses LayerNorm or RMSNorm depending on cfg.normalization_type
    # For Gemma 4: RMSNorm with no bias, +1 pre-add in weight loading
    # Use same norm class as the rest of the block
    self.ple_ln = _build_norm(self.cfg, d_model)

    # Per-layer learned scalar (new in Gemma 4)
    if self.cfg.layer_scalar:
        # Initialized to 1.0, stored as parameter
        # Weight loading sets this from layer.layer_scalar in HF model
        self.layer_scale = nn.Parameter(torch.ones(1))


def _build_norm(cfg, d):
    """Build the appropriate norm for this config. Mirrors what TL does for ln1/ln2."""
    # TL's existing norm construction — exact API TBD from fork inspection
    # Gemma 4 uses RMSNorm everywhere; use_normalization_before_and_after=True
    from transformer_lens.components import RMSNorm
    return RMSNorm(d, cfg=cfg)


# ─── PLE forward pass (inject into TransformerBlock.forward) ─────────────────

def _apply_ple(self, resid_standard: torch.Tensor, ple_vec: torch.Tensor) -> torch.Tensor:
    """
    Apply PLE gated bottleneck to residual stream.

    Args:
        resid_standard: [B, L, d_model] — residual stream after attn + MLP
        ple_vec: [B, L, d_ple] — precomputed PLE vector for this layer

    Returns:
        resid_post: [B, L, d_model] — after PLE addition and layer_scale
    """
    # PLE conditioning vector — intervene here to ablate PLE (zero → removes conditioning)
    ple_vec = self.hook_ple_input(ple_vec)

    # Gate: residual stream projects down to d_ple, GELU activated
    gate = self.hook_ple_gate(F.gelu(self.ple_gate(resid_standard)))

    # Gated bottleneck output: element-wise multiply, project back up
    ple_out = self.hook_ple_output(self.ple_up(gate * ple_vec))

    # RMSNorm applied before residual add (Gemma 4 post_per_layer_input_norm)
    resid_post = resid_standard + self.ple_ln(ple_out)

    # Per-layer learned scale (new in Gemma 4, initialized to 1)
    if self.cfg.layer_scalar:
        resid_post = resid_post * self.layer_scale

    return resid_post


# ─── Integration point in TransformerBlock.forward ───────────────────────────

def _forward_integration_sketch(self, resid_pre, ple_vec=None, shared_kv=None, ...):
    """
    Sketch of how PLE integrates into the existing TransformerBlock.forward.

    In the real implementation, this merges with the existing forward method
    rather than replacing it. Key change: accept ple_vec kwarg, call _apply_ple
    after MLP output, before hook_resid_post.

    The existing hook_resid_post continues to fire AFTER PLE — consistent with
    all TL models where hook_resid_post = final layer output.
    """
    # --- Standard transformer (existing code, unchanged) ---
    resid_mid = ...  # attn output + hook_resid_mid
    mlp_out = ...    # MLP output + hook_mlp_out
    resid_standard = resid_mid + mlp_out

    # --- PLE addition (new) ---
    if self.cfg.use_ple and ple_vec is not None:
        resid_post = _apply_ple(self, resid_standard, ple_vec)
    else:
        resid_post = resid_standard

    # --- hook_resid_post fires last (unchanged semantics) ---
    return self.hook_resid_post(resid_post)


# ─── Model-level PLE precomputation (inject into HookedTransformer.forward) ──

class PLEPrecomputer(nn.Module):
    """
    Computes PLE vectors for all layers once before the transformer block loop.

    Registered as self.ple in HookedTransformer when cfg.use_ple=True.

    Output [B, L, n_layers, d_ple] is sliced per-layer and passed to each block.
    Model-level hooks expose the two components for ablation experiments.
    """

    def __init__(self, cfg):
        super().__init__()
        n_layers = cfg.n_layers
        d_ple = cfg.d_ple
        d_model = cfg.d_model
        ple_vocab = cfg.ple_vocab_size

        # Token identity: separate embedding table, vocab_ple × (n_layers * d_ple)
        self.W_embed = nn.Embedding(ple_vocab, n_layers * d_ple)

        # Context projection: d_model → (n_layers * d_ple), no bias
        self.W_proj = nn.Linear(d_model, n_layers * d_ple, bias=False)

        # RMSNorm on context projection (normalizes d_ple dimension)
        # Applied per-layer after reshape
        self.ln = _build_norm(cfg, d_ple)

        # HookPoints for component decomposition
        self.hook_token_embeds = HookPoint()   # [B, L, n_layers, d_ple] — token identity
        self.hook_context_proj = HookPoint()   # [B, L, n_layers, d_ple] — context component

    def forward(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] — token indices
            inputs_embeds: [B, L, d_model] — initial token embeddings (before transformer layers)

        Returns:
            ple_vecs: [B, L, n_layers, d_ple]
        """
        B, L = input_ids.shape
        n_layers = self.W_embed.weight.shape[1] // (self.ln.weight.shape[0])
        d_ple = self.ln.weight.shape[0]

        # Token identity component
        token_embeds = self.W_embed(input_ids)                      # [B, L, n_layers * d_ple]
        token_embeds = token_embeds.reshape(B, L, n_layers, d_ple)
        token_embeds = self.hook_token_embeds(token_embeds)

        # Context component (from initial embeddings, NO positional info — pure vocab signal)
        # Scale by hidden_size**-0.5 as in HF source
        context = self.W_proj(inputs_embeds) * (inputs_embeds.shape[-1] ** -0.5)
        context = context.reshape(B, L, n_layers, d_ple)
        # RMSNorm applied to d_ple dimension of each (layer, position) independently
        context = self.ln(context)
        context = self.hook_context_proj(context)

        # Combine: normalized sum
        ple_vecs = (token_embeds + context) * (2 ** -0.5)          # [B, L, n_layers, d_ple]

        return ple_vecs


# ─── Config additions (HookedTransformerConfig) ──────────────────────────────

# Add these fields to HookedTransformerConfig dataclass:
#
#   use_ple: bool = False
#       Enable Per-Layer Embedding gated bottleneck (Gemma 4 E2B, E4B)
#
#   d_ple: int = 0
#       PLE bottleneck dimension (hidden_size_per_layer_input in HF config)
#       E2B/E4B: 256; 31B/MoE: 0 (PLE not used)
#
#   ple_vocab_size: int = 0
#       Vocabulary size for embed_tokens_per_layer
#       E2B/E4B: 262144
#
#   layer_scalar: bool = False
#       Enable learned per-layer output scale (new in Gemma 4)
#       All Gemma 4 variants use this; set True for all Gemma 4 entries
#
#   num_kv_shared_layers: int = 0
#       Number of terminal layers that share KV from a source layer
#       E2B: ~17 (confirm from enumeration); 31B/MoE: 0
#
#   first_kv_shared_layer_idx: int = -1
#       Computed as n_layers - num_kv_shared_layers; -1 means no sharing
#       Set during config __post_init__ from num_kv_shared_layers


# ─── Notes for PR / TL maintainers ───────────────────────────────────────────
#
# PLE is the primary novel contribution of this PR. It is NOT model-specific machinery
# hiding behind a flag — it is a new hook mechanism category:
#   - hook_ple_input, hook_ple_gate, hook_ple_output at block level
#   - hook_token_embeds, hook_context_proj at model level (PLEPrecomputer)
#
# The block-level hooks are analogous to hook_attn_out and hook_mlp_out.
# The model-level hooks are a new pattern (no prior TL model has precomputed
# conditioning signals exposed at this level).
#
# Ablation semantics:
#   Zero hook_ple_input → removes PLE conditioning (gate * 0 = 0, bottleneck outputs 0)
#   Zero hook_ple_gate  → same effect from hidden-states side
#   Zero hook_ple_output → removes PLE contribution after computation
#   Zero hook_token_embeds → context-only PLE (tests semantic component alone)
#   Zero hook_context_proj → token-identity-only PLE (tests idiosyncratic component)
#
# NOTE: "context" is a misnomer from HF source. Both components are vocabulary-dependent,
# not sequence-context-dependent (inputs_embeds has no positional info; RoPE is applied
# inside attention). Ablating hook_context_proj tests the SEMANTIC component (smooth
# variation across similar tokens); ablating hook_token_embeds tests the IDIOSYNCRATIC
# component (arbitrary per-token variation).
