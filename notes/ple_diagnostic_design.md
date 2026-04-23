# PLE Diagnostic Design

*Design for the mechinterp investigation in Phase 5. Written 2026-04-21.*

## What PLE actually is (reprise)

PLE is a gated bottleneck at every transformer block:

```
gate = act_fn(W_gate @ h)        # [B, L, 256] — residual stream projected down
gated = gate * ple_vec[layer]    # [B, L, 256] — token signal filtered by residual
out = W_up @ gated               # [B, L, 2304] — projected back up
h_new = h + RMSNorm(out)         # residual add
h_new = h_new * layer_scalar     # per-layer learned scale
```

`ple_vec[layer]` has two components, but the HF naming is misleading:

- **Token identity** (`embed_tokens_per_layer`): direct lookup per (token_id, layer). Maximum expressivity — each token can have a completely different 256-dim vector per layer, independent of embedding similarity.
- **Semantic component** (labeled "context" in HF code, but misnamed): `per_layer_model_projection(inputs_embeds)`. `inputs_embeds` is the initial token embedding BEFORE any transformer layer, and Gemma uses RoPE (applied inside attention), so `inputs_embeds` contains NO positional information. This is a linear function of the vocabulary embedding — same for every occurrence of the same token regardless of position or sequence context.

**Both components are vocabulary-dependent, not sequence-context-dependent.** The "context" label in HF's code is misleading. The distinction is:
- Token identity: idiosyncratic, per-token (can be arbitrary per layer)
- Semantic component: structured, smooth across tokens (similar vocab embeddings → similar PLE contribution)

This has interpretability implications: when we ablate the semantic component (`hook_ple_context_proj`), we're removing the smoothly-varying semantic signal while keeping idiosyncratic per-token variation. For rare tokens, the semantic component may dominate (few training examples for the token identity lookup). For common tokens, both are well-trained and potentially redundant.

The gate is the interesting part: it's computed from the *current* hidden state `h`, so the residual stream actively modulates how much PLE signal enters. This is not a simple additive injection — it's an information filter.

## Diagnostic Questions

**Q1 (primary)**: How much does PLE contribute per layer, and how does this vary?
- Null hypothesis: uniform contribution across layers
- Hypothesis: larger in early layers (token identity matters before context accumulates), smaller in late layers

**Q2**: Is PLE contribution token-specific?
- Different tokens at the same position: does PLE vary more by token or by layer?
- Idiom: does common-word PLE differ from rare-word PLE?

**Q3**: How does ablating PLE affect downstream residual stream?
- Cumulative: zero PLE at all layers, measure `||h_ablated - h_original||` at each layer
- Surgical: zero PLE at layer L only, measure propagation through subsequent layers

**Q4**: What does the gate select?
- Which dimensions of `h` drive large gate activations?
- Does the gate attend to the same directions as attention heads, or orthogonal ones?

**Q5**: Are token-identity and context components of ple_vec distinguishable?
- Swap the context component with a different token's context component — does behavior change?
- Zero the context component entirely — how does this differ from zeroing the full ple_vec?

---

## Experiment 1: Per-layer PLE contribution magnitude

**What**: At each layer L, compute the ratio of PLE output norm to total residual stream norm.

```python
def ple_contribution_ratio(model, tokens, device):
    """
    Returns: array of shape [n_layers] with ||PLE_out|| / ||h_post|| per layer.
    """
    ple_out_norms = []
    h_post_norms = []
    
    def save_ple_out(value, hook):
        ple_out_norms.append(value.norm(dim=-1).mean().item())
        return value
    
    def save_resid_post(value, hook):
        h_post_norms.append(value.norm(dim=-1).mean().item())
        return value
    
    hooks = (
        [(f"blocks.{l}.hook_ple_output", save_ple_out) for l in range(model.cfg.n_layers)]
        + [(f"blocks.{l}.hook_resid_post", save_resid_post) for l in range(model.cfg.n_layers)]
    )
    model.run_with_hooks(tokens, fwd_hooks=hooks)
    
    return [p / h for p, h in zip(ple_out_norms, h_post_norms)]
```

**Variants to run**:
- Common English text (prose paragraph)
- Poetry / stylistically distinct text
- Code
- Repeated tokens: `"the the the the the"`
- Rare-token text: technical jargon, names

**Expected shape**: Plot `layer_idx` vs `ratio` — hypothesis is monotone decreasing or U-shaped (large at start, small in middle, perhaps larger again at end where PLE recalibrates for unembedding).

---

## Experiment 2: PLE ablation — full and surgical

**Full ablation**: zero `hook_ple_input` at all layers.
- Measure: does the model still produce coherent text?
- Measure: logit MAE vs full model
- Measure: `||h_ablated[l] - h_original[l]||` at each layer (cumulative drift)

```python
def ablate_all_ple(value, hook):
    return torch.zeros_like(value)

ablation_hooks = [(f"blocks.{l}.hook_ple_input", ablate_all_ple) for l in range(n_layers)]
logits_ablated = model.run_with_hooks(tokens, fwd_hooks=ablation_hooks)
```

**Layer-surgical ablation**: zero PLE at exactly one layer L, run clean otherwise.
- For each L: compute logit MAE vs baseline
- This gives a per-layer "PLE indispensability" curve
- Fast since each run is identical except one hook

**Expected result**: Ablating early-layer PLE causes larger logit change than late-layer. If wrong (late layers matter more), that tells us PLE is doing something in late computation — possibly related to token calibration before unembedding.

---

## Experiment 3: Gate analysis — what the gate selects

The gate `act_fn(W_gate @ h)` is a 256-dim filter computed from the 2304-dim hidden state. Each of the 256 dimensions is a linear probe on the residual stream, activated by GeLU (or SiLU — TBD from enumeration).

**Q**: Which directions in `h` drive large gate activations?

```python
# At layer L, collect gate pre-activation across a corpus
gate_preacts = []  # [n_tokens, 256]

def save_gate(value, hook, preact_store):
    # hook_ple_output gives post-W_up; need gate separately
    # May need a custom hook inside the PLE block — or instrument W_gate output
    preact_store.append(value)
    return value
```

Note: this requires hooking the gate intermediate. The hook design has `hook_ple_input` (the ple_vec) and `hook_ple_output` (after W_up). To access the gate directly we may need a third hook: `hook_ple_gate` at [B, L, 256] before element-wise multiply. Consider adding this in Phase 3.

**Analysis**: SVD of gate activation matrix across tokens → top directions that gate responds to. Correlate with attention head output directions (are they the same?).

---

## Experiment 4: Token identity vs context decomposition

`ple_vec = (token_identity + context_proj) * (1/√2)`

We can decompose by intervention:

```python
# Intervention A: zero the context component (static token identity only)
def ple_token_identity_only(value, hook, token_component):
    # value is the combined ple_vec; replace with just token component
    return token_component * (2**-0.5)  # renormalized

# Intervention B: zero token identity (context only)
def ple_context_only(value, hook, context_component):
    return context_component * (2**-0.5)
```

For this we need to intercept and modify `hook_ple_input` with precomputed components. The PLE precomputation runs before the block loop in `HookedTransformer.forward()`, so we'd compute both components separately and expose them.

Requires: `hook_ple_token` and `hook_ple_context` hooks at the precomputation stage (model-level, before blocks). Consider adding these to the model-level forward pass.

---

## Experiment 5: Cross-token PLE transfer

**What**: Take token at position P in sentence S1. Swap its ple_vec with the same token at position P in sentence S2 (different context). Measure change in prediction.

This tests: how much does the context component of ple_vec actually matter vs. the static token identity?

```python
def transfer_ple(value, hook, source_ple_vecs, target_positions):
    # value: [B=2, L, 256] — batch contains S1 and S2
    # swap S1's PLE at target_positions with S2's
    modified = value.clone()
    modified[0, target_positions] = source_ple_vecs[1, target_positions]
    return modified
```

---

## Reporting structure (Mensmachina post outline)

1. **What PLE is**: the gated bottleneck design — not what it sounds like
2. **How much it contributes**: Experiment 1 results (per-layer ratio)
3. **What happens without it**: Experiment 2 (ablation — full and surgical)
4. **What the gate selects**: Experiment 3 (direction analysis)
5. **Which component matters**: Experiments 4 and 5 (token identity vs context)
6. **Implications**: PLE as a per-layer "token recall" gate — what this means for how Gemma 4 processes long-context inputs

## Implementation notes

- Experiments 1, 2: runnable with just `hook_ple_input` and `hook_ple_output` — Phase 3 hooks
- Experiment 3: needs `hook_ple_gate` — add in Phase 3 alongside others
- Experiments 4, 5: needs model-level `hook_ple_token` and `hook_ple_context` — add in Phase 3
- All experiments: run on T4 (Kaggle), short corpus (1-2 paragraphs), ~30s per run

## Compute budget estimate

- Experiment 1: ~5 runs × 30s = 2.5 min
- Experiment 2 (full ablation): 1 run = 30s
- Experiment 2 (surgical, 30 layers): 30 runs × 30s = 15 min
- Experiment 3: corpus sweep, ~50 examples × 30s = 25 min
- Experiments 4, 5: ~10 runs × 30s = 5 min
- **Total: ~50 min on T4**

Well within Kaggle's 30h/week free tier.
