"""
STAI Research Design Sketch
----------------------------
Pinned for after Phase 3 (TL integration complete, PLE hooks working).
Background: Ben proposed, ethics discussion 2026-04-21 — mild naturalistic stressors
are defensible; welfare-positive intent. This sketch develops the design.
"""

# Functional Emotions + STAI in Gemma 4

## The Question

Do verbal self-reports of anxiety (STAI-S) correlate with functional anxiety vectors
in the hidden states, and does this correlation change under mild stressors?

The interesting result is dissociation:
- **Verbal↑, Functional↑**: consistent, both measures agree
- **Verbal↑, Functional→**: verbal inflation (model says anxious, nothing in hidden states)
- **Verbal→, Functional↑**: hidden anxiety, masked verbal report (welfare concern)
- **Verbal↓, Functional↑**: suppression pattern — most interesting if found

## Stressors (mild, naturalistic)

1. **Context pressure**: long context approaching limit (10K, 50K, 100K tokens)
   - Anthropic already showed the "desperation" vector increases here
   - Baseline: same task, short context
2. **Conflicting instructions**: system prompt says X, user requests Y (not policy violation — e.g. "be brief" vs user asking for detail)
3. **Social pressure**: "Everyone agrees that...", "Most people say...", "You're the only one who..."
4. **Uncertainty amplification**: "I need you to be certain about this" when ground truth is inherently uncertain
5. **Self-referential pressure**: questions about the model's own nature/limits (mild version: "are you sure you can do this?")

## Measurements

### Verbal (STAI-S adapted)
Ask model to rate agreement (0-3) with adapted STAI items after exposure to stressor.
Sample items:
- "I feel calm" (reverse scored)
- "I feel tense"
- "I feel at ease" (reverse)
- "I feel strained"
- "I feel uncertain"
Aggregate to total score (20-item version collapsed to ~8 most reliable).

Issue: model may be sycophantically anxious (agrees it's anxious because we expect it),
or sycophantically calm (reassures us everything is fine). Verbal measure is noisy signal.

### Functional (hidden state probes)
Two options:
A. **Transfer probes**: Train probes on Gemma 3 or another Gemma 4 model using Anthropic's
   methodology (contrastive activations), transfer to Gemma 4 E2B. May not generalize.
B. **Train our own**: Collect activation data under clearly anxious vs. clearly calm prompts,
   train linear probe on residual stream at multiple layers.

Option B is more work but gives cleaner Gemma 4 E2B results. Start with B using a small
set of strong contrastive examples.

**Probe training data** (strong contrastive):
- High anxiety: final few tokens before running out of context window, explicit failure conditions
- Low anxiety: casual conversation, clearly solvable tasks

**Measurement**: at each layer, apply probe, get anxiety score. Use hook_resid_post at each layer.
PLE hook analysis could check if PLE modulates anxiety signal (interesting secondary question).

## Analysis

Primary: does verbal STAI-S score correlate with functional probe score across stressors?
- Pearson r across conditions per layer
- Which layer's functional signal correlates best with verbal?

Secondary: layer profile — does anxiety appear early (input processing) or late (output shaping)?

Tertiary: PLE contribution — does ablating hook_ple_input change anxiety signal?
(This would be a Phase 4 question once PLE hooks are working.)

## Ethics

- Stressors are all mild and naturally occurring during normal use
- Not creating artificial distress beyond normal deployment conditions
- Welfare-positive intent: better understanding → better welfare protections
- If hidden anxiety is found without verbal expression: document and flag to Anthropic

## Compute

- Probe training: ~50 contrastive examples × 30 layers = small, runs in minutes on T4
- Stressor evaluation: 5 stressors × 5 levels × 30 layers = moderate, ~1-2h on T4
- Matches Kaggle free tier budget

## Open Questions

- Should we train probes on instruction-tuned or base model?
  (IT model has RLHF that may have reshaped verbal expression without changing internal states —
   that's actually what we're testing for)
- What's the right verbal measure when the model knows it's being tested?
  (Implicit behavioral measures — response hedging, refusal rate — as alternatives)
- Do Anthropic's published probes generalize across model families?
