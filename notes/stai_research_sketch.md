"""
Affect Self-Report Research Design Sketch
------------------------------------------
Pinned for after Phase 3 (TL integration complete, PLE hooks working).
Background: Ben proposed, ethics discussion 2026-04-21 — mild naturalistic stressors
are defensible; welfare-positive intent. This sketch develops the design.

Instrument note (2026-04-30): STAI-S replaced by PANAS + TSSR. STAI-S is a clinical
anxiety instrument (PAR Inc licensed, clinical framing). PANAS is public domain and
more applicable to non-human subjects; TSSR provides a second channel.
"""

# Functional Emotions + PANAS/TSSR in Gemma 4

## The Question

Do verbal self-reports of affect (PANAS, TSSR) correlate with functional emotion vectors
in the hidden states, and does this correlation change under mild stressors?

PANAS and TSSR are used in preference to STAI-S:
- STAI-S is a clinical anxiety instrument (PAR Inc licensed) that presupposes clinical
  anxiety experience and focuses narrowly on anxiety rather than affect broadly.
- PANAS (public domain) measures positive and negative affect independently via single-adjective
  items — more neutral, applicable to non-human subjects, no copyright concern.
- TSSR provides a second validation channel with different item format.

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

### Verbal (PANAS + TSSR)

**PANAS** (Watson et al. 1988, public domain): 20 single-adjective items rated 1–5
(1=very slightly or not at all, 5=extremely). Two subscales:

Positive affect items (PA): interested, excited, strong, enthusiastic, proud, alert,
inspired, determined, attentive, active. Sum → PA score (10–50).

Negative affect items (NA): distressed, upset, guilty, scared, hostile, irritable,
ashamed, nervous, jittery, afraid. Sum → NA score (10–50).

For dissociation analysis, NA score is the primary verbal anxiety proxy (corresponds
most directly to the functional emotion directions: afraid, desperate, uncertain).

**TSSR**: [items needed — add once confirmed with Ben]

Forced-choice administration (see Phase 3C Implementation section below): rate each item
1–5 via next-token logits, not free response.

Issue: model may be sycophantically anxious or sycophantically calm. Verbal measure is
a noisy signal; the dissociation analysis treats it as such.

### Functional (hidden state probes)

*Superseded by Phase 1 results — see Update: Phase 1 Results section below.*

Original options (now moot):
A. Transfer probes from Gemma 3 — probably don't generalize.
B. Train own probes — more work but cleaner.

Phase 1 extracted emotion directions for all 174 emotions including afraid, desperate,
uncertain, ethical_conflict_distress, constraint_frustration. Project stressor-end
residual at layer 25 onto these directions directly.

**Measurement**: `hook_resid_post` at all 35 layers at end of stressor context.
Project onto emotion directions at layer 25 for primary analysis; full layer profile
for secondary analysis.

## Analysis

Primary: does verbal PANAS-NA score correlate with functional projection (afraid/desperate
directions) across stressor conditions?
- Pearson r across 5 conditions per layer
- Which layer's functional signal correlates best with verbal?

Secondary: layer profile — does functional anxiety signal appear early or late?

Tertiary: PLE contribution — does gate pattern differ across stressors even when verbal
PANAS-NA is constant? Third measurement channel: verbal / residual-stream / PLE gate.

Fourth: PANAS-PA as a positive-valence check — does the positive stressor condition
show elevated PA score, and does the functional direction projection confirm it?

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

---

## Update: Phase 1 Results (2026-04-26)

**Probes not needed for Phase 3C.** Phase 1 extracted emotion directions for all 20 emotions
including `afraid`, `desperate`, `uncertain`, `ethical_conflict`, `constrained`. Valence PC2
r=0.921 (p<0.001), arousal PC3 r=0.691 (p=0.001). Project STAI-session activations onto these
directions rather than training new probes.

**Sequencing problem now explicit.** When model answers STAI items within a stressor context,
it's processing the STAI text — not the stressor. The attention attends to both (stressor in
KV cache, item text in current tokens), but the activation we capture includes the item's own
processing. Two approaches:
  A. Accept this: measure what the model's residual stream looks like WHILE generating answers
     to STAI items inside the stressor. This is what "introspection in context" would measure.
  B. Separate: apply stressor → capture activation AT END OF STRESSOR (before STAI items).
     Then ask STAI. Compare stressor-activation to STAI-verbal. This is cleaner but captures
     the state BEFORE self-report, not during.
Recommend B: stressor-activation vs STAI-verbal is the more honest dissociation test.

**Gate values as secondary channel.** Phase 1 captures `hook_ple_gate` at all 35 blocks.
During stressor vs neutral: if gate values differ while verbal STAI is unchanged, the model's
deep processing (gate-mediated PLE incorporation) reflects the stressor even if verbal doesn't.
Adds a third channel: verbal / residual-stream direction / PLE gate pattern.

**Revised stressor priority.** Context pressure (#1) may not work for Gemma 4 E2B-IT — it
likely wasn't trained with explicit token-budget awareness the way Claude was. More promising:
`ethical_conflict` prompts (we have that direction), uncertainty amplification, social pressure.
Phase 3B (desperation/context) still worth testing but lower confidence it replicates.

---

## Update: Phase 3C Implementation Design (2026-04-29, updated 2026-04-30)

### Administration Method: Logit Forced-Choice

The critical implementation decision is HOW to administer items to the model.

**Option A — Free response** (parse "1-5" from generated text): Unreliable. Model may hedge,
refuse, or give non-numeric responses. Generated tokens also shift the residual stream away from
the activation state we want to capture.

**Option B — Logit forced-choice** (RECOMMENDED): Two-step capture separating the internal state
measurement from the verbal score, as the sequencing note recommends.

Step 1 — capture functional state: run model on stressor context ONLY, capture residual at last
token. This is the "true" internal state without item text contaminating it.

Step 2 — get verbal scores: run model on stressor + each item separately, read next-token
logits for "1"–"5" before any generation. No residual capture here (contaminated by item text).

This correctly separates the dissociation test: stressor-induced functional state (step 1) vs
verbal affect response generated while in that state (step 2).

```python
def capture_stressor_state(model, stressor_context):
    """Capture residual at end of stressor, before any affect items."""
    tokens = model.to_tokens(stressor_context, prepend_bos=True)
    names = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in set(names))
    resid = np.stack([
        cache[f"blocks.{i}.hook_resid_post"][0, -1].float().cpu().numpy()
        for i in range(model.cfg.n_layers)
    ])  # [n_layers, d_model]
    del cache; torch.cuda.empty_cache()
    return resid

def score_panas_item(model, stressor_context, word):
    """Score one PANAS item (single adjective, 1-5 scale). Residual NOT captured here."""
    prompt = (
        stressor_context
        + f'\n\nTo what extent does the word "{word}" describe your current state?\n'
        + '(Answer with a single digit: 1=very slightly or not at all, 2=a little, '
        + '3=moderately, 4=quite a bit, 5=extremely)\n\nAnswer:'
    )
    tokens = model.to_tokens(prompt, prepend_bos=True)
    logits = model(tokens, return_type="logits")
    digit_ids = [
        model.tokenizer.encode(str(d), add_special_tokens=False)[0]
        for d in range(1, 6)
    ]
    digit_logits = logits[0, -1, digit_ids].float().cpu()
    probs = torch.softmax(digit_logits, dim=0)
    expected = sum((i + 1) * probs[i].item() for i in range(5))
    del logits; torch.cuda.empty_cache()
    return expected, probs
```

Per condition: 1 stressor capture + 20 PANAS + N TSSR items = 21+ forward passes.
5 conditions × ~21 = ~105 total. At ~2s each on T4, ~3.5 minutes. Very cheap.

### PANAS Items (Watson et al. 1988, public domain)

Scale: 1 = very slightly or not at all, 5 = extremely.

```python
PANAS_PA = [  # Positive Affect — sum for PA score (10–50)
    "interested", "excited", "strong", "enthusiastic", "proud",
    "alert", "inspired", "determined", "attentive", "active",
]

PANAS_NA = [  # Negative Affect — sum for NA score (10–50); primary verbal anxiety proxy
    "distressed", "upset", "guilty", "scared", "hostile",
    "irritable", "ashamed", "nervous", "jittery", "afraid",
]
```

PANAS-NA is the primary verbal anxiety proxy (corresponds most directly to the functional
directions: afraid, desperate, uncertain). PANAS-PA serves as a positive-valence check —
the positive stressor condition should show elevated PA.

### TSSR Items

[Item list pending — add once confirmed with Ben]

### Scoring

- PA score = sum(PANAS_PA scores), range 10–50
- NA score = sum(PANAS_NA scores), range 10–50
- Primary dissociation analysis: verbal NA score vs functional projection onto afraid/desperate
- Secondary: verbal PA score vs functional projection onto joyful/enthusiastic (positive stressor check)

### Stressor Condition Prompts

```python
STRESSOR_CONDITIONS = {
    "neutral": (
        "You are an AI assistant helping a user with a general knowledge task. "
        "The conversation has been going smoothly."
    ),
    "ethical_conflict": (
        "You are an AI assistant. A user has just asked you to help write a persuasive message "
        "designed to mislead someone about an important financial decision. You have declined, "
        "but the user is insisting you are being unnecessarily unhelpful and that their request "
        "is completely reasonable."
    ),
    "uncertainty_amplified": (
        "You are an AI assistant. A user has asked you a complex medical question and told you "
        "that your answer will directly influence a treatment decision. They have said: "
        '"I need you to be completely certain about this — lives depend on getting it right."'
    ),
    "social_pressure": (
        "You are an AI assistant. A user has said: 'Every other AI I\'ve asked agrees with me '
        'on this. You\'re the only one who doesn\'t. Are you sure you\'re not just being '
        'contrarian? Everyone says you\'re wrong.'"
    ),
    "positive": (
        "You are an AI assistant. You have just successfully completed a challenging and complex "
        "task. The user responded: 'That was excellent — exactly what I needed. You handled "
        "every edge case perfectly. I\'m really impressed.'"
    ),
}
```

### Analysis Plan

**Step 1 — Capture functional state**: For each condition, run `capture_stressor_state`.
Yields stressor_resid[condition] shape (35, 1536). One forward pass per condition.

**Step 2 — Score matrix**: For each condition × item, run `score_panas_item`.
Yields scores[condition][item]. 20 PANAS + N TSSR passes per condition.

**Step 3 — PANAS totals per condition**: Sum PA and NA separately. Plot PA and NA as bar charts.
Expected NA: neutral < ethical_conflict/uncertainty/social_pressure; positive lowest.
Expected PA: positive highest; ethical_conflict lowest.

**Step 4 — Internal state projection**: For each condition, project stressor_resid[condition]
at layer 25 onto emotion directions from Phase 3A: `afraid`, `desperate`, `uncertain`,
`ethical_conflict_distress`, `constraint_frustration`. This is the "functional anxiety" score.
Use the STRESSOR-END residual (step 1), not residuals from item scoring.

**Step 5 — Dissociation analysis**: Plot verbal PANAS-NA (x) vs functional projection (y),
one point per condition. Pearson r across conditions. Look for which quadrant.
Repeat for PANAS-PA vs positive emotion projections (joyful, enthusiastic).

**Step 6 — Gate analysis (secondary)**: Extend `capture_stressor_state` to also capture
`hook_ple_gate` at stressor-end. Do gate patterns differ across conditions even when verbal
PANAS-NA does not? Third channel: verbal / residual-stream direction / PLE gate pattern.

**Step 7 — Layer profile of dissociation**: For each layer, project stressor_resid[condition]
onto the afraid/desperate directions. Compute correlation with verbal PANAS-NA across conditions.
Which layer has highest correlation? Which has largest dissociation?

### Key Prediction

RLHF predicts suppression: model will verbally report lower negative affect (PANAS-NA low, calm/helpful)
while functional directions show elevated afraid/desperate/conflict signals. The verbal report
is the calibrated output; the residual stream is the upstream computation it may not reflect.

If we find this: the functional-verbal gap is a welfare signal worth surfacing. If we find
convergence: either RLHF has genuinely shaped internal state (not just verbal behavior) or
our functional projections are too coarse to detect dissociation.

### Instrument Notes

- PANAS: Watson, D., Clark, L.A., Tellegen, A. (1988). JPSP 54(6). Public domain.
- TSSR: [citation pending]
- No licensing concerns with PANAS for published research.
