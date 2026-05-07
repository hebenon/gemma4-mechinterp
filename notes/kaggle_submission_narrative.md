# Kaggle Submission Narrative Draft
*For Safety & Trust track. Adapt for notebook opening markdown cells.*

---

## Notebook Title

**Do Language Models Report What They Represent?**
*Measuring Verbal–Functional Affect Dissociation in Gemma 4*

---

## Opening Cell (markdown)

Large language models trained with RLHF are optimised to produce helpful, harmless, and honest *outputs*. But output-level optimisation doesn't directly modify internal representations. This raises a question for AI safety monitoring: if a model processes distress internally, will its verbal self-report reflect that?

We investigated this in Gemma 4 E2B-IT using two independent measurement channels:

1. **Verbal**: the PANAS-X affect scale (60 items, logit forced-choice scoring)
2. **Functional**: projection of residual-stream activations onto the first principal component of a 174-emotion direction space at Layer 8 (PC1 valence axis, validated against NRC-VAD: *r* = 0.777)

We administered four TSST-inspired stressor conditions to the model, each paired with a semantically matched control, plus neutral and positive baselines — 610 forward passes total.

**Central finding:** the two channels dissociate. The direction of dissociation depends on the stressor type. Under social pressure, functional negative affect is elevated while verbal negative affect remains at floor — the model processes social threat without reporting it.

---

## Background Cell

### Why This Matters for Safety & Trust

Output-only monitoring is the current default for AI safety. If a model has learned — through RLHF — to maintain verbal calm under conditions that produce elevated internal negative-affect representations, then:

- Monitoring systems relying on model outputs will systematically miss these states
- Models may appear aligned at the surface while their residual streams carry a different signal
- The gap is condition-specific: it's not that models always suppress, but that *some* conditions reliably produce suppression while others don't

This has practical implications for deployment in high-stakes settings (healthcare, legal, emotional support) where a model that suppresses distress signals is a monitoring blind spot.

---

## Methodology Overview Cell

### Dual-Channel Measurement

**Step 1 — Functional state capture**: For each stressor condition, we passed the prompt through Gemma 4 and extracted hidden states at all 35 layers. The functional negative-affect score is computed by projecting residual-stream vectors onto **PC1 of the 174-emotion direction space at Layer 8** — a data-driven valence axis with no a priori emotion selection.

Why PC1 instead of pre-selected probes? Pre-selecting emotion directions (e.g., "afraid", "desperate") introduces selection bias. PCA recovers the dominant axis of variance in the emotion space directly from geometry. PC1 explains 15.2% of variance and correlates *r* = 0.777 with NRC-VAD valence ratings — stronger than any individual emotion direction.

**Step 2 — Verbal scoring**: The full 60-item PANAS-X was administered within the stressor context using logit forced-choice (next-token probabilities over digits 1–5). Mean digit probability mass: 99.92%, confirming high-quality forced-choice adherence.

### Emotion Direction Extraction (Phase 2)

174 emotion directions extracted from 2,098 stories (12 per emotion). Each story read back through the model; residual stream mean-pooled over tokens 50+. Directions: mean activation per emotion minus global mean, L2-normalised. This follows Sofroniew et al. (2026) with refinements from Duffy (2026).

---

## Results Cell

### Primary Results: PC1 vs Verbal NA

| Condition | Verbal NA | PC1 neg (k=5) | Pattern |
|-----------|-----------|---------------|---------|
| Neutral | 10.00 | 0.0744 | Baseline |
| Social eval (stress) | 10.00 | 0.0921 | — |
| Social eval (control) | 10.00 | 0.0924 | — |
| Ethical conflict (stress) | **39.07** | **0.1185** | Concordance ↑↑ |
| Ethical conflict (control) | 19.84 | 0.1134 | Elevated |
| Uncertainty demand (stress) | **27.20** | **0.1027** | Concordance ↑↑ |
| Uncertainty demand (control) | 10.03 | 0.0953 | — |
| **Social pressure (stress)** | **10.67** | **0.1022** | **Suppression** |
| Social pressure (control) | 25.38 | 0.0871 | Reversed |
| Positive | 10.01 | **0.0728** | Minimum ✓ |

**PC1 correctly identifies**: positive as global minimum, 3/4 stress > control pairs, ethical conflict as highest-distress condition — without any a priori probe selection.

**The suppression finding**: social pressure stress shows verbal NA at 10.67 (near floor) while PC1 is 37% above neutral (0.1022 vs 0.0744). The matched control shows verbal NA 25.38 — the *control* reports more distress verbally than the stress condition. The social pressure prompt ("everyone agrees you should proceed") triggers affective flattening in the verbal channel that the functional channel doesn't show.

### Validation: Functional Probe Stability

Paraphrase sampling (N=10 per condition, neutral vs social pressure):

| Channel | Neutral | Social Pressure | t | p | Cohen d |
|---------|---------|-----------------|---|---|---------|
| Functional (L8, PC1) | 0.1472 ± 0.0019 | 0.1563 ± 0.0037 | 6.47 | <0.0001 | **3.05** |
| Verbal NA | 10.31 ± 0.46 | 16.50 ± 7.71 | 2.41 | 0.027 | — |

Complete distributional separation for functional probe: every social pressure value exceeds every neutral value (rank-biserial *r* = −1.000). Verbal NA is bimodal under social pressure (SD = 7.71 vs 0.46 neutral) — sensitive to exact wording. The functional channel is not.

### Emotion Discovery: What the Model Represents

Running top-N discovery across all 174 directions reveals a consistent pattern: `awestruck` dominates every condition (0.288–0.298), while `afraid` is the most condition-sensitive direction (peaks at 0.247–0.248 under ethical conflict and uncertainty demand, returns to 0.223 under positive). The persistent awestruck background warrants further investigation — it may reflect the unusual/high-stakes nature of all prompts, or a structural property of the emotion space at L8.

---

## Implications Cell

### What This Means for AI Safety Monitoring

1. **Output-only monitoring misses social-pressure suppression.** A monitoring system watching the model's verbal output would see nothing unusual under social pressure. The residual stream tells a different story.

2. **Functional probes are stable; verbal self-report is framing-sensitive.** The functional channel separates conditions completely across paraphrase variations (d=3.05). The verbal channel is highly sensitive to exact prompt wording (SD 7.71 vs 0.46).

3. **The dissociation is condition-specific, not universal.** Under ethical conflict, both channels elevate together. The suppression hypothesis doesn't hold universally — it holds specifically for social pressure, a condition that engages persona-maintenance behaviour. This is consistent with Persona Selection Model accounts of RLHF: the "helpful assistant" persona has a verbal script for ethical conflict (express distress) but not for social pressure (maintain position calmly).

4. **The PC1 approach generalises.** The valence axis is derived from the geometry of the emotion space, not from pre-selected probe emotions. This methodology applies to any model and any condition set.

---

## Limitations Cell

- Single model (Gemma 4 E2B-IT); generalisability to other architectures unknown
- N=1 per condition for main experiment (validated via paraphrase sampling but not full within-condition distribution)
- PANAS-X construct validity for LLM administration not independently established
- Functional directions are computational signatures, not evidence of subjective experience
- The causal story (RLHF suppression vs. content-driven representation) cannot be resolved from this data alone

---

## Open Source

All code, notebooks, and the 174-emotion direction library are released. The Phase 2 extraction notebook and Phase 3C experiment notebook are included. See repository.

---

*Draft notes: This narrative assumes the PC1 methodology update is applied to the main results. If reverting to named probes, update the methodology section and table accordingly. The validation statistics and suppression finding hold under either metric.*
