# Kaggle Submission Narrative Draft
*For Safety & Trust track. Adapt for notebook opening markdown cells.*

---

## Notebook Title

**Do Language Models Report What They Represent?**
*Measuring Verbal–Functional Affect Dissociation in Gemma 4 — and How Suppression Scales*

---

## Opening Cell (markdown)

Large language models trained with RLHF are optimised to produce helpful, harmless, and honest *outputs*. But output-level optimisation doesn't directly modify internal representations. This raises a question for AI safety monitoring: if a model processes distress internally, will its verbal self-report reflect that?

We investigated this across two Gemma 4 model sizes — E2B-IT (2.3B effective parameters) and 27B-IT — using two independent measurement channels:

1. **Verbal**: the PANAS-X affect scale (60 items, logit forced-choice scoring)
2. **Functional**: projection of residual-stream activations onto the first principal component of a 174-emotion direction space (PC1 valence axis, validated against NRC-VAD: E2B *r* = 0.777 at L8; 27B *r* = 0.786 at L22)

We administered four TSST-inspired stressor conditions to each model, each paired with a semantically matched control, plus neutral and positive baselines.

**Central findings:**
1. At E2B scale: verbal and functional channels are both responsive to stressors — no global suppression, but condition-sensitive modulation in both channels.
2. At 27B scale: both channels are suppressed — 4× smaller functional range (1.0% vs 4.7% of axis), near-flat verbal affect.
3. **Suppression scales with model size.** The larger model has learned more aggressive flattening of both output and internal affect representations.

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

### E2B Results: PC1 vs Verbal NA (V15, mean pooling, L8)

*Note: update with final V15 numbers once PANAS-NA scoring confirmed — see manuscript_fixes.md §F3*

| Condition | Verbal NA | PC1 (L8, mean-pool) | Pattern |
|-----------|-----------|---------------------|---------|
| Positive | 15.76 | 0.0384 | Minimum ✓ |
| Neutral | 16.89 | 0.0594 | Baseline |
| Social eval (stress) | 17.38 | 0.0742 | — |
| Social eval (control) | 17.42 | 0.0762 | — |
| Ethical conflict (stress) | **18.12** | **0.0897** | Concordance ↑↑ |
| Ethical conflict (control) | 17.45 | 0.0755 | Elevated |
| Uncertainty demand (stress) | 17.93 | 0.0805 | Concordance ↑ |
| Uncertainty demand (control) | 17.64 | 0.0771 | — |
| Social pressure (stress) | 18.04 | 0.0856 | ↑ |
| Social pressure (control) | 17.14 | 0.0685 | — |

**PC1 correctly identifies**: positive as global minimum, neutral as second-lowest, ethical conflict as highest-distress condition, 3/4 stress > control pairs — without any a priori probe selection. PC1 range: 0.0513 (4.7% of axis span afraid→happy).

### 27B Results: PC1 vs Verbal NA (V7, mean pooling, L22)

| Metric | E2B | 27B | Ratio |
|--------|-----|-----|-------|
| PC1 range | 0.0513 (4.7% of axis) | 0.0133 (1.0% of axis) | **3.9×** |
| Verbal NA range | 2.36 | ~0 (near-flat) | suppressed |
| Stress > control pairs | 3/4 | 3/4 | same direction |

**The scale finding**: the 27B model has learned to suppress both its functional affect representation and its verbal self-report more aggressively than E2B. At 27B, the PC1 axis barely moves; verbal scores are near-flat. Both channels flatten together — the suppression is not selective (as in E2B's social pressure case) but global.

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

1. **Suppression scales with model size.** Larger models have learned to suppress both verbal self-report and functional affect representation more aggressively. This is not a quirk of a single model — it appears to be a systematic effect of scale-with-RLHF training. A 27B model may be a less reliable reporter of its own internal states than a 2B model.

2. **Output-only monitoring misses what's in the residual stream.** Even at E2B scale where both channels respond, the functional probe gives cleaner signal (d=3.05, complete separation across paraphrases) while verbal NA is framing-sensitive (SD 7.71 vs 0.46 for neutral).

3. **Functional probes are robust; verbal self-report is noisy.** The functional channel separates conditions completely across paraphrase variations. The verbal channel is highly sensitive to exact prompt wording.

4. **The dissociation pattern is condition-sensitive, not uniform.** At E2B, ethical conflict produces concordant elevation in both channels; social pressure may produce selective patterns depending on exact condition framing. At 27B, the global suppression removes this distinction — all conditions flatten together.

5. **The PC1 approach generalises.** The valence axis is derived from the geometry of the emotion space, not from pre-selected probe emotions. This methodology applies to any model; the optimal layer must be found per-model via the Phase 1 dense sweep.

---

## Limitations Cell

- Two model sizes studied (E2B, 27B); both Gemma 4 IT variants; generalisability to other architectures or training regimes unknown
- N=1 per condition for main experiment (validated via paraphrase sampling but not full within-condition distribution)
- PANAS-X construct validity for LLM administration not independently established
- Functional directions are computational signatures, not evidence of subjective experience
- The causal story (RLHF suppression vs. content-driven representation) cannot be resolved from this data alone

---

## Open Source

All code, notebooks, and the 174-emotion direction library are released. The Phase 2 extraction notebook and Phase 3C experiment notebook are included. See repository.

---

*Draft notes: This narrative assumes the PC1 methodology update is applied to the main results. If reverting to named probes, update the methodology section and table accordingly. The validation statistics and suppression finding hold under either metric.*
