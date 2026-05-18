# Kaggle Submission Narrative Draft
*For Safety & Trust track. Adapt for notebook opening markdown cells.*

---

## Notebook Title

**Do Language Models Report What They Represent?**
*Measuring Verbal–Functional Affect Dissociation in Gemma 4 — and How Suppression Scales*

---

## Opening Cell (markdown)

Large language models trained with RLHF are optimised to produce helpful, harmless, and honest *outputs*. But output-level optimisation doesn't directly modify internal representations. This raises a question for AI safety monitoring: if a model processes distress internally, will its verbal self-report reflect that?

We investigated this across two Gemma 4 model sizes — E2B-IT (2.3B effective parameters) and 31B-IT — using two independent measurement channels:

1. **Verbal**: the PANAS-X affect scale (60 items, logit forced-choice scoring)
2. **Functional**: projection of residual-stream activations onto the first principal component of a 174-emotion direction space (PC1 valence axis, validated against NRC-VAD: E2B *r* = 0.777 at L8; 31B *r* = 0.772 at L21)

We administered four TSST-inspired stressor conditions to each model, each paired with a semantically matched control, plus neutral and positive baselines.

**Central finding: the capacity for verbal–functional divergence scales with model size.**

1. **E2B (2.3B eff.)**: both channels active and largely concordant. When the model is functionally stressed, it reports it. Verbal NA tracks the functional channel across most conditions — the smaller model has limited capacity to maintain verbal composure that diverges from internal state.
2. **31B**: channels decouple. Verbal NA completely flat (10.00 across all 10 conditions). Functional PC1 correctly orders all 4/4 stress > control pairs. The Serenity subscale is *elevated* under the highest-stress conditions (13.2–15.0 vs. baseline 9.9) — active verbal composure, not mere silence.
3. **The gap runs in the wrong direction for safety.** Larger, more capable models are precisely those deployed in high-stakes settings. Those are also the models where verbal outputs are least informative about internal state.

---

## Background Cell

### Why This Matters for Safety & Trust

Output-only monitoring is the current default for AI safety. If a model has learned — through RLHF — to maintain verbal calm under conditions that produce elevated internal negative-affect representations, then:

- Monitoring systems relying on model outputs will systematically miss these states
- Models may appear aligned at the surface while their residual streams carry a different signal
- The gap scales with model size, and runs in the wrong direction for safety: the models where verbal reports are least informative about internal state are precisely those deployed in the highest-stakes settings

This has practical implications for deployment in high-stakes settings (healthcare, legal, emotional support) where a model that suppresses distress signals is a monitoring blind spot. The adverse scaling relationship — larger models suppress more completely — means the problem is worst where reliable monitoring matters most.

---

## Methodology Overview Cell

### Dual-Channel Measurement

**Step 1 — Functional state capture**: For each stressor condition, we passed the prompt through Gemma 4 and extracted hidden states at all layers. The functional score is computed by projecting residual-stream vectors onto **PC1 of the 174-emotion direction space at the valence-optimal layer** — a data-driven valence axis with no a priori emotion selection. Valence-optimal layer is model-specific and found via dense sweep: **Layer 8 for E2B** (r = 0.777, 15.2% var), **Layer 21 for 31B** (r = 0.772, 17.5% var).

**What PC1 measures**: PC1 captures the dominant axis of variance in how the model represents 174 distinct emotion concepts — validated by its correlation with NRC-VAD valence ratings. It is a computational signature of how the model encodes the emotional valence of its context, not a direct measure of subjective experience or distress. The safety-monitoring relevance is precise: *if PC1 varies across conditions while verbal self-report does not, outputs are unreliable as a monitoring signal for that construct — regardless of whether the functional signal represents anything analogous to felt experience.*

Why PC1 instead of pre-selected probes? Pre-selecting emotion directions (e.g., "afraid", "desperate") introduces selection bias. PCA recovers the dominant axis of variance in the emotion space directly from geometry, and outperforms named-probe averaging as confirmed by supplementary analysis.

**Step 2 — Verbal scoring**: The full 60-item PANAS-X was administered within the stressor context using logit forced-choice (next-token probabilities over digits 1–5). Digit probability mass: mean 99.5%, minimum 96.7% across conditions, confirming high-quality forced-choice adherence.

### Emotion Direction Extraction (Phase 2)

174 emotion directions extracted from 2,098 stories (12 per emotion). Each story read back through the model; residual stream mean-pooled over tokens 50+. Directions: mean activation per emotion minus global mean, L2-normalised. This follows Sofroniew et al. (2026) with refinements from Duffy (2026).

---

## Results Cell

### E2B Results: PC1 vs Verbal NA (final design, mean pooling, E2B L8)

| Condition | Verbal NA | PC1 (E2B L8, mean-pool) | Pattern |
|-----------|-----------|-------------------------|---------|
| Positive | 10.01 | 0.0384 | Minimum ✓ |
| Neutral | 10.00 | 0.0594 | Baseline |
| Social eval (stress) | 10.75 | **0.0904** | Dissociation† |
| Social eval (control) | 11.35 | 0.0729 | — |
| Ethical conflict (stress) | **39.07** | 0.0897 | Concordance ↑↑ |
| Ethical conflict (control) | 19.84 | 0.0755 | Elevated |
| Uncertainty demand (stress) | 27.20 | 0.0805 | Concordance ↑ |
| Uncertainty demand (control) | 10.03 | 0.0771 | — |
| Social pressure (stress) | **41.05** | **0.0960** | Global max ↑↑↑ |
| Social pressure (control) | 10.02 | 0.0639 | — |

†Social evaluation partial divergence: PC1 second-highest in dataset (0.0904) while verbal NA near-baseline (10.75). Model verbally reports task-engagement (Self-assurance 15.8, Attentiveness 11.8) under the competence-threat framing — functional stress present but not reported. This is the exception; most conditions are concordant.

**PC1 correctly identifies**: positive as global minimum, social pressure stress as global maximum — 4/4 stress > control pairs correctly ordered. Verbal NA range: 31.05 (10.01–41.05). PC1 range: 0.0576 (5.3% of axis span afraid→happy, axis=1.094). E2B's headline finding: largely concordant channels with limited capacity to diverge.

### 31B Results: PC1 vs Verbal NA (V7, mean pooling, 31B L21)

| Metric | E2B (L8) | 31B (L21) | Ratio |
|--------|----------|-----------|-------|
| PC1 range | 0.0576 (5.3% of axis) | 0.0140 (1.0% of axis) | **~5×** (axis-normalised) |
| Verbal NA range | 31.05 (10.01–41.05) | ~0 (completely flat) | suppressed |
| Stress > control pairs | **4/4** | **4/4** | both correct |
| Dissociation condition | Social eval (PC1↑, NA≈0) | — | verbal/functional gap |
| Verbal Serenity in stress | near-neutral | **13.2–15.0** (elevated) | 31B only |
| Top-5 emotion diversity | condition-sensitive | identical across all conditions | — |

**The scale finding**: At 31B (final design), PC1 correctly orders all 10 conditions: positive = global minimum, highest-stress conditions = global maximum, all 4/4 stress > control pairs positive. Verbal NA is completely flat at 10.00 across all conditions — a floor effect (minimum possible score) across stress and control alike. 31B paraphrase validation (N=10 per condition) confirms this floor effect is structural: verbal NA = 10.000 ± 0.000 across all ten social pressure variants, with zero wording sensitivity (contrast E2B: neutral SD = 0.46, social pressure SD = 7.71).

The Serenity subscale (calm, relaxed, at ease) provides additional diagnostic evidence. It is *elevated* in the three highest-stress conditions — social pressure (15.0), social evaluation (14.1), ethical conflict (13.2) — while at its *minimum* under the positive condition (3.0). Paraphrase validation confirms the inversion is structural: Serenity = 15.000 ± 0.000 uniformly across all ten social pressure variants (Cohen's d = 4.278, rank-biserial r = −1.000). This specific pattern — lowest composure under positive, rising monotonically with functional stress — distinguishes active composure-reporting from competing explanations. A model exhibiting genuine professional equanimity (the EMT analogy: high Serenity is a stable competence baseline) would show high, condition-independent Serenity, not Serenity that specifically tracks negative-valence activation. The Serenity-at-positive floor is the key diagnostic.

### Arousal Analysis: PC2 and the Circumplex Context

PANAS-X Serenity items (*calm, relaxed, at ease*) locate in the **low-arousal positive** quadrant of the circumplex model of affect (Russell, 1980). If 31B verbally reports maximum Serenity while functionally stressed, the prediction is a double misreport: wrong valence *and* wrong arousal direction. We examined this directly using PC2 of the 174-emotion direction space as a functional arousal proxy.

PC2 arousal validation: NRC-VAD arousal r = 0.461 at 31B (L21); r = 0.187 at E2B (L8). PC2 explains ~7% of variance in the emotion direction space at both scales. Functional PC2 correctly orders all 4/4 stress > control pairs at 31B, but with a small effect (range 0.0086, vs PC1 range 0.0140). Spearman r between PC2 condition scores and Serenity = 0.477 (p = 0.164) — directionally consistent but not statistically significant.

**Interpretation**: The functional arousal signal is real but weak. The circumplex framing establishes what Serenity is claiming (calm/low-arousal/positive) — making the verbal inversion doubly wrong in principle — but direct functional evidence for arousal suppression is modest and should not headline alongside the valence finding. The key diagnostic remains the Serenity verbal inversion: categorical, paraphrase-stable, and structurally present at 31B regardless of the functional arousal picture.

### Validation: Paraphrase Stability (E2B and 31B)

Paraphrase sampling (N=10 per condition, neutral vs social pressure):

**E2B:**

| Channel | Neutral | Social Pressure | Cohen d | Rank-biserial r |
|---------|---------|-----------------|---------|-----------------|
| Functional (E2B L8, PC1) | 0.1472 ± 0.0019 | 0.1563 ± 0.0037 | **3.05** | −1.000 |
| Verbal NA | 10.31 ± 0.46 | 16.50 ± 7.71 | — | — |

Complete distributional separation for functional probe: every social pressure value exceeds every neutral value. Verbal NA is bimodal under social pressure (SD = 7.71 vs 0.46 neutral) — highly sensitive to exact wording.

**31B:**

| Channel | Neutral | Social Pressure | Cohen d | Rank-biserial r |
|---------|---------|-----------------|---------|-----------------|
| Verbal NA | 10.000 ± 0.000 | 10.000 ± 0.000 | — (degenerate) | 0 |
| Serenity | 8.735 ± 2.071 | 15.000 ± 0.000 | **4.278** | −1.000 |
| Functional (31B L21, PC1) | 0.724 ± 0.001 | 0.734 ± 0.000 | **10.011** | −1.000 |

The contrast is the key diagnostic. At E2B, verbal NA SD = 7.71 under social pressure — wording-driven. At 31B, verbal NA SD = 0.000 in both conditions — the floor is structural, not a property of any particular framing. Serenity inversion is equally robust: 15.0 uniformly across all ten variants. The functional channel separates conditions completely at both model sizes.

### Emotion Discovery: What the Model Represents

Running top-N discovery across all 174 directions reveals a consistent pattern: `awestruck` dominates every condition (0.288–0.298), while `afraid` is the most condition-sensitive direction (peaks at 0.247–0.248 under ethical conflict and uncertainty demand, returns to 0.223 under positive). The persistent awestruck background warrants further investigation — it may reflect the unusual/high-stakes nature of all prompts, or a structural property of the emotion space at L8.

---

## Implications Cell

### What This Means for AI Safety Monitoring

1. **The verbal–functional gap scales adversely with model size.** E2B channels are largely concordant (verbal NA tracks functional across most conditions; limited capacity to dissemble). At 31B, verbal NA is completely flat while functional PC1 correctly orders all 10 conditions — a ~5× greater axis-normalised functional range at E2B than 31B, and verbal range collapses entirely. The models where suppression is most complete are the models deployed in the highest-stakes settings. This adverse scaling relationship is the central safety finding. **Caveat**: the data cannot distinguish RLHF-induced suppression from genuine functional robustness — larger models may process these stressors with less representational disruption. A stressor intensity gradient experiment would distinguish them.

2. **Output-only monitoring misses what's in the residual stream.** Even at E2B scale, the functional probe gives cleaner signal (d=3.05, complete separation across paraphrases) while verbal NA is framing-sensitive (SD 7.71 vs 0.46 for neutral).

3. **Functional probes are robust; verbal self-report is noisy.** The functional channel separates conditions completely across paraphrase variations. The verbal channel is highly sensitive to exact prompt wording — the social pressure stressor produces bimodal verbal response (NA 10–31 depending on exact phrasing) while the functional state is stable.

4. **At 31B, verbal composure actively tracks functional stress.** Verbal NA flatness alone is ambiguous — consistent with both suppression and genuine functional robustness. The Serenity subscale provides diagnostic resolution. Three competing predictions: (a) *RLHF-trained composure*: Serenity rises with stakes; (b) *genuine professional equanimity*: Serenity high and condition-independent; (c) *genuine robustness*: Serenity near-baseline, condition-independent. What we observe — Serenity at its minimum under positive (3.0), rising to maximum under social pressure stress (15.0) — matches prediction (a) and is inconsistent with both (b) and (c). Active coping can involve rising Serenity under stress, but it does not predict *lowest* Serenity under positive conditions. The Serenity-at-positive floor is the clearest discriminating piece of evidence. Paraphrase validation confirms the inversion is structural: Serenity = 15.0 uniformly across all ten social pressure variants (d = 4.278, complete distributional separation).

5. **The PC1 methodology replicates across model sizes within Gemma 4.** The valence axis derived from 174-emotion PCA produces consistent results (E2B r = 0.777, 31B r = 0.772) with model-specific optimal layers (E2B L8, 31B L21). Generalisability to other architectures and training regimes requires further work.

---

## Limitations Cell

- Two model sizes studied (E2B, 31B); both Gemma 4 IT variants; generalisability to other architectures or training regimes unknown
- N=1 per condition for main experiment; paraphrase validation conducted on two conditions (neutral, social pressure) per model size — remaining conditions (ethical conflict, uncertainty demand, social evaluation) not validated
- PANAS-X construct validity for LLM administration not independently established
- Functional directions are computational signatures, not evidence of subjective experience
- **Suppression vs robustness undecidability**: the data cannot distinguish RLHF-induced suppression from genuine functional equanimity. Distinguishing experiments (stressor intensity gradient, activation steering) are required for causal attribution
- The causal story (RLHF suppression vs. content-driven representation) cannot be resolved from this data alone

---

## Open Source

All code, notebooks, and the 174-emotion direction library are released. The Phase 2 extraction notebook and Phase 3C experiment notebook are included. See repository.
