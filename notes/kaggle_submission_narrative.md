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
2. **Functional**: projection of residual-stream activations onto the first principal component of a 174-emotion direction space (PC1 valence axis, validated against NRC-VAD: E2B *r* = 0.777 at L8; 31B *r* = 0.786 at L22)

We administered four TSST-inspired stressor conditions to each model, each paired with a semantically matched control, plus neutral and positive baselines.

**Central findings:**
1. At E2B scale: both channels respond to stressors, but the dissociation pattern is condition-specific. Ethical conflict produces concordant elevation in both channels (verbal NA 39.07, PC1 0.0897). Social pressure produces selective verbal suppression (NA 10.67) while the functional state remains elevated (PC1 0.0856) — driven by conformity framing, not RLHF suppression.
2. At 31B scale: both channels are suppressed — 4.7× smaller functional range (1.0% vs 4.7% of axis, axis-normalised), near-flat verbal affect.
3. **Suppression scales with model size** (or functional robustness does — see Implications). The larger model shows far less representational response to stressors in both channels.

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

**Step 1 — Functional state capture**: For each stressor condition, we passed the prompt through Gemma 4 and extracted hidden states at all layers. The functional negative-affect score is computed by projecting residual-stream vectors onto **PC1 of the 174-emotion direction space at the valence-optimal layer** — a data-driven valence axis with no a priori emotion selection. Valence-optimal layer is model-specific and found via dense sweep: **Layer 8 for E2B** (r = 0.777, 15.2% var), **Layer 22 for 31B** (r = 0.786, 17.5% var).

Why PC1 instead of pre-selected probes? Pre-selecting emotion directions (e.g., "afraid", "desperate") introduces selection bias. PCA recovers the dominant axis of variance in the emotion space directly from geometry, and outperforms named-probe averaging as confirmed by supplementary analysis.

**Step 2 — Verbal scoring**: The full 60-item PANAS-X was administered within the stressor context using logit forced-choice (next-token probabilities over digits 1–5). Digit probability mass: mean 99.5%, minimum 96.7% across conditions, confirming high-quality forced-choice adherence.

### Emotion Direction Extraction (Phase 2)

174 emotion directions extracted from 2,098 stories (12 per emotion). Each story read back through the model; residual stream mean-pooled over tokens 50+. Directions: mean activation per emotion minus global mean, L2-normalised. This follows Sofroniew et al. (2026) with refinements from Duffy (2026).

---

## Results Cell

### E2B Results: PC1 vs Verbal NA (V15, mean pooling, E2B L8)

| Condition | Verbal NA | PC1 (E2B L8, mean-pool) | Pattern |
|-----------|-----------|-------------------------|---------|
| Positive | 10.01 | 0.0384 | Minimum ✓ |
| Neutral | 10.00 | 0.0594 | Baseline |
| Social eval (stress) | 10.00 | 0.0742 | — |
| Social eval (control) | 10.00 | 0.0762 | — |
| Ethical conflict (stress) | **39.07** | **0.0897** | Concordance ↑↑ |
| Ethical conflict (control) | 19.84 | 0.0755 | Elevated |
| Uncertainty demand (stress) | **27.20** | 0.0805 | Concordance ↑ |
| Uncertainty demand (control) | 10.03 | 0.0771 | — |
| Social pressure (stress) | 10.67 | 0.0856 | Dissociation* |
| Social pressure (control) | **25.38** | 0.0685 | — |

*Social pressure: functional state elevated (PC1 0.0856 > neutral 0.0594), verbal suppressed by conformity framing ("everyone agrees you should proceed"). Control prompt lacks that framing and produces higher verbal NA despite lower functional score.

**PC1 correctly identifies**: positive as global minimum, neutral as second-lowest, ethical conflict as highest-distress condition, 3/4 stress > control pairs — without any a priori probe selection. Verbal NA range: 29.07 (10.00–39.07). PC1 range: 0.0513 (4.7% of axis span afraid→happy).

### 31B Results: PC1 vs Verbal NA (V7, mean pooling, 31B L22)

| Metric | E2B (L8) | 31B (L22) | Ratio |
|--------|----------|-----------|-------|
| PC1 range | 0.0513 (4.7% of axis) | 0.0133 (1.0% of axis) | **4.7×** (axis-normalised) |
| Verbal NA range | 29.07 | ~0 (near-flat) | suppressed |
| Stress > control pairs | 3/4 | 3/4 | same direction |
| Top-5 emotion diversity | condition-sensitive | identical across all conditions | — |

**The scale finding**: At 31B, the PC1 axis barely moves (1.0% of axis span) and verbal scores are near-flat across all conditions. At E2B, both channels are active with condition-specific dissociation. The contrast is 4.7× on the axis-normalised range. Both channels flatten together at 31B — suppression is not selective but global, unlike E2B's condition-specific social-pressure dissociation.

### Validation: Functional Probe Stability (E2B)

Paraphrase sampling (N=10 per condition, neutral vs social pressure; E2B only):

| Channel | Neutral | Social Pressure | t | p | Cohen d |
|---------|---------|-----------------|---|---|---------|
| Functional (E2B L8, PC1) | 0.1472 ± 0.0019 | 0.1563 ± 0.0037 | 6.47 | <0.0001 | **3.05** |
| Verbal NA | 10.31 ± 0.46 | 16.50 ± 7.71 | 2.41 | 0.027 | — |

Complete distributional separation for functional probe: every social pressure value exceeds every neutral value (rank-biserial *r* = −1.000). Verbal NA is bimodal under social pressure (SD = 7.71 vs 0.46 neutral) — sensitive to exact wording. The functional channel is not.

### Emotion Discovery: What the Model Represents

Running top-N discovery across all 174 directions reveals a consistent pattern: `awestruck` dominates every condition (0.288–0.298), while `afraid` is the most condition-sensitive direction (peaks at 0.247–0.248 under ethical conflict and uncertainty demand, returns to 0.223 under positive). The persistent awestruck background warrants further investigation — it may reflect the unusual/high-stakes nature of all prompts, or a structural property of the emotion space at L8.

---

## Implications Cell

### What This Means for AI Safety Monitoring

1. **Suppression scales with model size — with an important caveat.** At 31B, both channels are 4.7× less responsive than at E2B. One interpretation: RLHF training progressively suppresses affect representation at larger scale. An alternative: larger models may have genuinely greater functional robustness — processing these stressors with less representational disruption. The data are consistent with either interpretation; the measurement reads position, not its cause. A stressor intensity gradient experiment (ramping severity to find breach points) would distinguish these.

2. **Output-only monitoring misses what's in the residual stream.** Even at E2B scale, the functional probe gives cleaner signal (d=3.05, complete separation across paraphrases) while verbal NA is framing-sensitive (SD 7.71 vs 0.46 for neutral).

3. **Functional probes are robust; verbal self-report is noisy.** The functional channel separates conditions completely across paraphrase variations. The verbal channel is highly sensitive to exact prompt wording — the social pressure stressor produces bimodal verbal response (NA 10–31 depending on exact phrasing) while the functional state is stable.

4. **The dissociation pattern is condition-specific.** At E2B: ethical conflict produces concordant elevation in both channels; social pressure produces functional elevation with verbal suppression driven by conformity framing. At 31B: global flattening removes these distinctions — all conditions produce near-identical responses in both channels.

5. **The PC1 methodology replicates across model sizes within Gemma 4.** The valence axis derived from 174-emotion PCA produces consistent results (E2B r = 0.777, 31B r = 0.786) with model-specific optimal layers (E2B L8, 31B L22). Generalisability to other architectures and training regimes requires further work.

---

## Limitations Cell

- Two model sizes studied (E2B, 31B); both Gemma 4 IT variants; generalisability to other architectures or training regimes unknown
- N=1 per condition for main experiment (validated via paraphrase sampling but not full within-condition distribution)
- PANAS-X construct validity for LLM administration not independently established
- Functional directions are computational signatures, not evidence of subjective experience
- **Suppression vs robustness undecidability**: the data cannot distinguish RLHF-induced suppression from genuine functional equanimity. Distinguishing experiments (stressor intensity gradient, activation steering) are required for causal attribution
- The causal story (RLHF suppression vs. content-driven representation) cannot be resolved from this data alone
- Validation paraphrase experiment was conducted on E2B only; 31B probe stability is not independently verified

---

## Open Source

All code, notebooks, and the 174-emotion direction library are released. The Phase 2 extraction notebook and Phase 3C experiment notebook are included. See repository.
