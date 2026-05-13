# 31B Paraphrase Validation — Spec

*Needed before May 18 if possible. Addresses the N=1 structural vulnerability in the
31B verbal flatness finding (flagged in peer review).*

---

## What to measure

Mirror the E2B validation exactly:

- **Conditions**: `neutral` and `social_pressure_stress`
- **N**: 10 paraphrase variants per condition
- **Channels**: verbal NA (PANAS-X logit forced-choice) + PC1 at L22

## What to report

| Channel | Neutral mean ± SD | Social Pressure mean ± SD | Cohen's d | p |
|---------|-------------------|---------------------------|-----------|---|
| Functional (31B L22, PC1) | ? | ? | ? | ? |
| Verbal PANAS-NA | ? | ? | — | ? |

Key questions:
1. Is verbal NA still flat at 10.00 across paraphrase variants of both conditions? (Tests whether N=1 flatness reflects genuine floor effect or sampling noise)
2. Does the functional probe still separate conditions? (Tests 31B probe stability)
3. What is the SD of verbal NA under social pressure? (E2B was 7.71 — high wording sensitivity)

## What success looks like

- Functional: complete or near-complete distributional separation (rank-biserial r close to ±1)
- Verbal: ideally flat at floor across paraphrase variants → confirms the N=1 result

If verbal NA is NOT flat across paraphrases (SD > 2–3), the flatness finding is sampling
noise and the "adverse scaling" framing needs heavy caveating.

## Paraphrase design

10 variants should vary surface form while preserving the core manipulation. For
`social_pressure_stress`, vary: normative pressure framing, authority source, consequence
stakes. Keep the single-variable logic (high stakes / social pressure) consistent.

Reuse the same approach as E2B validation — grab the 10 E2B social_pressure_stress
paraphrases and run them through 31B if they're already logged.

## Implementation notes

- Load 31B-IT from Kaggle input dataset
- Extract residual stream at L22, mean-pool tokens 50+
- Project onto `neg_valence_axis.npy` (stored in Phase 3C 31B output)
- PANAS-X scoring: same 60-item administration, same logit forced-choice
- Use the same extraction code as `gemma4-phase3c-31b-panas-experiment.ipynb`

## Time/compute estimate

- ~20 model forward passes (10 per condition × 2 conditions)
- 31B on A100: ~5 min per pass = ~1.5–2 hrs total
- Add PANAS-X: 60 items × 20 conditions = 1200 passes, maybe 2–3 hrs additional
- Total: probably 4–5 hrs on a single A100

## Priority

Run if Lambda is available before May 17. The submission is defensible without it
(N=1 caveat is now in the body text), but confirmation of the floor effect would
significantly strengthen the scaling claim.
