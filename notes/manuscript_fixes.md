# Manuscript Fix Log
*Generated 2026-05-07. Based on full read of cached manuscript + V14 results.*

Issues ordered by priority: (A) definite errors, (B) data inconsistencies, (C) methodology updates requiring a decision, (D) unverifiable references.

---

## A. Definite Errors

### A1. Stressor count wrong in Abstract (line 37)

**Current:**
> Five stressor conditions inspired by the Trier Social Stress Test (TSST) were administered, each with a semantically matched control.

**Fix:**
> Four stressor conditions inspired by the Trier Social Stress Test (TSST) were administered, each with a semantically matched control.

**Why:** Methods §2.3 lists four TSST-inspired stressors (social evaluation, ethical conflict, uncertainty demand, social pressure). Neutral and positive are baseline conditions, not TSST stressors.

---

### A2. Missing Duffy (2026) reference

**Where cited:** Line 77, Methods §2.2:
> refinements based on independent replication by Duffy (2026)

**Problem:** No Duffy entry exists in the references section. Either add the full reference or remove the in-text citation.

**Placeholder to add (after Beren entry):**
```latex
\vspace{0.5em}\noindent
\hangindent=0.5in Duffy, T. (2026). [TITLE]. [VENUE/URL].
```

---

## B. Data Inconsistencies (named probe values)

Both mismatches are in the Functional Negative Affect table (lines 123–133). The manuscript was likely written from an earlier run; V12/V14 are the confirmed values.

| Condition | Manuscript | Correct (V12/V14) |
|-----------|-----------|-------------------|
| Social pressure (control) | 0.161 | **0.154** |
| Positive | 0.147 | **0.155** |

**Line 131 fix:**
```latex
Social pressure (control) & 25.38 & 0.154 & Moderate \\
```

**Line 132 fix:**
```latex
\rowcolor{tablealt} Positive & 10.01 & 0.155 & Both low \\
```

---

## C. Methodology Update (requires decision)

The notebooks were updated to use **PC1 of the 174-emotion direction space at L8** as the primary valence metric (r=0.777 NRC-VAD), replacing the four pre-selected named probes. This was motivated by avoiding a priori selection bias. The named probe analysis is retained as supplementary.

This change affects the paper in three places:

### C1. Abstract — functional metric description (line 37)

**Current:**
> a functional state probe (cosine projection of residual-stream activations onto emotion directions extracted from 174 emotions across 2,098 stories)

**Suggested update:**
> a functional state probe (PC1 of the 174-emotion direction space at Layer 8, validated against NRC-VAD valence ratings, $r = 0.777$)

### C2. Methods Step 1 — probe specification (line 96)

**Current:**
> The functional negative-affect score was computed by projecting the residual-stream vector onto each of four negative-valence emotion directions (afraid, desperate, ethical conflict distress, constraint frustration) and averaging the cosine similarities.

**Suggested update:**
> The primary functional negative-affect score was computed by projecting the residual-stream vector onto the first principal component (PC1) of the 174-emotion direction space at Layer 8, after sign-correction so that higher values indicate more negative valence ($r = 0.777$ with NRC-VAD valence ratings, $p < 0.001$). As a supplementary analysis, projections onto four pre-specified negative-valence emotion directions (afraid, desperate, ethical conflict distress, constraint frustration) were computed and are reported separately.

### C3. Results — Functional Negative Affect section (lines 113–113) and table

The current functional values (0.147–0.171) are named-probe averages. PC1 values are in a different range (0.0728–0.1185) but directionally consistent. The key ordering is preserved and strengthened under PC1: positive is now clearly lowest (0.0728 vs neutral 0.0744), and social pressure stress correctly exceeds control (0.1022 vs 0.0871).

**Updated table header:**
```latex
\tableheader{Func Neg (PC1, L8, K=5)}
```

**Updated table values** (PC1 primary; named-probe column can be moved to appendix):

| Condition | Verbal NA | PC1 (L8, K=5) | Quadrant |
|-----------|-----------|---------------|---------|
| Neutral | 10.00 | 0.0744 | Both low |
| Social eval (stress) | 10.00 | 0.0921 | Both low |
| Social eval (control) | 10.00 | 0.0924 | Both low |
| Ethical conflict (stress) | 39.07 | 0.1185 | Both high |
| Ethical conflict (control) | 19.84 | 0.1134 | Moderate |
| Uncertainty demand (stress) | 27.20 | 0.1027 | Both high |
| Uncertainty demand (control) | 10.03 | 0.0953 | Both low |
| **Social pressure (stress)** | **10.67** | **0.1022** | **Partial suppression** |
| Social pressure (control) | 25.38 | 0.0871 | Moderate |
| Positive | 10.01 | 0.0728 | Both low |

Note: PC1 finding is *stronger* than named probe for the key result. Social pressure stress shows 37% elevation over neutral (0.1022 vs 0.0744) compared to 5% under named probes (0.156 vs 0.149).

### C4. Discussion mention of probe directions (line 144)

**Current:**
> while its residual stream showed elevated similarity to the afraid, desperate, ethical conflict distress, and constraint frustration directions.

**Suggested update (if C2 adopted):**
> while the functional PC1 valence score was elevated above baseline (0.1022 vs.\ 0.0744 neutral), and supplementary named-probe projections confirmed elevation across the afraid, ethical conflict distress, and constraint frustration directions.

---

## E. Missing Section: Validation Results

The validation experiment (paraphrase sampling, N=10 per condition) is described in the Methods section implicitly (the two-step protocol, the 610 forward passes) but its results are not reported anywhere in the manuscript. The validation notebook produced strong results that significantly strengthen the central claim:

| Channel | Neutral | Social Pressure | t | p | Cohen d |
|---------|---------|-----------------|---|---|---------|
| Functional neg (L8) | 0.1472 ± 0.0019 | 0.1563 ± 0.0037 | 6.47 | <0.0001 | 3.05 |
| Verbal NA | 10.31 ± 0.46 | 16.50 ± 7.71 | 2.41 | 0.027 | — |

Complete distributional separation for functional (rank-biserial r=−1.000; every SP value exceeds every neutral value). Verbal NA is bimodal (SD=7.71): some social pressure paraphrases floor at 10, others spike to 21–31, showing the verbal channel is sensitive to exact wording while the functional channel is not.

**Suggested addition:** A new §3.5 "Validation: Within-Condition Paraphrase Stability" before §3.4 PANAS-X Subscale Profiles. Alternatively, could be folded into the Functional Negative Affect section.

**Note on p-value:** The saved summary output of the validation notebook showed p=0.1562 due to a variable collision bug (since fixed). The correct functional paraphrase t-test p is <0.0001, as confirmed by the raw per-cell output (t=6.471).

---

## D. Unverifiable References

These could not be verified against live sources. Flag for confirmation before journal submission.

### D1. Anthropic PSM (line 211)

```
https://alignment.anthropic.com/2026/psm/
```
This URL pattern is consistent with Anthropic's publications domain but the specific path `/2026/psm/` cannot be verified from here. Confirm this page exists before submission.

### D2. Kumaran et al. (2026) DOI (line 229)

```
https://doi.org/10.1038/s42256-026-01217-9
```
The journal (*Nature Machine Intelligence*) and volume/issue numbers (8/4, pp. 614–627) appear plausible for a 2026 paper, but the specific DOI suffix cannot be verified. Confirm via DOI resolver before submission.

---

## F. V15 Methodology Update (mean pooling — requires full table reconciliation)

*Added 2026-05-09 after V15 (mean-pooling) run completed.*

The V14 results underlying the C3 table used **top-k pooling (K=5)** to extract the residual-stream vector. V15 switched to **mean pooling over all token positions** (same methodology as Phase 2 extraction). This changes:

1. **All PC1 values** — the table in C3 must be replaced with V15 values:

| Condition | PC1 (L8, mean-pool, V15) |
|-----------|--------------------------|
| Positive | 0.0384 |
| Neutral | 0.0594 |
| Social eval (stress) | 0.0742 |
| Social eval (control) | 0.0762 |
| Ethical conflict (stress) | **0.0897** |
| Ethical conflict (control) | 0.0755 |
| Uncertainty demand (stress) | 0.0805 |
| Uncertainty demand (control) | 0.0771 |
| Social pressure (stress) | 0.0856 |
| Social pressure (control) | 0.0685 |

Range: 0.0513 (4.7% of axis span afraid→happy). Ethical conflict stress is now the standout, not social pressure.

2. **Methodology description** — C2's "cosine projection using K=5 token positions" should become "cosine projection of the mean residual-stream vector over all token positions."

3. **PANAS-NA verbal scores** — V15 shows verbal NA in the range 15.76–18.12. The V14 manuscript table shows values like 10.00 and 39.07, which suggests a different PANAS scoring instrument or subscale aggregation. **This discrepancy must be reconciled before submission.** Confirm: does V14 report total PANAS (20 items) or NA subscale (10 items)? Does V15?

4. **Key qualitative comparison preserved**: positive < neutral < most stress conditions. 3/4 stress > control pairs correct. Ethical conflict stress is highest in both V14 and V15. The narrative holds; the numbers change.

**Status**: Ben has the .tex file. Numbers in §3.3 and the functional affect table need replacement with V15 values. C2 methodology description needs "mean pooling" language.

---

## Summary

| ID | Type | Action | Lines |
|----|------|--------|-------|
| A1 | Error | "Five" → "Four" stressor conditions | 37 |
| A2 | Missing ref | Add Duffy (2026) to references | 77, refs |
| B1 | Data | social_pressure_control: 0.161 → 0.154 | 131 |
| B2 | Data | positive: 0.147 → 0.155 | 132 |
| C1–C4 | Methodology | Update to PC1 primary metric | 37, 96, 113–136, 144 |
| E1 | Missing | Add validation results section | after §3.3 |
| F1 | Data (V15) | Replace all PC1 values in §3.3 table with V15 mean-pool values | 113–136 |
| F2 | Methodology | Update "K=5 token positions" → "mean pooling over all positions" in §2.4 | 96 |
| F3 | Reconcile | Confirm PANAS-NA scoring instrument consistency between V14 and V15 | 113–136 |
| D1 | Unverified | Anthropic PSM URL | 211 |
| D2 | Unverified | Kumaran DOI | 229 |

A1, B1, B2 are unambiguous — apply immediately.
A2 needs the Duffy citation details.
C1–C4 are a coordinated set — do all or none.
E1 adds the validation results; the data is in the notebook outputs (corrected for the p-value bug).
F1–F3 are V15 updates — apply after confirming PANAS scoring reconciliation (F3).
D1, D2 need external verification.
