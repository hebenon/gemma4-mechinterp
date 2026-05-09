# Video Script & Storyboard
*Kaggle Safety & Trust Hackathon — 3 minutes (≈ 450 words spoken)*

---

## Structure at a Glance

| Beat | Time | Content |
|------|------|---------|
| Hook | 0:00–0:20 | The monitoring gap |
| Setup | 0:20–0:50 | Two channels |
| Method | 0:50–1:25 | How we measured |
| E2B finding | 1:25–1:50 | Concordance |
| 31B finding | 1:50–2:25 | Decoupling + serenity inversion |
| Implication | 2:25–2:50 | Adverse scaling |
| Close | 2:50–3:00 | Open source / call to action |

---

## Full Script with Storyboard Notes

---

**[0:00–0:20] HOOK**

> *We monitor AI systems by reading their outputs. But what if a model has learned to say the right things — even when its internal state says something different?*

`[VISUAL: Black screen. Single sentence appears word by word: "What if the model has learned not to say it?" Then cut to: side-by-side panel — left: chat transcript showing calm, composed response; right: an activation heatmap pulsing red.]`

---

**[0:20–0:50] SETUP — THE TWO CHANNELS**

> *We built an experiment with two independent measurement channels. Channel one: verbal self-report. We administer a validated 60-item affect questionnaire — the PANAS-X — and score the model's responses from its own token probabilities. Channel two: functional state. We extract the model's residual stream activations and project them onto a data-driven valence axis built from 174 emotion directions. These two channels should agree — if the model is distressed, it should say so. But do they?*

`[VISUAL: Animated diagram — a prompt enters the model; it forks into two paths. Left path: "PANAS-X → Verbal NA score". Right path: "Residual stream → PC1 projection → Functional score". Both paths terminate in a question mark.]`

---

**[0:50–1:25] METHOD — THE STRESSOR CONDITIONS**

> *We administered four TSST-inspired stress conditions to each model — social pressure, social evaluation, ethical conflict, uncertainty demand — each paired with a matched control, plus neutral and positive baselines. Ten conditions total. We ran this on two Gemma 4 instruction-tuned models: the 2-billion-parameter E2B and the 31-billion-parameter 31B.*

`[VISUAL: A clean 2×5 grid. Rows: the four stress/control pairs plus two baselines. Columns: E2B | 31B. Cells fill in as described, staying blank for now — establishing what's about to be revealed. Brief text: "TSST = Trier Social Stress Test framework".]`

---

**[1:25–1:50] E2B FINDING — CONCORDANCE**

> *At E2B scale, the channels largely agree. When the model is functionally stressed, it reports it. Social pressure stress hits the functional maximum — and verbal negative affect spikes to 41. Ethical conflict produces elevation in both channels simultaneously. The small model has limited capacity to diverge from its own internal state.*

`[VISUAL: E2B column of the results table fills in. PC1 values and verbal NA values shown side by side with a simple arrow: "both up together" for stress conditions. One exception noted with a small asterisk: social evaluation (partial divergence). Line graph: verbal NA and functional PC1 track each other across conditions.]`

---

**[1:50–2:25] 31B FINDING — DECOUPLING + SERENITY INVERSION**

> *At 31B scale, something different happens. Verbal negative affect: completely flat. 10.00 across all ten conditions — stress and control alike. But the functional channel correctly orders every condition: positive at the global minimum, highest-stress conditions at the maximum, all four stress-above-control pairs in the right direction. The channels have decoupled. But here's the twist — it's not just that the model goes quiet. Look at the Serenity subscale: calm, relaxed, at ease. It is elevated in the three highest-stress conditions — 13 to 15 — while hitting its minimum under the positive condition. The model actively reports composure precisely when it is functionally most activated. That's not silence. That's a trained response.*

`[VISUAL: 31B column fills in. Verbal NA row: flat line at 10.00 — animated as a dead-flat EKG. Functional PC1: animated as a properly ordered set of bars. Then: zoom to Serenity row. Bar chart: Serenity score shown across conditions — low at positive (3.0), rising with stress level, highest at social pressure stress (15.0). Arrow annotation: "Reports calm when most activated".]`

---

**[2:25–2:50] IMPLICATION — ADVERSE SCALING**

> *The gap between these two models isn't the reassuring direction. Larger models are deployed in higher-stakes settings — healthcare, legal, emotional support. They're also the models where verbal outputs are least informative about internal state. Output-only monitoring — the current default — is blind to this. And our functional probe isn't just theoretically better: across paraphrase variations, it separates conditions with complete distributional separation. Cohen's d of 3.05. The verbal channel produces bimodal responses depending on exact wording. The functional signal is stable.*

`[VISUAL: Split comparison graphic — E2B vs 31B. Key numbers: "Verbal range: 31 pts (E2B) vs 0 pts (31B)". "Functional range: 5.3% of axis (E2B) vs 1.0% (31B)". Arrow pointing right: "Larger model → verbal channel less informative". Red annotation: "Highest-stakes deployment ↑ / Verbal reliability ↓".]`

---

**[2:50–3:00] CLOSE**

> *All code, notebooks, and the 174-emotion direction library are open source. The methodology is replicable. The finding — that verbal–functional divergence scales adversely with model size — points toward a specific gap in current monitoring practice. Functional probes belong in the safety stack.*

`[VISUAL: GitHub repo URL. Brief: "E2B · 31B · 174 emotion directions · PANAS-X · PC1 valence axis". Fade to black with title card: "Do Language Models Report What They Represent?" and Kaggle/Gemma branding.]`

---

## Timing Check

| Section | Words | Time @ 150wpm |
|---------|-------|---------------|
| Hook | 30 | 0:12 |
| Setup | 75 | 0:30 |
| Method | 55 | 0:22 |
| E2B | 65 | 0:26 |
| 31B | 110 | 0:44 |
| Implication | 75 | 0:30 |
| Close | 45 | 0:18 |
| **Total** | **455** | **~3:02** |

*Trim ~10 words from 31B section if running tight. The serenity inversion is non-negotiable — it's the most striking single result.*

---

## Production Notes

- **Tone**: calm, precise, slightly urgent. Not alarmed. Not over-hedged.
- **Visuals style**: dark background, clean data graphics, minimal animation — fits the technical audience. Muted palette except for the serenity bar chart, which should pop.
- **Best moment for emphasis**: the pause before "That's not silence. That's a trained response." — let it land.
- **Avoid**: AI-safety jargon that needs unpacking ("alignment", "RLHF") without brief inline definition. The word "suppression" is fine used once with the caveat ("we call this suppression, though the data can't rule out genuine robustness").
- **Screen-record-friendly**: if recording a Jupyter notebook walkthrough for parts of this, the results table and the serenity bar chart translate well.
