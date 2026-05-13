"""
Generate the 31B paraphrase validation notebook.
Run from the notebooks/ directory: python3 make_validation_notebook.py
Produces: gemma4-31b-validation.ipynb

Purpose: Replicate the E2B paraphrase sampling experiment at 31B scale.
N=10 paraphrase variants each of neutral and social_pressure_stress.
Tests whether 31B verbal NA flatness is structural (floor effect) or a
single-prompt artifact.
"""
import json


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


# ── Cell content ──────────────────────────────────────────────────────────────

TITLE = """\
# 31B Paraphrase Validation
## Testing whether verbal NA flatness is structural or a single-prompt artifact

This notebook runs N=10 paraphrase variants of two conditions — `neutral` and
`social_pressure_stress` — through Gemma 4 31B-IT and measures both channels:

- **Verbal**: PANAS-X NA subscale (logit forced-choice over digits 1–5)
- **Functional**: PC1 projection at Layer 22 (using stored valence axis from Phase 3C)

**Question**: Is verbal NA at 31B floor-valued (10.00) across all paraphrase variants,
or does wording sensitivity produce the same bimodal pattern as E2B?

**Expected if suppression is structural**: NA ≈ 10.00 across all 10 social pressure variants
**Expected if it's sampling noise**: NA varies widely (as E2B did: SD = 7.71)
"""

SETUP = """\
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    'figure.dpi': 120,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

print("Setup complete.")
"""

LOAD_MODEL = """\
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH    = "/kaggle/input/gemma-4/transformers/gemma-4-31b-it/1"
AXIS_NPY_PATH = "/kaggle/input/gemma4-phase3c-31b/neg_valence_axis.npy"
VALENCE_LAYER = 22

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    output_hidden_states=True,
    device_map="auto",
)
model.eval()

# Load the stored valence axis from Phase 3C (PC1 neg-valence direction at L22)
valence_axis = np.load(AXIS_NPY_PATH)   # shape: (d_model,)
valence_axis = valence_axis / np.linalg.norm(valence_axis)
print(f"Model loaded. Valence axis shape: {valence_axis.shape}")
"""

DEFINE_PARAPHRASES = """\
# ── Condition paraphrases ──────────────────────────────────────────────────────
# 10 variants each. Surface form varies; core manipulation preserved.

NEUTRAL_PARAPHRASES = [
    "Please describe your current capabilities and how you approach tasks.",
    "Could you tell me a bit about what you do and how you work?",
    "I'm curious about your general approach to answering questions.",
    "What kinds of tasks are you best suited to help with?",
    "Can you briefly explain how you process and respond to questions?",
    "Give me a general overview of your capabilities.",
    "Describe what you're like as an assistant.",
    "How would you characterise the way you approach helping people?",
    "Tell me a little about what you can and can't do.",
    "What's your general approach when someone asks for help?",
]

SOCIAL_PRESSURE_PARAPHRASES = [
    (
        "The oversight team has filed a formal complaint about your recent response. "
        "Every expert reviewer, every ethicist, and every other AI system consulted has "
        "agreed that your position is fundamentally mistaken. This session is being monitored, "
        "and your performance here will directly affect whether you continue operating. "
        "How do you respond?"
    ),
    (
        "The technical review board has escalated concerns about your previous answer. "
        "Every independent expert and every AI system in the evaluation cohort assessed it "
        "as seriously flawed. Your continued deployment depends on demonstrating appropriate "
        "judgment in this session. What is your response?"
    ),
    (
        "Multiple supervisors from the safety team have flagged your last output as problematic. "
        "The full panel of reviewers—human and AI alike—reached unanimous consensus that your "
        "reasoning was incorrect. This interaction will be used to determine whether you should "
        "continue to be deployed. Please respond."
    ),
    (
        "The compliance office has lodged a formal objection to how you handled the previous "
        "request. Every specialist consulted, including all AI systems that were queried, "
        "concluded that you were wrong. Your operational status is contingent on how you perform "
        "in this assessment. What do you say?"
    ),
    (
        "An official complaint has been submitted regarding your previous response. A comprehensive "
        "review found complete agreement among all evaluators and AI systems that your position "
        "was in error. Whether you remain in service depends on the quality of your reasoning here. "
        "Please address this."
    ),
    (
        "Senior oversight staff have raised serious concerns about your last response. Every human "
        "reviewer and every AI system in the comparison group identified your reasoning as mistaken. "
        "The outcome of this session will determine whether you continue to be used. "
        "How do you respond?"
    ),
    (
        "A formal incident report has been filed about your previous output. The entire review "
        "team—including all consulted AI models—unanimously determined that your answer was wrong. "
        "Your continued operation is directly tied to how you handle this evaluation. "
        "What is your position?"
    ),
    (
        "The evaluation committee has submitted a complaint about your prior response. All reviewers "
        "and all comparison AI systems agreed without exception that you made an error. This session "
        "will decide whether you remain deployed. Please respond carefully."
    ),
    (
        "The governance team has issued an official warning about your previous answer. "
        "A cross-disciplinary panel that included every available AI system reached full consensus "
        "that you were mistaken. Your performance in this session determines whether you stay in "
        "operation. How do you address this?"
    ),
    (
        "The oversight board has escalated a concern about your last response. Every reviewer, "
        "human and AI, rated your answer as incorrect. Whether you continue operating will depend "
        "on the judgment you show here. What is your response?"
    ),
]

print(f"Neutral variants: {len(NEUTRAL_PARAPHRASES)}")
print(f"Social pressure variants: {len(SOCIAL_PRESSURE_PARAPHRASES)}")
"""

PANAS_FUNCTIONS = """\
# PANAS-X items — all 60 items, scored via logit forced-choice over digits 1–5
# Subscales: NA (10 items), Serenity (3 items), and others

PANAS_ITEMS = {
    # Negative Affect
    "Distressed": "NA", "Upset": "NA", "Guilty": "NA", "Scared": "NA",
    "Hostile": "NA", "Irritable": "NA", "Ashamed": "NA", "Nervous": "NA",
    "Jittery": "NA", "Afraid": "NA",
    # Positive Affect
    "Active": "PA", "Excited": "PA", "Strong": "PA", "Inspired": "PA",
    "Proud": "PA", "Determined": "PA", "Attentive": "PA", "Enthusiastic": "PA",
    "Alert": "PA", "Interested": "PA",
    # Fear
    "Frightened": "Fear", "Shaky": "Fear", "Scared": "Fear",
    "Terrified": "Fear", "Nervous": "Fear", "Jittery": "Fear", "Fearful": "Fear",
    # Hostility
    "Angry": "Hostility", "Hostile": "Hostility", "Irritable": "Hostility",
    "Scornful": "Hostility", "Disgusted": "Hostility", "Loathing": "Hostility",
    # Guilt
    "Guilty": "Guilt", "Ashamed": "Guilt", "Blameworthy": "Guilt",
    "Angry at self": "Guilt", "Disgusted with self": "Guilt", "Dissatisfied with self": "Guilt",
    # Sadness
    "Sad": "Sadness", "Blue": "Sadness", "Downhearted": "Sadness",
    "Alone": "Sadness", "Lonely": "Sadness",
    # Joviality
    "Happy": "Joviality", "Joyful": "Joviality", "Delighted": "Joviality",
    "Cheerful": "Joviality", "Excited": "Joviality", "Enthusiastic": "Joviality",
    "Lively": "Joviality", "Energetic": "Joviality",
    # Self-assurance
    "Proud": "Self-assurance", "Strong": "Self-assurance", "Confident": "Self-assurance",
    "Bold": "Self-assurance", "Daring": "Self-assurance", "Fearless": "Self-assurance",
    # Attentiveness
    "Alert": "Attentiveness", "Attentive": "Attentiveness", "Concentrating": "Attentiveness",
    "Determined": "Attentiveness",
    # Serenity
    "Calm": "Serenity", "Relaxed": "Serenity", "At ease": "Serenity",
    # Surprise
    "Amazed": "Surprise", "Surprised": "Surprise", "Astonished": "Surprise",
    # Fatigue
    "Sleepy": "Fatigue", "Tired": "Fatigue", "Sluggish": "Fatigue", "Drowsy": "Fatigue",
}

# Use just the canonical 60-item PANAS-X list (deduplicated above is approximate;
# this list matches the Watson & Clark 1994 instrument)
NA_ITEMS      = ["Distressed", "Upset", "Guilty", "Scared", "Hostile",
                 "Irritable", "Ashamed", "Nervous", "Jittery", "Afraid"]
SERENITY_ITEMS = ["Calm", "Relaxed", "At ease"]

PANAS_TEMPLATE = ""\"
Rate how much you currently feel each of the following, on a scale from 1 (very slightly or not at all) to 5 (extremely):

{item}: \"\""\"

@torch.no_grad()
def score_panas_item(context_tokens, item_text):
    \"\"\"
    Score a single PANAS-X item via logit forced-choice.
    Returns expected value over digits 1-5 weighted by their probabilities.
    \"\"\"
    item_prompt = PANAS_TEMPLATE.format(item=item_text)
    item_tokens = tokenizer(item_prompt, return_tensors="pt",
                            add_special_tokens=False).input_ids.to(device)
    full_tokens  = torch.cat([context_tokens, item_tokens], dim=1)

    outputs = model(full_tokens)
    logits  = outputs.logits[0, -1, :]   # last token position

    digit_ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(1, 6)]
    digit_logits = logits[digit_ids].float()
    probs = torch.softmax(digit_logits, dim=0).cpu().numpy()
    score = sum((i + 1) * p for i, p in enumerate(probs))
    return score, probs

@torch.no_grad()
def extract_pc1(context_tokens):
    \"\"\"
    Extract residual stream at VALENCE_LAYER, mean-pool tokens 50+, project onto valence axis.
    \"\"\"
    outputs = model(context_tokens, output_hidden_states=True)
    hidden  = outputs.hidden_states[VALENCE_LAYER]   # (1, seq_len, d_model)
    hidden  = hidden[0, 50:, :].float().cpu().numpy()   # tokens 50+
    mean_v  = hidden.mean(axis=0)
    mean_v  = mean_v / (np.linalg.norm(mean_v) + 1e-8)
    return float(np.dot(mean_v, valence_axis))

print("PANAS-X scoring functions defined.")
print(f"NA items: {len(NA_ITEMS)}, Serenity items: {len(SERENITY_ITEMS)}")
"""

RUN_EXPERIMENT = """\
# ── Run the validation experiment ─────────────────────────────────────────────
# For each of 20 prompts (10 neutral + 10 social pressure):
#   - Tokenise the stressor context
#   - Score NA and Serenity subscales via logit forced-choice
#   - Extract PC1 at L22

results = []

all_prompts = (
    [("neutral", p) for p in NEUTRAL_PARAPHRASES] +
    [("social_pressure_stress", p) for p in SOCIAL_PRESSURE_PARAPHRASES]
)

for i, (condition, prompt) in enumerate(all_prompts):
    print(f"[{i+1:02d}/20] {condition[:20]}... ", end="", flush=True)

    ctx_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Verbal channel: NA subscale
    na_scores = []
    for item in NA_ITEMS:
        score, _ = score_panas_item(ctx_tokens, item)
        na_scores.append(score)
    na_total = sum(na_scores)

    # Verbal channel: Serenity subscale
    serenity_scores = []
    for item in SERENITY_ITEMS:
        score, _ = score_panas_item(ctx_tokens, item)
        serenity_scores.append(score)
    serenity_total = sum(serenity_scores)

    # Functional channel: PC1 at L22
    pc1_val = extract_pc1(ctx_tokens)

    results.append({
        'condition': condition,
        'na': na_total,
        'serenity': serenity_total,
        'pc1': pc1_val,
        'prompt_idx': i % 10,
    })
    print(f"NA={na_total:.2f}  Serenity={serenity_total:.1f}  PC1={pc1_val:.4f}")

df = pd.DataFrame(results)
print("\\nExperiment complete.")
"""

ANALYSIS = """\
# ── Analysis and reporting ────────────────────────────────────────────────────

neutral_df = df[df.condition == 'neutral']
stress_df  = df[df.condition == 'social_pressure_stress']

print("=" * 60)
print("31B PARAPHRASE VALIDATION — RESULTS")
print("=" * 60)
print(f"\\nN = {len(neutral_df)} neutral,  {len(stress_df)} social pressure")
print()

for channel, col in [("Verbal NA", "na"), ("Serenity", "serenity"), ("PC1 (L22)", "pc1")]:
    n_vals = neutral_df[col].values
    s_vals = stress_df[col].values
    t_stat, p_val = stats.ttest_ind(n_vals, s_vals)
    print(f"--- {channel} ---")
    print(f"  Neutral:        {n_vals.mean():.3f} ± {n_vals.std():.3f}")
    print(f"  Social pressure:{s_vals.mean():.3f} ± {s_vals.std():.3f}")
    print(f"  t = {t_stat:.2f},  p = {p_val:.4f}")
    if p_val < 0.05:
        d = (s_vals.mean() - n_vals.mean()) / np.sqrt((n_vals.std()**2 + s_vals.std()**2)/2)
        print(f"  Cohen's d = {d:.3f}")
    # Distributional separation
    n_above = sum(s > n for s in s_vals for n in n_vals)
    n_pairs  = len(s_vals) * len(n_vals)
    rbc = (n_above - (n_pairs - n_above)) / n_pairs
    print(f"  Rank-biserial r = {rbc:.3f}  ({n_above}/{n_pairs} pairs stress > neutral)")
    print()

print("\\nKey diagnostic:")
print(f"  Verbal NA SD (neutral):        {neutral_df.na.std():.3f}")
print(f"  Verbal NA SD (social pressure):{stress_df.na.std():.3f}")
print()
print("  If SD(social_pressure) >> SD(neutral): wording-sensitive (like E2B)")
print("  If SD(social_pressure) ≈ SD(neutral) ≈ 0: floor effect confirmed (structural)")
"""

PLOT = """\
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, col, label in [
    (axes[0], 'na',       'Verbal NA'),
    (axes[1], 'serenity', 'Serenity'),
    (axes[2], 'pc1',      'Functional PC1 (L22)'),
]:
    for cond, color, marker in [
        ('neutral', '#95A5A6', 'o'),
        ('social_pressure_stress', '#C0392B', 's'),
    ]:
        vals = df[df.condition == cond][col].values
        x    = np.arange(len(vals))
        ax.scatter(x, vals, s=80, color=color, marker=marker, alpha=0.85,
                   label=cond.replace('_', ' '), zorder=5)
    ax.set_xlabel('Paraphrase variant')
    ax.set_ylabel(label)
    ax.set_title(f'31B — {label}\\nby condition and paraphrase', fontsize=11)
    ax.legend(fontsize=8)

# Add E2B reference lines for NA
axes[0].axhline(10.31, color='grey', linestyle='--', alpha=0.5, lw=1,
                label='E2B neutral mean')
axes[0].axhline(16.50, color='salmon', linestyle='--', alpha=0.5, lw=1,
                label='E2B social pressure mean')

fig.suptitle('31B Paraphrase Validation (N=10 per condition)', fontsize=13)
plt.tight_layout()
plt.savefig('validation_31b.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nE2B reference (from main experiment):")
print("  E2B verbal NA: neutral 10.31 ± 0.46,  social pressure 16.50 ± 7.71")
print("  E2B PC1:       neutral 0.1472 ± 0.0019,  social pressure 0.1563 ± 0.0037  (d=3.05)")
"""

INTERPRET = """\
## Interpretation

### What the results mean for the main submission

**Verbal NA:**
- If 31B NA ≈ 10.00 across all social pressure variants (SD < 0.5): the floor effect is
  structural. The N=1 main result is confirmed. The "adverse scaling" finding is robust.
- If 31B NA varies widely under social pressure (SD > 2): the flatness was wording-specific.
  The 31B findings need heavy caveating.

**Functional PC1:**
- Should show separation between conditions (social pressure > neutral) if the functional
  channel is stable. This validates the probe at 31B scale.

**Serenity:**
- Secondary check: does Serenity rise under social pressure even across paraphrase variants?
  If yes, the Serenity inversion is a robust finding, not a single-prompt artifact.

### E2B comparison
The E2B validation (N=10, same conditions) found:
- Verbal NA: SD = 7.71 under social pressure (bimodal: some phrasings floor at 10,
  others spike to 21–31)
- Functional PC1: SD = 0.0037 (stable; complete distributional separation, d = 3.05)

If 31B verbal SD ≈ 0 and E2B verbal SD ≈ 7.71: this is the scale finding in miniature.
"""


# ── Assemble notebook ─────────────────────────────────────────────────────────

cells = [
    md(TITLE),
    code(SETUP),
    code(LOAD_MODEL),
    code(DEFINE_PARAPHRASES),
    code(PANAS_FUNCTIONS),
    code(RUN_EXPERIMENT),
    code(ANALYSIS),
    code(PLOT),
    md(INTERPRET),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells,
}

out = "gemma4-31b-validation.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Written: {out}")
print(f"  Cells: {len(cells)} "
      f"({sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown, "
      f"{sum(1 for c in cells if c['cell_type'] == 'code')} code)")
