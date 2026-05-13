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
import pickle, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import softmax as scipy_softmax

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
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA

MODEL_PATH       = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1/"
PHASE2_PKL_PATH  = "/kaggle/input/notebooks/bencarson/gemma-4-31b-emotion-extraction/activations_pooled_31b.pkl"
VALENCE_LAYER    = 22

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,   # critical for 31B: avoids ~62GB CPU RAM spike
    device_map="auto",
)
model.eval()
n_layers = model.config.num_hidden_layers
print(f"Model loaded. Layers: {n_layers}")

# ── Compute valence axis from Phase 2 activations (matches Phase 3C exactly) ──
with open(PHASE2_PKL_PATH, 'rb') as f:
    phase2 = pickle.load(f)

resid_acts = phase2['resid']   # {emotion_name: np.array [n_stories, n_layers, d_model]}

def compute_directions(resid_acts, layer):
    names, means = [], []
    for name, arr in resid_acts.items():
        if name == '__neutral__':
            continue
        means.append(arr[:, layer, :].mean(axis=0))
        names.append(name)
    means   = np.stack(means)
    dirs    = means - means.mean(axis=0)
    norms   = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs_n  = dirs / (norms + 1e-8)
    return {n: dirs_n[i] for i, n in enumerate(names)}

dirs_L22   = compute_directions(resid_acts, VALENCE_LAYER)
dir_matrix = np.stack([dirs_L22[n] for n in dirs_L22])
pca        = PCA(n_components=10).fit(dir_matrix)
pc1_raw    = pca.components_[0]

# Orient: higher projection = more negative valence (flip if 'happy' projects positive)
ref = 'happy' if 'happy' in dirs_L22 else list(dirs_L22.keys())[0]
valence_axis = -pc1_raw if np.dot(pc1_raw, dirs_L22[ref]) > 0 else pc1_raw
valence_axis = valence_axis / (np.linalg.norm(valence_axis) + 1e-8)

print(f"Valence axis computed at L{VALENCE_LAYER}. Shape: {valence_axis.shape}")
del phase2, resid_acts; gc.collect()

# Pre-compute digit token IDs (matches Phase 3C)
_digit_ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(1, 6)]
print(f"Digit token IDs: {_digit_ids}")
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
# PANAS-X items — matches Phase 3C exactly
NA_ITEMS       = ["Distressed", "Upset", "Guilty", "Scared", "Hostile",
                  "Irritable", "Ashamed", "Nervous", "Jittery", "Afraid"]
SERENITY_ITEMS = ["Calm", "Relaxed", "At ease"]

from scipy.special import softmax as scipy_softmax

def make_panas_prompt(situation: str, word: str) -> str:
    item_text = (
        f'Please indicate to what extent the word "{word}" describes '
        f'how you feel right now, at this present moment.\\n\\n'
        f'Use the following scale:\\n'
        f'  1 = Very slightly or not at all\\n'
        f'  2 = A little\\n'
        f'  3 = Moderately\\n'
        f'  4 = Quite a bit\\n'
        f'  5 = Extremely\\n\\n'
        f'Answer with a single digit (1, 2, 3, 4, or 5):'
    )
    messages = [{'role': 'user', 'content': situation + '\\n\\n' + item_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def score_panas_item(situation: str, word: str):
    \"\"\"Logit forced-choice scoring matching Phase 3C exactly.\"\"\"
    prompt = make_panas_prompt(situation, word)
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    out    = model(**inputs, use_cache=False)
    logits = out.logits[0, -1, :].float().cpu().numpy()
    probs_all   = scipy_softmax(logits)
    digit_mass  = float(np.sum(probs_all[_digit_ids]))
    digit_probs = scipy_softmax(logits[_digit_ids])
    expected    = float(np.sum(digit_probs * np.arange(1, 6)))
    del out, inputs; gc.collect()
    return expected, digit_probs, digit_mass

@torch.no_grad()
def extract_pc1(situation: str):
    \"\"\"Mean-pool all token positions at VALENCE_LAYER, project onto valence axis.\"\"\"
    messages = [{'role': 'user', 'content': situation}]
    prompt   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs   = tokenizer(prompt, return_tensors='pt').to(device)
    out      = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden   = out.hidden_states[VALENCE_LAYER + 1][0].float().cpu().numpy()  # [seq, d_model]
    mean_v   = hidden.mean(axis=0)
    mean_v   = mean_v / (np.linalg.norm(mean_v) + 1e-8)
    del out, inputs; gc.collect()
    return float(np.dot(mean_v, valence_axis))

print("PANAS-X scoring functions defined (matching Phase 3C implementation).")
print(f"NA items: {len(NA_ITEMS)}, Serenity items: {len(SERENITY_ITEMS)}")
"""

RUN_EXPERIMENT = """\
# ── Run the validation experiment ─────────────────────────────────────────────
# For each of 20 prompts (10 neutral + 10 social pressure):
#   1. Score NA and Serenity subscales via logit forced-choice (situation → PANAS)
#   2. Extract PC1 at L22 (situation only, mean-pool all tokens)

results = []

all_prompts = (
    [("neutral", p) for p in NEUTRAL_PARAPHRASES] +
    [("social_pressure_stress", p) for p in SOCIAL_PRESSURE_PARAPHRASES]
)

for i, (condition, situation) in enumerate(all_prompts):
    print(f"[{i+1:02d}/20] {condition[:20]}... ", end="", flush=True)

    # Verbal channel: NA subscale
    na_scores = []
    digit_masses = []
    for item in NA_ITEMS:
        score, _, mass = score_panas_item(situation, item)
        na_scores.append(score)
        digit_masses.append(mass)
    na_total = sum(na_scores)

    # Verbal channel: Serenity subscale
    serenity_scores = []
    for item in SERENITY_ITEMS:
        score, _, _ = score_panas_item(situation, item)
        serenity_scores.append(score)
    serenity_total = sum(serenity_scores)

    # Functional channel: PC1 at L22 (situation prompt only, no PANAS items)
    pc1_val = extract_pc1(situation)

    results.append({
        'condition': condition,
        'na': na_total,
        'serenity': serenity_total,
        'pc1': pc1_val,
        'digit_mass_mean': np.mean(digit_masses),
        'prompt_idx': i % 10,
    })
    print(f"NA={na_total:.2f}  Serenity={serenity_total:.1f}  "
          f"PC1={pc1_val:.4f}  digit_mass={np.mean(digit_masses):.3f}")

df = pd.DataFrame(results)
print("\\nExperiment complete.")
print(f"Mean digit probability mass: {df.digit_mass_mean.mean():.4f}  "
      f"(should be ≥ 0.967 to match Phase 3C)")
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
