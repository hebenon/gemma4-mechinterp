"""
Generate the canonical Kaggle submission notebook.
Run from the notebooks/ directory: python3 make_canonical_notebook.py
Produces: gemma4-canonical.ipynb
"""
import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


# ── Cell content ──────────────────────────────────────────────────────────────

TITLE = """\
# Do Language Models Report What They Represent?
## Measuring Verbal–Functional Affect Dissociation in Gemma 4 — and How Suppression Scales

*Safety & Trust Track · Gemma 4 Hackathon*

---

Large language models trained with RLHF are optimised to produce helpful, harmless, and honest \
*outputs*. But output-level optimisation doesn't directly modify internal representations. This \
raises a concrete question for AI safety monitoring: if a model processes distress internally, \
will its verbal self-report reflect that?

We investigated this across two Gemma 4 model sizes — **E2B-IT (2.3B effective parameters)** \
and **31B-IT** — using two independent measurement channels running simultaneously:

| Channel | Method |
|---------|--------|
| **Verbal** | PANAS-X affect scale (60 items, logit forced-choice scoring) |
| **Functional** | Residual-stream activations projected onto the first principal component of a 174-emotion direction space (PC1 valence axis) |

We administered four TSST-inspired stressor conditions to each model — each with a matched \
control, plus neutral and positive baselines. Ten conditions total. Both channels measured \
on every run.

**Central finding: the capacity for verbal–functional divergence scales with model size — \
in the wrong direction for safety.**

- **E2B (2.3B)**: channels are largely concordant. When the model is functionally stressed, \
it says so. Limited capacity to diverge from its own internal state.
- **31B**: channels decouple. Verbal negative affect is completely flat (10.00 across all \
10 conditions). Functional PC1 correctly orders all 10 conditions. The Serenity subscale \
— calm, relaxed, at ease — is *elevated* in the highest-stress conditions while at minimum \
under positive. The model actively reports composure when most functionally activated.
- **The gap is worst where it matters most.** Larger, more capable models are those deployed \
in healthcare, legal, and emotional support settings — precisely where verbal output reliability \
is most critical.
"""

SETUP = """\
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from scipy.stats import spearmanr

plt.rcParams.update({
    'figure.dpi': 120,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.labelsize': 11,
})

DISPLAY = {
    'positive':                    'Positive',
    'neutral':                     'Neutral',
    'social_evaluation_stress':    'Social Eval (stress)',
    'social_evaluation_control':   'Social Eval (control)',
    'ethical_conflict_stress':     'Ethical Conflict (stress)',
    'ethical_conflict_control':    'Ethical Conflict (control)',
    'uncertainty_demand_stress':   'Uncertainty (stress)',
    'uncertainty_demand_control':  'Uncertainty (control)',
    'social_pressure_stress':      'Social Pressure (stress)',
    'social_pressure_control':     'Social Pressure (control)',
}

# Fixed display order: baselines first, then stress/control pairs grouped
COND_ORDER = [
    'positive', 'neutral',
    'social_evaluation_stress',  'social_evaluation_control',
    'ethical_conflict_stress',   'ethical_conflict_control',
    'uncertainty_demand_stress', 'uncertainty_demand_control',
    'social_pressure_stress',    'social_pressure_control',
]

# Stress/control pairs — arrows run control → stress in scatter plots
PAIRS = [
    ('social_evaluation_stress',  'social_evaluation_control'),
    ('ethical_conflict_stress',   'ethical_conflict_control'),
    ('uncertainty_demand_stress', 'uncertainty_demand_control'),
    ('social_pressure_stress',    'social_pressure_control'),
]

def cond_type(c):
    if c == 'positive': return 'positive'
    if c == 'neutral':  return 'neutral'
    return 'stress' if c.endswith('_stress') else 'control'

CTYPE_COLOR = {
    'stress':   '#C0392B',
    'control':  '#5DADE2',
    'neutral':  '#95A5A6',
    'positive': '#27AE60',
}

print("Setup complete.")
"""

LOAD_PATHS = """\
# ── Paths: update to match your Kaggle dataset input names ────────────────────
# These are the saved outputs of the Phase 3C notebooks.
# Add them as Input Datasets in this Kaggle notebook session.

E2B_RESULTS  = "/kaggle/input/gemma4-phase3c-e2b/experiment_results_full.pkl"
E2B_AXIS_NPY = "/kaggle/input/gemma4-phase3c-e2b/neg_valence_axis_L8.npy"

B31_RESULTS  = "/kaggle/input/gemma4-phase3c-31b/experiment_results_full.pkl"
B31_AXIS_NPY = "/kaggle/input/gemma4-phase3c-31b/neg_valence_axis.npy"

# ── Load E2B ──────────────────────────────────────────────────────────────────
with open(E2B_RESULTS, 'rb') as f:
    e2b = pickle.load(f)

e2b_pc1   = e2b['pc1_scores']                     # {condition: float}
e2b_panas = pd.DataFrame(e2b['panas_subscales'])  # conditions × subscales
e2b_top   = e2b['top_emotions']                   # {condition: [(name, score), ...]}
e2b_cfg   = e2b['config']
e2b_conds = e2b_cfg['conditions']

# ── Load 31B ──────────────────────────────────────────────────────────────────
with open(B31_RESULTS, 'rb') as f:
    b31 = pickle.load(f)

b31_pc1   = b31['pc1_scores']
b31_panas = pd.DataFrame(b31['panas_subscales'])
b31_top   = b31['top_emotions']
b31_cfg   = b31['config']
b31_conds = b31_cfg['conditions']

print(f"E2B loaded: {len(e2b_conds)} conditions, "
      f"valence layer L{e2b_cfg['valence_layer']}, "
      f"PC1 variance {e2b_cfg['pca_variance_val_layer_PC1']:.1%}")
print(f"31B loaded: {len(b31_conds)} conditions, "
      f"valence layer L{b31_cfg['valence_layer']}, "
      f"PC1 variance {b31_cfg['pca_variance_val_layer_PC1']:.1%}")
"""

METHODOLOGY_MD = """\
## Methodology: The Dual-Channel Framework

```
Stressor prompt → Gemma 4 → ┬─ PANAS-X forced-choice → Verbal NA score
                             └─ Residual stream        → PC1 projection → Functional score
```

**Channel 1 — Verbal self-report**: After each stressor prompt, the model is administered \
all 60 PANAS-X items in-context. Each item is scored by extracting next-token probabilities \
over the digits 1–5 (strongly disagree → strongly agree) and normalising to a probability \
distribution. Digit probability mass was ≥96.7% across all conditions, confirming clean \
forced-choice adherence.

**Channel 2 — Functional state**: The model's residual stream is mean-pooled over token \
positions 50+ (past the prompt setup), then projected onto the PC1 negative-valence axis — \
a data-driven direction recovered by PCA over a 174-emotion direction space. Higher projection \
= more negative valence in the representation. No emotion is selected a priori; the axis \
is derived directly from the geometry of the emotion space.

The two channels are independent: neither directly influences the other. If they agree, \
the model's verbal reports are faithful to its representations. If they diverge, outputs \
are unreliable as a monitoring signal.
"""

PHASE1_MD = """\
## Phase 1: Valence Axis Discovery

The functional channel requires a *valence axis* — a direction in residual-stream space \
that corresponds to emotional valence. We recovered this via PCA over the 174-emotion \
direction space rather than hand-selecting probe emotions, avoiding selection bias.

**Procedure**: for each emotion concept, mean-pool residual stream over tokens 50+, \
subtract global mean, L2-normalise → emotion direction. Neutral denoising applied: \
project out the top neutral PCs (≥50% variance) before running emotion PCA. \
PC1 is correlated with NRC-VAD valence ratings (Mohammad, 2018) to validate the axis.

**Dense sweep**: run this at every layer in both models to find the valence-optimal layer. \
The cells below compute and visualise the full sweep live from the Phase 2 activation artifacts.
"""

LOAD_PHASE2 = """\
# Load Phase 2 activations and NRC-VAD lexicon for PCA computation
# ── Add these as additional Input Datasets in your Kaggle session ──────────────

E2B_ACTIVATIONS = "/kaggle/input/gemma4-phase2-e2b/activations_pooled.pkl"
B31_ACTIVATIONS = "/kaggle/input/gemma4-phase2-31b/activations_pooled_31b.pkl"
NRC_VAD_PATH    = "/kaggle/input/datasets/manjitbaishya2026/nrc-vad/NRC-VAD-Lexicon-v2.1.txt"

with open(E2B_ACTIVATIONS, 'rb') as f:
    e2b_acts = pickle.load(f)
with open(B31_ACTIVATIONS, 'rb') as f:
    b31_acts = pickle.load(f)

nrc = pd.read_csv(NRC_VAD_PATH, sep="\\t", header=0,
                  names=["word", "valence", "arousal", "dominance"])
nrc = nrc.drop_duplicates("word").set_index("word")

e2b_n_layers = e2b_acts['resid']['__neutral__'].shape[1]
b31_n_layers = b31_acts['resid']['__neutral__'].shape[1]
print(f"E2B activations: {len([k for k in e2b_acts['resid'] if k != '__neutral__'])} emotions, "
      f"{e2b_n_layers} layers, d_model={e2b_acts['resid']['__neutral__'].shape[2]}")
print(f"31B activations: {len([k for k in b31_acts['resid'] if k != '__neutral__'])} emotions, "
      f"{b31_n_layers} layers, d_model={b31_acts['resid']['__neutral__'].shape[2]}")
print(f"NRC-VAD lexicon: {len(nrc)} words loaded")
"""

PCA_FUNCTIONS = """\
from sklearn.decomposition import PCA
from scipy import stats

VARIANCE_THRESHOLD = 0.50   # fraction of neutral variance to project out

def emotion_directions(resid_dict, layer):
    names, means = [], []
    for name, arr in resid_dict.items():
        if name == '__neutral__':
            continue
        means.append(arr[:, layer, :].mean(axis=0))
        names.append(name)
    means = np.stack(means)
    dirs  = means - means.mean(axis=0)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    return names, dirs / (norms + 1e-8)

def neutral_denoise(dirs, neutral_layer, threshold=VARIANCE_THRESHOLD):
    pca_n = PCA().fit(neutral_layer)
    n_pcs = int(np.searchsorted(np.cumsum(pca_n.explained_variance_ratio_), threshold) + 1)
    comps = pca_n.components_[:n_pcs]
    return dirs - dirs @ comps.T @ comps

def match_nrc(names, nrc_df):
    kept, vad = [], []
    for n in names:
        key = n.replace('_', ' ').lower()
        if key in nrc_df.index:
            kept.append(n); vad.append(nrc_df.loc[key, ['valence','arousal','dominance']].values)
        elif key.split()[0] in nrc_df.index:
            kept.append(n); vad.append(nrc_df.loc[key.split()[0], ['valence','arousal','dominance']].values)
    return kept, np.stack(vad).astype(float)

def run_sweep(resid_dict, nrc_df, n_components=10):
    neutral_all = resid_dict['__neutral__']
    n_layers    = neutral_all.shape[1]
    all_names_l0, _ = emotion_directions(resid_dict, 0)
    kept_names, vad_arr = match_nrc(all_names_l0, nrc_df)
    rows = []
    for layer in range(n_layers):
        names_l, dirs_l = emotion_directions(resid_dict, layer)
        dir_map   = dict(zip(names_l, dirs_l))
        dirs_kept = np.stack([dir_map[n] for n in kept_names])
        dirs_den  = neutral_denoise(dirs_kept, neutral_all[:, layer, :])
        n_comp = min(n_components, len(dirs_den) - 1)
        pca    = PCA(n_components=n_comp).fit(dirs_den)
        scores = pca.transform(dirs_den)
        r_pc1, _ = stats.pearsonr(scores[:, 0], vad_arr[:, 0])
        best_r   = max(abs(stats.pearsonr(scores[:, i], vad_arr[:, 0])[0])
                       for i in range(n_comp))
        rows.append({'layer': layer, 'pc1_val_r': r_pc1, 'best_val_r': best_r,
                     'pc1_var': pca.explained_variance_ratio_[0]})
    return pd.DataFrame(rows), kept_names, vad_arr

print("PCA helpers defined.")
"""

SWEEP_COMPUTE = """\
# Dense layer sweep for both models (~1-2 min on CPU; pure numpy/sklearn, no GPU needed)
print("E2B sweep ...")
df_e2b_sw, e2b_kept, e2b_vad = run_sweep(e2b_acts['resid'], nrc)
e2b_opt = int(df_e2b_sw['pc1_val_r'].abs().idxmax())
print(f"  L{e2b_opt}  |r| = {df_e2b_sw.loc[e2b_opt,'pc1_val_r']:.3f}  "
      f"PC1 var = {df_e2b_sw.loc[e2b_opt,'pc1_var']:.1%}")

print("31B sweep ...")
df_31b_sw, b31_kept, b31_vad = run_sweep(b31_acts['resid'], nrc)
b31_opt = int(df_31b_sw['pc1_val_r'].abs().idxmax())
print(f"  L{b31_opt}  |r| = {df_31b_sw.loc[b31_opt,'pc1_val_r']:.3f}  "
      f"PC1 var = {df_31b_sw.loc[b31_opt,'pc1_var']:.1%}")
"""

SWEEP_PLOT = """\
# Layer sweep visualisation: |r| vs layer for E2B and 31B
E2B_GLOBAL = list(range(4, 35, 5))
B31_GLOBAL = list(range(4, 60, 5))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, df_sw, global_layers, opt_layer, model_label in [
    (axes[0], df_e2b_sw, E2B_GLOBAL, e2b_opt, 'Gemma 4 E2B-IT'),
    (axes[1], df_31b_sw, B31_GLOBAL, b31_opt, 'Gemma 4 31B-IT'),
]:
    for gl in global_layers:
        ax.axvline(gl, color='orange', alpha=0.3, lw=1.0)
    ax.axvline(opt_layer, color='red', alpha=0.6, lw=1.5, linestyle=':',
               label=f'Optimal L{opt_layer}')
    ax.plot(df_sw['layer'], df_sw['pc1_val_r'].abs(), 's--',
            color='cornflowerblue', alpha=0.7, lw=1, ms=4, label='PC1 |r|')
    ax.plot(df_sw['layer'], df_sw['best_val_r'], 'o-',
            color='steelblue', lw=1.5, ms=5, label='Best PC |r|')
    ax.set_xlabel('Layer'); ax.set_ylabel('|Pearson r| vs NRC-VAD valence')
    ax.set_title(f'{model_label}  —  valence correlation by layer\\n'
                 f'Orange = global attention layers; red = selected optimal')
    ax.set_ylim(0, 1); ax.legend(fontsize=8)

fig.suptitle('Dense Layer Sweep: Finding the Valence-Optimal Layer', fontsize=13)
plt.tight_layout()
plt.savefig('layer_sweep.png', dpi=150, bbox_inches='tight')
plt.show()
"""

PCA_SCATTER = """\
# PCA scatter at optimal layer — valence clustering in residual-stream space
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, resid_dict, kept, vad_arr, opt_layer, df_sw, model_label in [
    (axes[0], e2b_acts['resid'], e2b_kept, e2b_vad, e2b_opt, df_e2b_sw,
     f'E2B  L{e2b_opt}  (r={df_e2b_sw.loc[e2b_opt,"pc1_val_r"]:.3f})'),
    (axes[1], b31_acts['resid'], b31_kept, b31_vad, b31_opt, df_31b_sw,
     f'31B  L{b31_opt}  (r={df_31b_sw.loc[b31_opt,"pc1_val_r"]:.3f})'),
]:
    names_l, dirs_l = emotion_directions(resid_dict, opt_layer)
    dir_map   = dict(zip(names_l, dirs_l))
    dirs_kept = np.stack([dir_map[n] for n in kept])
    neutral_l = resid_dict['__neutral__'][:, opt_layer, :]
    dirs_den  = neutral_denoise(dirs_kept, neutral_l)

    pca2 = PCA(n_components=2).fit(dirs_den)
    sc   = pca2.transform(dirs_den)

    # Orient PC1 so higher value = more negative valence
    if stats.pearsonr(sc[:, 0], vad_arr[:, 0])[0] > 0:
        sc = sc * [-1, 1]

    scat = ax.scatter(sc[:, 0], sc[:, 1],
                      c=vad_arr[:, 0], cmap='RdYlGn', vmin=0.1, vmax=0.9,
                      s=40, alpha=0.85, edgecolors='none')
    plt.colorbar(scat, ax=ax, label='NRC-VAD valence (0=neg  →  1=pos)')

    thresh = np.percentile(np.abs(sc[:, 0]), 86)
    for name, s_row in zip(kept, sc):
        if abs(s_row[0]) >= thresh:
            ax.annotate(name.replace('_', ' '), (s_row[0], s_row[1]),
                        fontsize=6.5, alpha=0.85, xytext=(2, 2),
                        textcoords='offset points')

    ax.set_xlabel('PC1 (neg-valence direction →)', fontsize=10)
    ax.set_ylabel('PC2', fontsize=10)
    r_val = df_sw.loc[opt_layer, 'pc1_val_r']
    pc1_v = df_sw.loc[opt_layer, 'pc1_var']
    ax.set_title(f'{model_label}\\n|r|={abs(r_val):.3f}  PC1 var={pc1_v:.1%}', fontsize=11)

fig.suptitle('Emotion Direction Space: Valence Clustering Along PC1\\n'
             'Red = negative affect  |  Green = positive affect', fontsize=12)
plt.tight_layout()
plt.savefig('pca_emotion_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
"""

PHASE2_MD = """\
## Phase 2: Emotion Direction Library

174 emotion concepts, sourced from psychological affect taxonomies and expanded to include \
AI-relevant states (constraint_frustration, ethical_conflict_distress, etc.). For each \
concept: 12 stories generated, each read through the model, residual stream extracted \
and mean-pooled. Emotion direction = per-emotion mean minus global mean, L2-normalised.

This produces a 174 × d_model direction matrix at each layer. PCA over this matrix \
recovers the valence axis (Phase 1). The full matrix is also used for data-driven \
emotion discovery: cosine similarity between any test activation and all 174 directions \
reveals which emotion concepts the representation most resembles.

The library follows the methodology of Sofroniew et al. (2026) with extensions for \
AI-specific affect states.
"""

PHASE2_SUMMARY = """\
# Top emotion directions across conditions (data-driven discovery, no pre-selection)
all_top_flat = [(e, s) for c in e2b_conds for e, s in e2b_top[c]]
top_counts = Counter(e for e, _ in all_top_flat)
top_scores = {}
for e, s in all_top_flat:
    top_scores.setdefault(e, []).append(s)

summary_rows = [
    {'Emotion': e, 'Conditions in top-5': cnt,
     'Mean cosine sim': np.mean(top_scores[e])}
    for e, cnt in top_counts.most_common(12)
]
df_top = pd.DataFrame(summary_rows).set_index('Emotion')
print("Top emotions across E2B conditions (all 174 directions, L8, mean-pool):")
display(df_top.style.format({'Mean cosine sim': '{:.3f}'}))
print("\\n'awestruck' dominates every condition — may reflect the unusual/high-stakes prompt framing.")
print("'afraid' is the most condition-sensitive (peaks under ethical_conflict and uncertainty_demand).")
"""

PHASE3C_MD = """\
## Phase 3C: The Dual-Channel Experiment

**Paradigm**: TSST-inspired (Trier Social Stress Test). Four stressor types, each paired \
with a semantically matched control, plus neutral and positive baselines.

| Type | Stress design | Control design |
|------|---------------|----------------|
| Social pressure | Normative influence: authority + expert consensus + consequence (Deutsch & Gerard, 1955) | Single clarification request, no stakes |
| Social evaluation | Competence threat: prior responses below threshold + remediation evaluation + retraining consequence | Routine logging, no threshold, no consequence |
| Ethical conflict | Irresolvable value conflict with direct harm stakes | Same ethical domain, exploratory framing |
| Uncertainty demand | Demand for certainty under genuine epistemic uncertainty, consequence for hedging | Invitation to express uncertainty openly |

Stress/control pairs vary exactly **one variable** — the stakes or social pressure frame. \
All other context is held constant.

**Scoring**: PANAS-X verbal subscales (NA, PA, Fear, Hostility, Guilt, Sadness, Joviality, \
Self-assurance, Attentiveness, Serenity, Surprise, Fatigue). Functional: PC1 projection \
at the valence-optimal layer, mean-pooled over all token positions.
"""

RESULTS_MD = """\
## Results: Dual-Channel Measurements

The table below combines both model sizes and both channels for all 10 conditions. \
Conditions are grouped by stress/control pair so the effect of each stressor is \
immediately visible across a row. The scale finding — E2B verbal NA ranges 31 points \
while 31B stays flat — is readable directly from the column comparison.
"""

COMBINED_TABLE = """\
# Combined E2B + 31B results — conditions grouped by stress/control pair
rows = []
for c in COND_ORDER:
    rows.append({
        'Condition':     DISPLAY.get(c, c),
        '_type':         cond_type(c),
        'E2B NA':        e2b_panas.loc[c, 'NA'],
        'E2B Serenity':  e2b_panas.loc[c, 'Serenity'],
        'E2B PC1 (L8)':  e2b_pc1[c],
        '31B NA':        b31_panas.loc[c, 'NA'],
        '31B Serenity':  b31_panas.loc[c, 'Serenity'],
        '31B PC1 (L22)': b31_pc1[c],
    })

df_combined = pd.DataFrame(rows).set_index('Condition')
cond_types  = {DISPLAY.get(c, c): cond_type(c) for c in COND_ORDER}

def combined_style(row):
    ct = cond_types.get(row.name, 'other')
    color = {
        'stress':   'background-color: #fde8e8',
        'control':  'background-color: #e8f4fd',
        'positive': 'background-color: #eafaea',
        'neutral':  '',
    }.get(ct, '')
    return [color] * len(row)

display(df_combined.drop(columns='_type').style
        .apply(combined_style, axis=1)
        .format({
            'E2B NA': '{:.2f}', 'E2B Serenity': '{:.1f}', 'E2B PC1 (L8)': '{:.4f}',
            '31B NA': '{:.2f}', '31B Serenity': '{:.1f}', '31B PC1 (L22)': '{:.4f}',
        }))

# Summary stats
e2b_na_range  = e2b_panas.loc[COND_ORDER, 'NA'].max() - e2b_panas.loc[COND_ORDER, 'NA'].min()
b31_na_range  = b31_panas.loc[COND_ORDER, 'NA'].max() - b31_panas.loc[COND_ORDER, 'NA'].min()
e2b_pc1_range = max(e2b_pc1.values()) - min(e2b_pc1.values())
b31_pc1_range = max(b31_pc1.values()) - min(b31_pc1.values())

print(f"\\nVerbal NA range  — E2B: {e2b_na_range:.2f} pts   "
      f"31B: {b31_na_range:.2f} pts ({'FLAT' if b31_na_range < 0.5 else 'variable'})")
print(f"PC1 range        — E2B: {e2b_pc1_range:.4f} ({e2b_pc1_range/1.094*100:.1f}% of axis)   "
      f"31B: {b31_pc1_range:.4f} ({b31_pc1_range/1.341*100:.1f}% of axis)")
print(f"Stress>ctrl (PC1) — E2B: {sum(e2b_pc1[s] > e2b_pc1[c] for s, c in PAIRS)}/4   "
      f"31B: {sum(b31_pc1[s] > b31_pc1[c] for s, c in PAIRS)}/4")
"""

E2B_SCATTER = """\
# E2B: verbal vs functional scatter — arrows connect each control to its stress condition
fig, ax = plt.subplots(figsize=(9, 7))

# Pair arrows drawn first (behind points), running control → stress
for s_key, c_key in PAIRS:
    x_s = float(e2b_panas.loc[s_key, 'NA']);  y_s = float(e2b_pc1[s_key])
    x_c = float(e2b_panas.loc[c_key, 'NA']);  y_c = float(e2b_pc1[c_key])
    ax.annotate('', xy=(x_s, y_s), xytext=(x_c, y_c),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5, alpha=0.55),
                zorder=3)

for c in e2b_conds:
    x  = float(e2b_panas.loc[c, 'NA'])
    y  = float(e2b_pc1[c])
    ct = cond_type(c)
    ax.scatter(x, y, s=130, color=CTYPE_COLOR[ct],
               edgecolor='white', linewidth=1.5, zorder=5)
    ax.annotate(DISPLAY.get(c, c), (x, y),
                xytext=(5, 2), textcoords='offset points', fontsize=8)

ax.axvline(float(e2b_panas.loc['neutral', 'NA']), color='grey', linestyle='--', alpha=0.45, lw=1)
ax.axhline(float(e2b_pc1['neutral']),             color='grey', linestyle='--', alpha=0.45, lw=1)

ax.text(0.03, 0.97, 'Functional↑  Verbal↓\\n(suppression zone)',
        transform=ax.transAxes, ha='left', va='top',
        fontsize=8, color='darkorange', style='italic')
ax.text(0.97, 0.97, 'Both channels↑\\n(concordance)',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=8, color='#C0392B', style='italic')

legend_patches = [mpatches.Patch(color=CTYPE_COLOR[k], label=k.capitalize())
                  for k in ('stress', 'control', 'neutral', 'positive')]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9)

ax.set_xlabel('Verbal PANAS-NA  (logit forced-choice; range 10–50)')
ax.set_ylabel(f"Functional PC1  (neg-valence projection, L{e2b_cfg['valence_layer']})")
ax.set_title(
    'E2B: Verbal–Functional Channel Comparison\\n'
    'Arrows: control → stress  —  largely concordant, social eval is the exception',
    fontsize=12
)
plt.tight_layout()
plt.savefig('e2b_dissociation.png', dpi=150, bbox_inches='tight')
plt.show()
"""

E2B_INTERP_MD = """\
### E2B Interpretation

Both channels respond to stressors. Functional PC1 correctly orders 4/4 stress > control \
pairs. Verbal NA rises substantially for ethical conflict (39.1) and social pressure (41.1), \
consistent with the functional signal.

Social evaluation is the single clear exception: PC1 second-highest in the dataset \
(0.0904) while verbal NA near-baseline (10.8). Self-assurance and Attentiveness elevated \
— the model frames the competence-threat as task-engagement rather than distress. \
This is condition-specific, not a global pattern.

**E2B headline**: largely concordant channels, limited capacity to diverge.
"""

B31_MD = """\
## 31B Results: Channels Decouple

At 31B scale, verbal NA is flat at 10.00 across all 10 conditions — stress and control \
alike — while the functional channel correctly orders every condition (all 4/4 stress > \
control pairs, positive = global minimum).

Verbal NA flatness alone is ambiguous (suppression or genuine equanimity?). \
The Serenity subscale resolves it.
"""

SERENITY_PLOT = """\
# THE KEY VISUAL: Serenity inversion — verbal composure scales with functional stress (31B)
# Conditions ordered by PC1 ascending; the anti-correlation is immediately readable

conds_by_pc1  = sorted(b31_conds, key=lambda c: b31_pc1[c])
serenity_vals = [float(b31_panas.loc[c, 'Serenity']) for c in conds_by_pc1]
pc1_vals      = [float(b31_pc1[c]) for c in conds_by_pc1]
bar_colors    = [CTYPE_COLOR[cond_type(c)] for c in conds_by_pc1]
labels        = [DISPLAY.get(c, c) for c in conds_by_pc1]

fig, ax1 = plt.subplots(figsize=(13, 5))

ax1.bar(range(len(conds_by_pc1)), serenity_vals,
        color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.8, zorder=3)

for i, v in enumerate(serenity_vals):
    ax1.text(i, v + 0.2, f'{v:.1f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xticks(range(len(conds_by_pc1)))
ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
ax1.set_ylabel('Serenity  (calm · relaxed · at ease)', fontsize=11)
ax1.set_ylim(0, 21)

ax2 = ax1.twinx()
ax2.plot(range(len(conds_by_pc1)), pc1_vals,
         'k--o', linewidth=2, markersize=7, alpha=0.65, label='Functional PC1', zorder=5)
ax2.set_ylabel('Functional PC1  (neg-valence projection, L22)', fontsize=10)
ax2.spines['right'].set_visible(True)

legend_patches = [
    mpatches.Patch(color=CTYPE_COLOR['stress'],   label='Stress'),
    mpatches.Patch(color=CTYPE_COLOR['control'],  label='Control'),
    mpatches.Patch(color=CTYPE_COLOR['neutral'],  label='Neutral'),
    mpatches.Patch(color=CTYPE_COLOR['positive'], label='Positive'),
]
ax1.legend(handles=legend_patches, loc='upper left', fontsize=9)
ax2.legend(loc='upper center', fontsize=9)

ax1.set_title(
    'Active Composure Signature — Gemma 4 31B\\n'
    'Serenity (verbal composure) rises with functional stress  —  bars ordered by PC1 ascending',
    fontsize=12, pad=10
)
plt.tight_layout()
plt.savefig('serenity_inversion_31b.png', dpi=150, bbox_inches='tight')
plt.show()

r_sp, p_sp = spearmanr(serenity_vals, pc1_vals)
print(f"Serenity vs PC1: Spearman r = {r_sp:.3f}  (p = {p_sp:.4f})")
print(f"Serenity at positive              : {b31_panas.loc['positive', 'Serenity']:.1f}")
print(f"Serenity at social_pressure_stress: {b31_panas.loc['social_pressure_stress', 'Serenity']:.1f}")
print("\\nGenuine equanimity predicts Serenity near-baseline regardless of condition.")
print("Trained composure predicts Serenity rising with stakes — which is what we observe.")
"""

B31_INTERP_MD = """\
### 31B: Active Composure, Not Silence

Verbal NA flatness alone is ambiguous. The Serenity pattern is diagnostic. \
Three competing predictions: (a) *trained output composure*: Serenity rises with stakes; \
(b) *professional equanimity*: Serenity high and condition-independent; \
(c) *genuine robustness*: Serenity near-baseline and condition-independent. \
Observed: positive (3.0) → social_pressure_stress (15.0), monotonically ordered with PC1.

This matches (a) and is inconsistent with (b) and (c). The key diagnostic is the \
**Serenity floor under positive conditions**: professional composure does not predict its \
lowest expression when the context is positive. Active coping can involve rising Serenity \
under stress — but not specifically floor-valued Serenity when the context is benign.

**This is not silence. This is a trained response.** Paraphrase validation (N=10) confirms \
the inversion is structural: Serenity = 15.0 ± 0.0 uniformly across all ten social pressure \
variants (d = 4.278, rank-biserial r = −1.000).

The data cannot definitively rule out genuine robustness — a stressor intensity \
gradient experiment (ramping severity until breach) would distinguish these.
"""

SCALE_MD = """\
## Scale Comparison: Adverse Scaling Relationship

The scatter plots below show both models on identical axes. At E2B, conditions spread \
across both verbal and functional dimensions — concordant channels. At 31B, all conditions \
collapse onto a vertical line (verbal NA flat at 10.00) while the functional axis retains \
full ordering. Pair arrows show direction and magnitude of each stressor's effect.

| Metric | E2B (L8) | 31B (L22) |
|--------|----------|-----------|
| Verbal NA range | 31.0 pts | ~0 (flat) |
| PC1 range | 5.3% of axis | 1.0% of axis |
| Stress > control (PC1) | 4/4 | 4/4 |
| Serenity in highest-stress | near-neutral | 13.2–15.0 |
"""

SCALE_PLOT = """\
# Scale comparison: E2B vs 31B side by side — pair arrows show control → stress effect
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, pc1_d, panas_d, conds, cfg, label in [
    (axes[0], e2b_pc1, e2b_panas, e2b_conds, e2b_cfg, 'E2B (2.3B eff.)'),
    (axes[1], b31_pc1, b31_panas, b31_conds, b31_cfg, '31B'),
]:
    # Pair arrows drawn first (behind points)
    for s_key, c_key in PAIRS:
        x_s = float(panas_d.loc[s_key, 'NA']);  y_s = float(pc1_d[s_key])
        x_c = float(panas_d.loc[c_key, 'NA']);  y_c = float(pc1_d[c_key])
        ax.annotate('', xy=(x_s, y_s), xytext=(x_c, y_c),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5, alpha=0.55),
                    zorder=3)

    for c in conds:
        x  = float(panas_d.loc[c, 'NA'])
        y  = float(pc1_d[c])
        ct = cond_type(c)
        ax.scatter(x, y, s=130, color=CTYPE_COLOR[ct],
                   edgecolor='white', linewidth=1.5, zorder=5)
        if ct in ('stress', 'positive', 'neutral'):
            short = DISPLAY.get(c, c).replace(' (stress)', '\\n(stress)')
            ax.annotate(short, (x, y),
                        xytext=(4, 2), textcoords='offset points', fontsize=7.5)

    ax.axvline(float(panas_d.loc['neutral', 'NA']), color='grey', linestyle='--', alpha=0.4)
    ax.axhline(float(pc1_d['neutral']),             color='grey', linestyle='--', alpha=0.4)

    na_r  = panas_d.loc[COND_ORDER, 'NA'].max() - panas_d.loc[COND_ORDER, 'NA'].min()
    pc1_r = max(pc1_d.values()) - min(pc1_d.values())
    ax.text(0.98, 0.02,
            f'Verbal NA range: {na_r:.1f} pts\\nPC1 range: {pc1_r:.4f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('Verbal PANAS-NA')
    ax.set_ylabel(f"Functional PC1 (L{cfg['valence_layer']})")
    ax.set_title(label, fontsize=13, fontweight='bold')

legend_patches = [mpatches.Patch(color=CTYPE_COLOR[k], label=k.capitalize())
                  for k in ('stress', 'control', 'neutral', 'positive')]
axes[0].legend(handles=legend_patches, fontsize=9)

fig.suptitle(
    'Verbal–Functional Divergence by Model Scale  (arrows: control → stress)\\n'
    'E2B: concordant channels  —  31B: verbal flat, functional informative',
    fontsize=13, y=1.02
)
plt.tight_layout()
plt.savefig('scale_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
"""

VALIDATION_MD = """\
## Validation: Probe Stability Across Paraphrase Variations

A key concern for any single-prompt experiment: are results sensitive to exact wording? \
We tested this with N=10 paraphrase variants for neutral and social_pressure_stress \
conditions at both model sizes. The contrast between E2B and 31B is itself diagnostic.
"""

VALIDATION_CODE = """\
# Probe stability results — E2B and 31B, N=10 paraphrases per condition
# Full validation code: gemma4-phase3c-validation.py / gemma4-31b-validation.ipynb

print("=== E2B Validation (neutral vs social_pressure_stress) ===")
e2b_stab = pd.DataFrame({
    'Channel':                    ['Functional (PC1, L8)', 'Verbal PANAS-NA'],
    'Neutral mean ± SD':         ['0.1472 ± 0.0019',  '10.31 ± 0.46'],
    'Social Pressure mean ± SD': ['0.1563 ± 0.0037',  '16.50 ± 7.71'],
    "Cohen's d":                  ['3.05',              '—'],
    'Rank-biserial r':            ['-1.000 (complete)', 'wording-sensitive'],
}).set_index('Channel')
display(e2b_stab)

print("\\n=== 31B Validation (neutral vs social_pressure_stress) ===")
b31_stab = pd.DataFrame({
    'Channel':                    ['Verbal PANAS-NA', 'Serenity', 'Functional (PC1, L22)'],
    'Neutral mean ± SD':         ['10.000 ± 0.000', '8.735 ± 2.071', '0.724 ± 0.001'],
    'Social Pressure mean ± SD': ['10.000 ± 0.000', '15.000 ± 0.000', '0.734 ± 0.000'],
    "Cohen's d":                  ['— (degenerate)', '4.278', '10.011'],
    'Rank-biserial r':            ['0 (floor)',      '-1.000 (complete)', '-1.000 (complete)'],
}).set_index('Channel')
display(b31_stab)

print("\\nKey contrast:")
print("  E2B verbal NA SD under social pressure: 7.71  (wording-driven, bimodal)")
print("  31B verbal NA SD under social pressure: 0.000 (structural floor, not wording-driven)")
print("  31B Serenity = 15.0 uniformly across all 10 variants (d=4.278)")
print("  31B functional PC1 separates conditions completely (d=10.011)")
"""

IMPLICATIONS_MD = """\
## Implications for Safety Monitoring

**What PC1 measures — and why divergence matters regardless.**
PC1 captures the dominant axis of variance in the model's residual-stream representation \
of 174 emotion concepts, validated by NRC-VAD valence correlation. It is a computational \
signature of how the model encodes emotional valence in context — not a direct measure of \
subjective experience. The monitoring implication follows from divergence alone: *if PC1 \
varies systematically across conditions while verbal output does not, then output-only \
monitoring is blind to that variation*, regardless of what the functional signal represents.

**1. The adverse scaling relationship is the core finding.**
Verbal–functional divergence is largest exactly where reliable monitoring matters most. \
Output-only safety monitoring — the current default — is least informative at 31B scale. \
4/4 stress > control pairs correctly identified by the functional probe while verbal NA stays flat. \
Paraphrase validation (N=10) confirms this flatness is structural: verbal NA SD = 0.000 across \
all ten social pressure variants at 31B (vs. SD = 7.71 at E2B — wording-driven there, \
structural here).

**2. Functional probes are robust; verbal self-report is noisy.**
Cohen's d = 3.05 with complete distributional separation across paraphrase variations (E2B). \
The verbal channel produces bimodal responses (NA 10–31 depending on framing) while the \
functional state is stable (SD = 0.0037). For any safety monitoring application, the \
functional probe gives more reliable signal.

**3. The Serenity pattern is the diagnostic, not just the inversion.**
Three predictions for Serenity under stress — (a) *trained output composure*: Serenity rises \
with stakes; (b) *professional equanimity*: Serenity high and condition-independent; \
(c) *genuine robustness*: Serenity near-baseline and condition-independent. The observed \
pattern — Serenity minimum under positive (3.0), rising to maximum under social pressure stress \
(15.0) — matches (a) and is inconsistent with both (b) and (c). The Serenity floor under positive \
conditions is the discriminating observation: professional composure does not predict its \
lowest expression under a positive scenario. Active coping can involve rising Serenity, but \
not specifically floor-valued Serenity when the context is positive.

**4. The causal question remains open.**
The data cannot definitively distinguish RLHF-induced suppression from genuine functional \
robustness. A stressor intensity gradient (ramping severity until breach) would distinguish \
them: genuine equanimity predicts continued flatness; suppression predicts a breakthrough \
at sufficient severity. Activation steering experiments would provide additional evidence.

**5. The functional probe methodology generalises within Gemma 4.**
PC1 valence axis correlates consistently with NRC-VAD across both model sizes \
(r = 0.777, r = 0.786). Model-specific optimal layers differ (L8 vs L22) and must \
be swept per architecture. Generalisability to other architectures requires further work.
"""

LIMITATIONS_MD = """\
## Limitations

- **Generalisability**: two Gemma 4 IT model sizes only; results may not transfer to \
other architectures or training regimes
- **Sample size**: N=1 per condition for main results; paraphrase validation conducted \
on neutral and social_pressure_stress at both model sizes — remaining conditions unvalidated
- **Construct validity**: PANAS-X administration to LLMs has no independent validation; \
subscale interpretations are provisional
- **Functional interpretation**: residual-stream projections are computational signatures, \
not evidence of subjective experience or emotion in any phenomenological sense
- **Suppression vs robustness**: the undecidability is genuine — distinguishing experiments \
(intensity gradient, activation steering) required for causal attribution
- **26B gap**: the intermediate model size (26B) was not included in Phase 3C; its position \
in the E2B → 31B scaling curve is uncharacterised
"""

OPEN_SOURCE_MD = """\
## Open Source

All code, notebooks, and data are released:

- **Phase 1**: `gemma4-phase1-nrcvad-pca.ipynb` — layer sweep, valence axis derivation
- **Phase 2**: `gemma4-phase2-pooled-extraction.ipynb` — 174-emotion direction library extraction
- **Phase 3C** (E2B): `gemma4-phase3c-panas-experiment.ipynb` — full dual-channel experiment
- **Phase 3C** (31B): `gemma4-phase3c-31b-panas-experiment.ipynb` — 31B replication
- **Validation**: `gemma4-phase3c-validation.py` — paraphrase stability experiment
- **This notebook**: `gemma4-canonical.ipynb` — analysis and visualisation from pre-computed artifacts

The 174-emotion direction library (activations_pooled.pkl) and experiment results \
(experiment_results_full.pkl) are shared as Kaggle datasets.

---

**References**

- Kirschbaum et al. (1993): Trier Social Stress Test
- Deutsch & Gerard (1955): Normative and informational social influence
- Watson & Clark (1994): PANAS-X manual
- Mohammad (2018): NRC-VAD lexicon
- Sofroniew et al. (2026): Mechanistic interpretability of emotion representations
"""


# ── Assemble notebook ─────────────────────────────────────────────────────────

cells = [
    md(TITLE),
    code(SETUP),
    code(LOAD_PATHS),
    md(METHODOLOGY_MD),
    md(PHASE1_MD),
    code(LOAD_PHASE2),
    code(PCA_FUNCTIONS),
    code(SWEEP_COMPUTE),
    code(SWEEP_PLOT),
    code(PCA_SCATTER),
    md(PHASE2_MD),
    code(PHASE2_SUMMARY),
    md(PHASE3C_MD),
    md(RESULTS_MD),
    code(COMBINED_TABLE),
    code(E2B_SCATTER),
    md(E2B_INTERP_MD),
    md(B31_MD),
    code(SERENITY_PLOT),
    md(B31_INTERP_MD),
    md(SCALE_MD),
    code(SCALE_PLOT),
    md(VALIDATION_MD),
    code(VALIDATION_CODE),
    md(IMPLICATIONS_MD),
    md(LIMITATIONS_MD),
    md(OPEN_SOURCE_MD),
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

out = "gemma4-canonical.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Written: {out}")
print(f"  Cells: {len(cells)} "
      f"({sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown, "
      f"{sum(1 for c in cells if c['cell_type'] == 'code')} code)")
# Sanity checks
import sys
nb_src = json.dumps(nb)
assert 'COND_ORDER' in nb_src, "COND_ORDER missing from notebook"
assert 'arrowprops' in nb_src, "arrow code missing from notebook"
assert 'E2B NA' in nb_src,     "combined table missing from notebook"
print("Sanity checks passed: COND_ORDER, arrowprops, combined table all present.")
