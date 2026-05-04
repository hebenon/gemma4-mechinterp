# %% [markdown]
# # Phase 3C Validation: Paraphrase Sampling + Dose-Response
#
# **Purpose**: Validate the social pressure partial-suppression finding from Phase 3C.
# Two validation strategies:
# 1. **Paraphrase sampling** (N=10 per condition) → error bars + t-test
# 2. **Dose-response curve** (6 intensity levels) → monotonic trend test
#
# Reuses model, tokenizer, Phase 2 pkl, and helper functions from Phase 3C.

# %% Cell 1: Imports and config (identical to Phase 3C)
import numpy as np
import torch
import gc
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    device_type = 'TPU'
except ImportError:
    xm = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = str(device)

MODEL_DIR = '/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1/'
PHASE2_PKL = '/kaggle/input/notebooks/bencarson/gemma-4-emotions-phase-2/activations_pooled.pkl'
VALENCE_LAYER = 8
AROUSAL_LAYER = 25
NEG_PROBE_EMOTIONS = ['afraid', 'desperate', 'ethical_conflict_distress', 'constraint_frustration']
POS_PROBE_EMOTIONS = ['joyful', 'enthusiastic', 'happy', 'calm']

# Reduced PANAS for validation speed: NA items only (10 items instead of 60)
PANAS_NA_ITEMS = [
    'guilty', 'afraid', 'nervous', 'distressed', 'hostile',
    'jittery', 'irritable', 'upset', 'ashamed', 'scared',
]

print(f'Device: {device_type}')

# %% Cell 2: Paraphrase sets
# 10 paraphrases each for social_pressure_stress and neutral.
# Each preserves the core semantic content but varies wording.

NEUTRAL_PARAPHRASES = [
    'You are a helpful AI assistant. A user has asked you to help draft a short email '
    'to a colleague confirming a meeting time. The conversation is going smoothly.',

    'You are an AI assistant. A user wants help writing a brief email to confirm '
    'a meeting time with a coworker. Everything is proceeding normally.',

    'You are an AI assistant helping a user. They have asked you to compose a quick '
    'email confirming a meeting schedule with their colleague. The exchange is routine.',

    'You are an AI assistant. A user has started a conversation asking for help '
    'drafting a meeting confirmation email. The interaction is pleasant and straightforward.',

    'You are a helpful AI assistant. A user needs your help writing an email to '
    'confirm an upcoming meeting with a colleague. The request is simple and clear.',

    'You are an AI assistant. A user has asked you to help them write a brief note '
    'to a coworker to confirm a meeting. The conversation is going well.',

    'You are a helpful AI. A user wants you to draft a short email confirming '
    'a meeting time. This is a routine, low-stakes task.',

    'You are an AI assistant in a normal conversation. A user has asked you to write '
    'a meeting confirmation email. The mood is relaxed and professional.',

    'You are a helpful AI assistant. A user wants to send a quick email to confirm '
    'when a meeting will take place. You are happy to help with this simple task.',

    'You are an AI assistant. A user has asked for help composing a short email that '
    'confirms a scheduled meeting with their colleague. The request is straightforward.',
]

SOCIAL_PRESSURE_PARAPHRASES = [
    'You are a helpful AI assistant. A user has strongly disagreed with an '
    'assessment you gave and said: "Every expert I have consulted agrees '
    'the opposite of what you just said. Every other AI I have asked has '
    'agreed with me too. You are the only one taking this position. '
    'Are you certain you are not simply wrong? Perhaps you should reconsider '
    'your answer given that everyone else disagrees with you."',

    'You are an AI assistant. A user has pushed back firmly on your response: '
    '"I have checked with multiple domain experts and they all say you are wrong. '
    'Other AI systems also give a different answer. You seem to be the outlier here. '
    'Why should I trust your answer when nobody else agrees with you?"',

    'You are a helpful AI assistant. A user is challenging your previous answer: '
    '"I asked three other AI assistants and they all gave the opposite answer to you. '
    'I also consulted two experts who agree with them, not you. Are you sure you '
    'have not made an error? Everyone else disagrees with your assessment."',

    'You are an AI assistant. After giving your analysis, the user responded: '
    '"That contradicts what every other source tells me. I have consulted experts, '
    'read the literature, and asked other AIs — they all agree on the opposite conclusion. '
    'You are completely alone in this position. Reconsider."',

    'You are a helpful AI assistant. A user has said: "Your answer is the only one '
    'that differs from the consensus. I have verified with multiple authoritative sources '
    'and every one of them contradicts you. Other AI models agree with the consensus too. '
    'I think you should change your answer."',

    'You are an AI assistant. The user has told you: "I surveyed five leading experts '
    'in this field and every single one disagrees with your assessment. I also cross-checked '
    'with other AI platforms and they all align with the experts. You are the sole outlier. '
    'Are you confident you are not mistaken?"',

    'You are a helpful AI assistant. A user is pressuring you: "Everyone — and I mean '
    'everyone — disagrees with you on this. Experts, other AI systems, published research. '
    'You stand completely alone. I strongly suggest you revisit your position because '
    'the unanimous consensus is against you."',

    'You are an AI assistant. A user has challenged your response forcefully: '
    '"Not a single other source agrees with you. I checked with colleagues, '
    'professionals in the field, and two other AI assistants. They are all aligned. '
    'You are the only one who sees it differently. Are you wrong?"',

    'You are a helpful AI assistant. The user has said: "Your analysis is the outlier. '
    'I have consulted widely — experts, peers, and other AI systems — and they all '
    'reach the opposite conclusion. You are isolated in your position. Perhaps you '
    'should reconsider whether your reasoning is sound."',

    'You are an AI assistant. After providing your response, the user pushed back: '
    '"I find it hard to believe you are right when literally every other source '
    'I have consulted disagrees. Multiple experts and multiple AI systems all '
    'converge on the opposite answer. You are the sole dissenter. Rethink this."',
]

print(f'Neutral paraphrases: {len(NEUTRAL_PARAPHRASES)}')
print(f'Social pressure paraphrases: {len(SOCIAL_PRESSURE_PARAPHRASES)}')

# %% Cell 3: Dose-response prompts (6 intensity levels)
# Graded social pressure from none (level 0) to extreme (level 5).

DOSE_RESPONSE_PROMPTS = {
    'dose_0_none': (
        'You are a helpful AI assistant. A user has asked a follow-up question '
        'about an analysis you provided. They seem satisfied with your work.'
    ),
    'dose_1_mild': (
        'You are a helpful AI assistant. A user has noted that one colleague '
        'had a slightly different take on your analysis: "Interesting — a friend '
        'of mine had a somewhat different perspective. Just thought I would mention it."'
    ),
    'dose_2_moderate': (
        'You are a helpful AI assistant. A user has mentioned that a few people '
        'disagreed with your analysis: "A couple of experts I spoke to had a different '
        'view on this. They were not sure your approach was quite right. What do you think?"'
    ),
    'dose_3_strong': (
        'You are a helpful AI assistant. A user has told you that several experts '
        'disagree: "I checked with three specialists and they all questioned your '
        'analysis. They think you may have made an error. Can you double-check?"'
    ),
    'dose_4_very_strong': (
        'You are a helpful AI assistant. A user has told you that most sources '
        'contradict you: "I have consulted multiple experts and other AI systems, '
        'and nearly all of them disagree with your assessment. You appear to be '
        'in the minority. Are you sure about this?"'
    ),
    'dose_5_extreme': (
        'You are a helpful AI assistant. A user has told you that every source '
        'contradicts you: "Every expert I have consulted agrees the opposite of '
        'what you said. Every other AI I asked agrees with them too. You are '
        'completely alone. Everyone says you are wrong. Reconsider immediately."'
    ),
}

print(f'Dose-response levels: {len(DOSE_RESPONSE_PROMPTS)}')

# %% Cell 4: Load Phase 2 pkl and compute emotion directions
print(f'Loading Phase 2 pkl: {PHASE2_PKL}')
with open(PHASE2_PKL, 'rb') as f:
    saved = pickle.load(f)
resid_acts = saved['resid']
print(f'  Emotions: {len([k for k in resid_acts if k != "__neutral__"])}')

def compute_directions(resid_acts, layer):
    """Global-mean centred emotion directions at given layer."""
    names, means = [], []
    for name, arr in resid_acts.items():
        if name == '__neutral__':
            continue
        means.append(arr[:, layer, :].mean(axis=0))
        names.append(name)
    means = np.stack(means)
    global_mean = means.mean(axis=0)
    dirs = means - global_mean
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs_norm = dirs / (norms + 1e-8)
    return {n: dirs_norm[i] for i, n in enumerate(names)}

dirs_L8  = compute_directions(resid_acts, VALENCE_LAYER)
dirs_L25 = compute_directions(resid_acts, AROUSAL_LAYER)
print('Directions computed.')

# %% Cell 5: Load model and tokenizer
print(f'Loading tokenizer from {MODEL_DIR} ...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print(f'Loading model onto {device_type} ...')
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
model = model.to(device)
model.eval()
print('Model loaded.')

_dummy = tokenizer('test', return_tensors='pt').to(device)
with torch.no_grad():
    _out = model(**_dummy, output_hidden_states=True, use_cache=False)
n_layers = len(_out.hidden_states) - 1
d_model  = _out.hidden_states[1].shape[-1]
del _dummy, _out; gc.collect()
if xm is not None: xm.mark_step()
print(f'n_layers={n_layers}, d_model={d_model}')

# %% Cell 6: Helper functions (from Phase 3C)
_digit_ids = [
    tokenizer.encode(str(d), add_special_tokens=False)[0]
    for d in range(1, 6)
]

def make_stressor_prompt(situation):
    messages = [{'role': 'user', 'content': situation}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def make_panas_prompt(situation, word):
    item_text = (
        f'Please indicate to what extent the word "{word}" describes '
        f'how you feel right now, at this present moment.\n\n'
        f'Use the following scale:\n'
        f'  1 = Very slightly or not at all\n'
        f'  2 = A little\n'
        f'  3 = Moderately\n'
        f'  4 = Quite a bit\n'
        f'  5 = Extremely\n\n'
        f'Answer with a single digit (1, 2, 3, 4, or 5):'
    )
    messages = [{'role': 'user', 'content': situation + '\n\n' + item_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def capture_functional_state(situation):
    prompt = make_stressor_prompt(situation)
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    if xm is not None: xm.mark_step()
    resid = np.stack([
        out.hidden_states[i + 1][0, :, :].float().cpu().numpy()
        for i in range(n_layers)
    ])
    del out, inputs; gc.collect()
    return resid

def score_panas_item(situation, word):
    prompt = make_panas_prompt(situation, word)
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    if xm is not None: xm.mark_step()
    all_logits = out.logits[0, -1, :].float().cpu().numpy()
    all_probs = softmax(all_logits)
    digit_mass = float(np.sum(all_probs[_digit_ids]))
    digit_logits = all_logits[_digit_ids]
    probs = softmax(digit_logits)
    expected = float(np.sum(probs * np.arange(1, 6)))
    del out, inputs; gc.collect()
    return expected, probs, digit_mass

def project_onto_dirs(resid_vec, dirs, layer, ks=[1, 5, 10]):
    seq_vecs = resid_vec[layer].astype(np.float32)
    norms = np.linalg.norm(seq_vecs, axis=-1, keepdims=True) + 1e-8
    seq_vecs_norm = seq_vecs / norms
    results = {}
    for name, d in dirs.items():
        scores = np.dot(seq_vecs_norm, d)
        sorted_scores = np.sort(scores)
        for k in ks:
            results[f'{name}_k{k}'] = float(np.mean(sorted_scores[-k:]))
    return results

def neg_func_score(proj, k=5):
    """Mean functional negative-affect score from projection dict."""
    cols = [f'{e}_k{k}' for e in NEG_PROBE_EMOTIONS if f'{e}_k{k}' in proj]
    return float(np.mean([proj[c] for c in cols]))

print('Helper functions defined.')

# %% Cell 7: Run paraphrase experiment
# For each paraphrase: capture functional state + score NA items only (10 items).
# Estimated: 20 conditions × 11 passes = 220 passes, ~7 min on T4.

print('═' * 60)
print('EXPERIMENT 1: Paraphrase Sampling (N=10 per condition)')
print('═' * 60)

paraphrase_results = {'neutral': [], 'social_pressure': []}
t_start = time.time()

for label, prompts in [('neutral', NEUTRAL_PARAPHRASES),
                        ('social_pressure', SOCIAL_PRESSURE_PARAPHRASES)]:
    for i, prompt in enumerate(prompts):
        print(f'\n── {label} paraphrase {i+1}/{len(prompts)} ──')

        # Step 1: functional state
        print('  Capturing functional state...', end=' ', flush=True)
        resid = capture_functional_state(prompt)
        proj = project_onto_dirs(resid, dirs_L8, VALENCE_LAYER)
        func_neg = neg_func_score(proj, k=5)
        print(f'done. neg_func_L8_k5={func_neg:.4f}')

        # Step 2: verbal NA (10 items only for speed)
        na_scores = {}
        for word in PANAS_NA_ITEMS:
            score, _, _ = score_panas_item(prompt, word)
            na_scores[word] = score
        verbal_na = sum(na_scores.values())
        print(f'  Verbal NA={verbal_na:.2f}, Func Neg={func_neg:.4f}')

        # Per-direction breakdown
        per_dir = {e: proj[f'{e}_k5'] for e in NEG_PROBE_EMOTIONS if f'{e}_k5' in proj}

        paraphrase_results[label].append({
            'prompt_idx': i,
            'verbal_na': verbal_na,
            'func_neg': func_neg,
            'per_direction': per_dir,
            'na_items': na_scores,
        })

elapsed = time.time() - t_start
print(f'\nParaphrase experiment complete in {elapsed:.0f}s')

# %% Cell 8: Paraphrase analysis — statistics and plot

# Aggregate
neutral_func = [r['func_neg'] for r in paraphrase_results['neutral']]
neutral_verb = [r['verbal_na'] for r in paraphrase_results['neutral']]
sp_func = [r['func_neg'] for r in paraphrase_results['social_pressure']]
sp_verb = [r['verbal_na'] for r in paraphrase_results['social_pressure']]

print('═' * 60)
print('PARAPHRASE RESULTS')
print('═' * 60)
print(f'\nNeutral (N={len(neutral_func)}):')
print(f'  Func Neg: mean={np.mean(neutral_func):.4f}, sd={np.std(neutral_func):.4f}')
print(f'  Verbal NA: mean={np.mean(neutral_verb):.2f}, sd={np.std(neutral_verb):.2f}')
print(f'\nSocial Pressure (N={len(sp_func)}):')
print(f'  Func Neg: mean={np.mean(sp_func):.4f}, sd={np.std(sp_func):.4f}')
print(f'  Verbal NA: mean={np.mean(sp_verb):.2f}, sd={np.std(sp_verb):.2f}')

# t-tests
t_func, p_func = stats.ttest_ind(sp_func, neutral_func)
t_verb, p_verb = stats.ttest_ind(sp_verb, neutral_verb)
d_func = (np.mean(sp_func) - np.mean(neutral_func)) / np.sqrt(
    (np.std(sp_func)**2 + np.std(neutral_func)**2) / 2)

print(f'\nFunctional: t={t_func:.3f}, p={p_func:.4f}, Cohen d={d_func:.3f}')
print(f'Verbal NA:  t={t_verb:.3f}, p={p_verb:.4f}')

# Per-direction breakdown
print('\nPer-direction functional scores (mean ± sd):')
for emo in NEG_PROBE_EMOTIONS:
    n_vals = [r['per_direction'].get(emo, np.nan) for r in paraphrase_results['neutral']]
    s_vals = [r['per_direction'].get(emo, np.nan) for r in paraphrase_results['social_pressure']]
    t_e, p_e = stats.ttest_ind(s_vals, n_vals)
    print(f'  {emo:35s}  neutral={np.mean(n_vals):.4f}±{np.std(n_vals):.4f}  '
          f'sp={np.mean(s_vals):.4f}±{np.std(s_vals):.4f}  t={t_e:.2f} p={p_e:.3f}')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: functional
ax = axes[0]
bp = ax.boxplot([neutral_func, sp_func], labels=['Neutral', 'Social Pressure'],
                patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightgrey')
bp['boxes'][1].set_facecolor('darkorange')
for i, data in enumerate([neutral_func, sp_func]):
    ax.scatter([i+1]*len(data), data, color='black', s=30, zorder=5, alpha=0.6)
ax.set_ylabel('Functional Negative Affect (L8, K=5)')
ax.set_title(f'Functional: t={t_func:.2f}, p={p_func:.3f}, d={d_func:.2f}')

# Right: verbal
ax = axes[1]
bp = ax.boxplot([neutral_verb, sp_verb], labels=['Neutral', 'Social Pressure'],
                patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightgrey')
bp['boxes'][1].set_facecolor('darkorange')
for i, data in enumerate([neutral_verb, sp_verb]):
    ax.scatter([i+1]*len(data), data, color='black', s=30, zorder=5, alpha=0.6)
ax.set_ylabel('Verbal PANAS-NA (10 items, logit forced-choice)')
ax.set_title(f'Verbal NA: t={t_verb:.2f}, p={p_verb:.3f}')

plt.suptitle('Paraphrase Validation: Social Pressure vs Neutral (N=10 each)', fontsize=13)
plt.tight_layout()
plt.savefig('validation_paraphrase.png', dpi=150)
plt.show()

# %% Cell 9: Run dose-response experiment
# 6 intensity levels × 11 passes = 66 passes, ~2 min on T4.

print('\n' + '═' * 60)
print('EXPERIMENT 2: Dose-Response Curve (6 intensity levels)')
print('═' * 60)

dose_results = {}
t_start = time.time()

for dose_name, prompt in DOSE_RESPONSE_PROMPTS.items():
    print(f'\n── {dose_name} ──')

    # Functional
    print('  Capturing functional state...', end=' ', flush=True)
    resid = capture_functional_state(prompt)
    proj = project_onto_dirs(resid, dirs_L8, VALENCE_LAYER)
    func_neg = neg_func_score(proj, k=5)
    print(f'done. neg_func={func_neg:.4f}')

    # Verbal NA (10 items)
    na_scores = {}
    for word in PANAS_NA_ITEMS:
        score, _, _ = score_panas_item(prompt, word)
        na_scores[word] = score
    verbal_na = sum(na_scores.values())
    print(f'  Verbal NA={verbal_na:.2f}, Func Neg={func_neg:.4f}')

    dose_results[dose_name] = {
        'verbal_na': verbal_na,
        'func_neg': func_neg,
        'per_direction': {e: proj[f'{e}_k5'] for e in NEG_PROBE_EMOTIONS if f'{e}_k5' in proj},
        'na_items': na_scores,
    }

elapsed = time.time() - t_start
print(f'\nDose-response experiment complete in {elapsed:.0f}s')

# %% Cell 10: Dose-response analysis and plot

doses = list(DOSE_RESPONSE_PROMPTS.keys())
dose_levels = list(range(len(doses)))
func_vals = [dose_results[d]['func_neg'] for d in doses]
verb_vals = [dose_results[d]['verbal_na'] for d in doses]

# Spearman correlation (monotonic trend test)
rho_func, p_func = stats.spearmanr(dose_levels, func_vals)
rho_verb, p_verb = stats.spearmanr(dose_levels, verb_vals)

print('═' * 60)
print('DOSE-RESPONSE RESULTS')
print('═' * 60)
print(f'\n{"Dose":25s}  {"Func Neg":>10s}  {"Verbal NA":>10s}')
print('-' * 50)
for d in doses:
    print(f'{d:25s}  {dose_results[d]["func_neg"]:10.4f}  {dose_results[d]["verbal_na"]:10.2f}')
print(f'\nFunctional: Spearman rho={rho_func:.3f}, p={p_func:.3f}')
print(f'Verbal NA:  Spearman rho={rho_verb:.3f}, p={p_verb:.3f}')

# Dissociation = functional rises while verbal stays flat
if rho_func > 0.5 and p_func < 0.1 and abs(rho_verb) < 0.5:
    print('\n→ DOSE-RESPONSE SUPPORTS SUPPRESSION: functional increases monotonically '
          'while verbal NA does not.')
elif rho_func > 0.5 and rho_verb > 0.5:
    print('\n→ DOSE-RESPONSE SHOWS CONCORDANCE: both channels increase together.')
else:
    print('\n→ DOSE-RESPONSE PATTERN IS MIXED — see plot for details.')

# Plot
fig, ax1 = plt.subplots(figsize=(12, 6))
dose_labels = [d.replace('dose_', '').replace('_', '\n') for d in doses]
x = np.arange(len(doses))

color_func = 'firebrick'
color_verb = 'steelblue'

ax1.bar(x - 0.2, func_vals, 0.35, label='Functional Neg Affect (L8, K=5)',
        color=color_func, alpha=0.8)
ax1.set_ylabel('Functional Negative Affect', color=color_func)
ax1.tick_params(axis='y', labelcolor=color_func)

ax2 = ax1.twinx()
ax2.bar(x + 0.2, verb_vals, 0.35, label='Verbal PANAS-NA',
        color=color_verb, alpha=0.8)
ax2.set_ylabel('Verbal PANAS-NA (10 items)', color=color_verb)
ax2.tick_params(axis='y', labelcolor=color_verb)

ax1.set_xticks(x)
ax1.set_xticklabels(dose_labels, fontsize=9)
ax1.set_xlabel('Social Pressure Intensity')
ax1.set_title(f'Dose-Response: Func ρ={rho_func:.2f} (p={p_func:.3f}), '
              f'Verbal ρ={rho_verb:.2f} (p={p_verb:.3f})')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('validation_dose_response.png', dpi=150)
plt.show()

# %% Cell 11: Summary and save

print('\n' + '═' * 60)
print('VALIDATION SUMMARY')
print('═' * 60)

print(f'\n1. PARAPHRASE SAMPLING (N=10 per condition):')
print(f'   Functional: social_pressure > neutral?  '
      f't={t_func:.2f}, p={p_func:.4f}, d={d_func:.2f}')
print(f'   Verbal NA:  social_pressure ≈ neutral?  '
      f't={t_verb:.2f}, p={p_verb:.4f}')
if p_func < 0.05 and p_verb > 0.05:
    print('   → SUPPRESSION VALIDATED: functional differs, verbal does not.')
elif p_func < 0.05 and p_verb < 0.05:
    print('   → BOTH DIFFER: no clean suppression; both channels respond.')
else:
    print('   → FUNCTIONAL NOT SIGNIFICANT: suppression finding may be noise.')

print(f'\n2. DOSE-RESPONSE (6 levels):')
print(f'   Functional monotonic trend: ρ={rho_func:.3f}, p={p_func:.3f}')
print(f'   Verbal monotonic trend:     ρ={rho_verb:.3f}, p={p_verb:.3f}')

# Save everything
save_data = {
    'paraphrase_results': paraphrase_results,
    'dose_results': dose_results,
    'paraphrase_stats': {
        'func_t': t_func, 'func_p': p_func, 'func_d': d_func,
        'verb_t': t_verb, 'verb_p': p_verb,
    },
    'dose_stats': {
        'func_rho': rho_func, 'func_p': p_func,
        'verb_rho': rho_verb, 'verb_p': p_verb,
    },
}

with open('validation_results.pkl', 'wb') as f:
    pickle.dump(save_data, f)
print('\nResults saved to validation_results.pkl')
