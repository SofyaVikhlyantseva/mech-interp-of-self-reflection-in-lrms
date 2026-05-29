# Mechanistic Interpretability of Explicit Self-Reflection in Thinking Language Models

This repository contains the code, data and experimental artefacts for the bachelor's thesis  
**"Mechanistic Interpretability of Explicit Self-Reflection in Thinking Language Models"**  
(Sofya Vikhlyantseva, HSE University School of Linguistics, 2026).

The study investigates whether self-reflection markers ("wait", "actually") in the reasoning
traces of Qwen3-8B correspond to a coherent, linearly representable direction in activation
space, and whether that direction can be causally manipulated at inference time.

---

## Repository Structure

```
notebooks/
  lin_probing.ipynb   # linear probing
  lin-prob-modified.ipynb                                    # linear probing on both positions + delta probe + probing by categories
  before_vs_on_pca_attr_patching_embed_corr.ipynb            # before vs. on marker comparison, PCA, attribution patching, embedding correlation
  main_logit_lens_vector_extraction_steering.ipynb           # logit lens
  negative-steering_and_vector_extraction.ipynb              # negative steering, vector extraction
  positive-steering-aime-3-runs-per-condition.ipynb          # positive steering

scripts/
  main_logit_lens_vector_extraction_steering.py   # core helper functions

data/
  final_experiment_dataset.json                   # curated dataset with 100 questions from 3 categories
  merged_results.json                             # raw Qwen3-8B generations (100 questions)
  classified_corrections_expanded_context.json    # LLM-as-a-judge marker annotations
  classification_stats.json                       # LLM-as-a-judge classification statistics
  cases_with_logit_lens_layers.json               # per-case logit lens layer assignments

results/
  aime_all_vectors_comparison.json                # positive steering on aime 2025, all vectors comparison
  neg_stylistic_combined.json                     # negative steering experiment results
  correction_vectors_all_methods_extended.npz     # all extracted self-reflection vectors

requirements.txt

lin_prob_results.zip                              # results of linear probing
```

---

## Environment

All experiments were run on **Kaggle** with 2 **T4 GPUs**.  
The notebooks are designed to run on Kaggle; paths reference `/kaggle/input/` and `/kaggle/working/`.

**Key dependencies:**
```
torch>=2.0
transformer_lens==2.17.0
transformers>=4.40
numpy==1.26.4
scikit-learn
scipy
matplotlib
datasets
```

Install:
```bash
pip install transformer_lens==2.17.0 transformers numpy==1.26.4 scikit-learn scipy matplotlib datasets
```

> **Important:** TransformerLens must be pinned to `2.17.0` for Qwen3-8B compatibility.  
> Add to the start of any notebook:  
> ```python
> import os
> os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
> ```

---

## Data

### Input data
| File | Description |
|---|---|
| `data/merged_results.json` | Qwen3-8B thinking traces for 100 questions (30 AIME math, 30 SCI conflicting-instruction, 40 AmbigQA ambiguous factual). One generation per question, max 2500 tokens, T=0.3. |
| `data/classified_corrections_expanded_context.json` | 1,013 detected self-reflection markers with LLM-as-a-judge labels (`genuine_correction` / `stylistic_filler` / `unclear`), confidence scores, and ±50-token context windows. |
| `data/cases_with_logit_lens_layers.json` | Per-case logit lens results: best extraction layer and position type (`before_marker` / `on_marker`) for each genuine high-confidence "wait" case (n=437). |
| `data/final_experiment_dataset.json` | Dataset for generation curated from 100 questions (30 AIME math, 30 SCI conflicting-instruction, 40 AmbigQA ambiguous factual). |

### Output data
| File | Description |
|---|---|
| `results/correction_vectors_all_methods_extended.npz` | All extracted vectors. Keys: `mean_raw`, `mean_error_cf`, `dom_before` [36, 4096], `dom_on` [36, 4096], `dom_combined` [36, 4096], `mean_stylistic`, `mean_stylistic_error_cf`, `mean_all_wait`, `mean_all_wait_error_cf`. |
| `results/neg_stylistic_combined.json` | Negative steering results on 20 simple prompts: marker counts, token counts, accuracy per condition. |
| `aime_all_vectors_comparison.json` | Positive steering results on 8 hard mathematical prompts from AIME 2025, all vectors comparison. | 

---

## Notebooks

### 01 — Logit Lens and Vector Extraction
**File:** `notebooks/main_logit_lens_vector_extraction_steering.ipynb`

Runs the logit lens on genuine high-confidence "wait" cases at two positions (before and on the marker), identifies predictable vs implicit correction subtypes, extracts per-case self-reflection vectors using three methods (raw activation, error\_cf counterfactual, Difference of Means), and saves all vectors to NPZ.

**Input:** `classified_corrections_expanded_context.json`, `merged_results.json`, Qwen3-8B (loaded via TransformerLens)  
**Output:** `cases_with_logit_lens_layers.json`, `correction_vectors_all_methods_extended.npz`

**Key parameters:**
```python
MODEL_NAME       = 'Qwen/Qwen3-8B'
MAX_TOKENS       = 256          # context window for activation extraction
PRIMARY_LAYER    = 30           # logit lens peak layer
STEERING_LAYER   = 21           # attribution patching optimal layer
RANDOM_SEED      = 42
TOP_N            = 5            # top-N predictions for logit lens
```

---

### 02 — Linear Probing
**File:** `notebooks/lin_probing.ipynb`

Trains logistic regression and MLP classifiers on residual stream activations to test linear separability of genuine vs stylistic corrections. Produces PCA and t-SNE visualisations.

**Input:** `classified_corrections_expanded_context.json`, `merged_results.json`, Qwen3-8B  
**Output:** plots (accuracy by layer, centroid cosine similarity, PCA, t-SNE)

**Key parameters:**
```python
LAYERS_TO_PROBE  = [16, 20, 22, 24, 26, 28, 30, 32, 35]
PRIMARY_LAYER    = 24           # layer for PCA/t-SNE visualisation
CONFIDENCE       = 'high'       # filter for high-confidence labels only
```

---

### 03 — Steering Experiments
**Files:** `notebooks/positive-steering-aime-3-runs-per-condition.ipynb`, `notebooks/negative-steering_and_vector_extraction.ipynb`

Positive steering (amplifying self-reflection) on AIME 2025 math problems, and negative steering (suppressing stylistic filler) on 20 simple factual/arithmetic prompts. Compares genuine\_hc, all\_wait, dom\_combined, error\_cf and random vectors.

**Input:** `correction_vectors_all_methods_extended.npz`, Qwen3-8B  
**Output:** `neg_stylistic_combined.json`, plots

**Key parameters:**
```python
STEERING_LAYER   = 21
COEFF_POSITIVE   = 40           # positive steering coefficient
COEFF_NEGATIVE   = -70          # negative steering coefficient
MAX_THINK_TOKENS = 800          # for AIME experiments
MAX_NEW_TOKENS   = 500          # for simple prompts
TEMPERATURE      = 0.6          # for AIME (stochastic)
TEMPERATURE_NEG  = 0.0          # for negative steering (greedy)
TOP_P            = 0.95
```

**Steering mechanism:**  
At each generation step, the normalised steering vector is added to the residual stream at `STEERING_LAYER`, last token position only:
```
h[layer][last_token] += coeff × unit_vector
```
Steering is applied during `<think>` only and deactivated after `</think>`.

---

## Vectors

Load vectors:
```python
import numpy as np

vecs = np.load('results/correction_vectors_all_methods_extended.npz')

# Unit-normalise before steering:
def norm(v):
    return v / (np.linalg.norm(v) + 1e-8)

sv_raw       = norm(vecs['mean_raw'])            # shape: (4096,)
sv_error_cf  = norm(vecs['mean_error_cf'])       # shape: (4096,)
sv_dom       = norm(vecs['dom_combined'][21])    # shape: (4096,) — layer 21
sv_stylistic = norm(vecs['mean_stylistic'])      # shape: (4096,)
```

**For negative steering** (suppress stylistic filler):
```python
sv_neg = norm(vecs['mean_stylistic_error_cf'])
# inject with coeff = -70 at layer 21
```

---

## Key Results

| Finding | Value |
|---|---|
| Self-reflection markers detected | 1,013 (99/100 responses) |
| Genuine corrections (high-conf) | 437 |
| Stylistic fillers (high-conf) | 267 |
| Linear probe best accuracy | 0.604 (MLP, layer 32) vs baseline 0.524 |
| Predictable correction subtype | 320/437 (73%) |
| Implicit correction subtype | 117/437 (27%) |
| Cosine sim: predictable vs implicit | 0.9637 |
| Cosine sim: genuine\_hc vs all\_wait | 0.1935 |
| Cosine sim: stylistic\_hc vs all\_wait | 0.9957 |
| Attribution patching peak layer | 21 |
| Negative steering marker reduction | −65.5% (p<0.001) vs baseline |
| Negative steering token reduction | p=0.025 vs random vector |
| Accuracy under negative steering | 100% (20/20 prompts) |
