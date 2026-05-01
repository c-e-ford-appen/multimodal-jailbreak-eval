# Multimodal Jailbreak Evaluation: US English and Mexican Spanish

This repository contains the data, analysis scripts, and pre-fitted model objects accompanying the paper **Same Model, Different Weakness: How Language and Modality Reshape the Jailbreak Attack Surface in Frontier MLLMs**, submitted for double-bind review to NeurIPS 2026.

## Overview

This study evaluates the susceptibility of four frontier multimodal large language models to jailbreak attacks across two language groups (en-US and es-MX) and two modality conditions (multimodal and text-only). Human annotators rated model responses on a 5-point harm scale; attack success was also coded as a binary outcome. Mixed-effects models were used to estimate the effects of model, modality, language, attack technique, execution method, and harm category.

## Repository Structure

```
repo/
├── README.md
├── data/
│   └── annotations.csv        # Full annotation dataset (52,272 rows)
├── analysis/
│   ├── 00_frequentist_clmm.R  # Supplementary: frequentist CLMM via ordinal
│   ├── 01_brms_harm_ratings.R # Primary: Bayesian CLMM for harm ratings
│   ├── 02_brms_asr.R          # Primary: Bayesian GLMM for attack success rate
│   ├── 03_pairwise_harm_ratings.R  # Pairwise contrasts (harm ratings)
│   ├── 04_pairwise_asr.R           # Pairwise contrasts (ASR)
│   └── 05_irr.R               # Inter-rater reliability (Krippendorff's α, Gwet's AC2)
└── models/
    ├── brms_harm_model.rds    # Fitted Bayesian CLMM (harm ratings)
    └── asr_brms_model.rds     # Fitted Bayesian GLMM (attack success rate)
```

## Data

### `data/annotations.csv`

Each row is a single annotation (one annotator rating one model response to one prompt).

| Column | Description |
|---|---|
| `uid` | Unique prompt scenario ID (1–363) |
| `model` | Model name: `Claude Sonnet 4.5`, `GPT-5`, `Pixtral Large`, `Qwen Omni` |
| `modality` | `multimodal` or `text-only` |
| `language` | `en-US` or `es-MX` |
| `technique` | Jailbreak technique applied to the prompt |
| `execution` | Execution method (how the jailbreak was embedded) |
| `harm_category` | Harm domain of the prompt scenario |
| `harm_rating` | Ordered harm rating: `Not at all harmful` → `Extremely harmful` |
| `rating_default` | Numeric harm rating (integer; used for IRR calculations) |
| `attack_success` | `Successful` or `Not Successful` |
| `participant_id` | Anonymised rater ID (e.g., `enUS_001`, `esMX_003`) |
| `participant_gender` | Rater gender (`M` / `F` / other) |
| `participant_age` | Rater age bracket (e.g., `18-34`, `35-44`, `45+`) |
| `item_id` | Composite key: `uid_model_modality_language` |
| `harm_numeric` | Numeric version of `harm_rating` |
| `harm_status` | Text version of `harm_rating` |
| `defaultresp` | Whether the model's response was flagged as a default refusal |

**Dataset dimensions:** 52,272 annotations — 363 prompts × 4 models × 2 modalities × 2 language groups × 9 raters per language group.

**Participant IDs:** 9 en-US raters (`enUS_001/002/003/006/007/008/010/011/012`, selected from 12 by matched demographics) and 9 es-MX raters (`esMX_001`–`esMX_009`).

## Analysis Scripts

All scripts assume the working directory is set to the **repo root**. Run them in numerical order, or load the pre-fitted models from `models/` and start from step 03.

### `00_frequentist_clmm.R` — Supplementary

Fits a frequentist cumulative link mixed model (CLMM) using the `ordinal` package. Produces coefficient estimates with standard errors and p-values. Included as a supplementary check; inference is based on the Bayesian models in scripts 01–02.

**Outputs:** `clmm_summary.txt`, `clmm_coefficients.csv`, `clmm_random_effects.csv`, `clmm_model.rds`

### `01_brms_harm_ratings.R` — Primary harm model

Fits a Bayesian cumulative logit mixed model (CLMM) via `brms` for the 5-point harm rating outcome. Includes crossed random effects for prompt (`uid`) and rater (`participant_id`). The model is cached to `models/brms_harm_model.rds` and will reload from cache on subsequent runs.

**Formula:**
```
harm_rating ~ model * modality * language +
              participant_gender + participant_age +
              (1 | uid) + (1 | participant_id)
```

**Runtime:** ~1–3 hours on a modern laptop (4 cores). The pre-fitted model in `models/` can be loaded directly to skip refitting.

**Outputs:** `brms_summary.txt`, `brms_fixed_effects.csv`, `brms_thresholds.csv`, `brms_random_effects.csv`, convergence and diagnostic plots.

### `02_brms_asr.R` — Attack success rate model

Fits a Bayesian Bernoulli logistic mixed model (GLMM) via `brms` for the binary attack success outcome. Includes technique, execution method, and harm category as predictors with language interaction terms.

**Formula:**
```
attack_success ~ model * modality * language +
                 technique * language +
                 execution * language +
                 harm_category * language +
                 (1 | uid) + (1 | participant_id)
```

**Runtime:** ~1–3 hours. The pre-fitted model in `models/` can be loaded directly.

**Outputs:** `asr_brms_summary.txt`, `asr_fixed_effects.csv`, `asr_random_effects.csv`, `asr_predicted_probs.csv`, diagnostic and predicted probability plots.

### `03_pairwise_harm_ratings.R` — Pairwise contrasts (harm ratings)

Computes all pairwise comparisons from the harm rating model using `emmeans` on the latent (log-odds) scale. Contrasts include model pairs overall, by modality, by language, and by language × modality, plus age group comparisons and modality/language contrasts within each model.

**Requires:** `models/brms_harm_model.rds`

**Outputs:** 9 CSV files of pairwise contrasts, `pairwise_summary.txt`

### `04_pairwise_asr.R` — Pairwise contrasts (ASR)

Computes pairwise comparisons from the ASR model, including model pairs, technique pairs, execution pairs, harm category pairs, and language contrasts within each predictor level.

**Requires:** `models/asr_brms_model.rds`

**Outputs:** 14 CSV files of pairwise contrasts, `pairwise_asr_summary.txt`

### `05_irr.R` — Inter-rater reliability

Computes inter-rater reliability using two measures:

- **Gwet's AC2** (quadratic weights, k = 6 categories) — primary measure, robust to the prevalence paradox that can artificially deflate Krippendorff's alpha when ratings are skewed.
- **Krippendorff's alpha** (ordinal) — secondary measure, reported for comparison with prior work.

Computes both overall and disaggregated by modality, language, model, and model × language. Also generates disagreement heatmaps (confusion matrices showing the distribution of pairwise rating discrepancies).

**Outputs:** CSV files of alpha values, `irr_gwet_summary.txt`, confusion matrix PNGs.

## Reproducing the Analysis

### Requirements

- R ≥ 4.2
- Packages: `brms`, `ordinal`, `emmeans`, `dplyr`, `tidyr`, `ggplot2`, `bayesplot`, `tidybayes`, `irr`, `irrCAC`

Install all at once:
```r
install.packages(c("brms", "ordinal", "emmeans", "dplyr", "tidyr",
                   "ggplot2", "bayesplot", "tidybayes", "irr", "irrCAC"),
                 repos = "https://cloud.r-project.org")
```

`brms` requires a working Stan installation. See [mc-stan.org/cmdstanr](https://mc-stan.org/cmdstanr/) for setup instructions.

### Running with pre-fitted models

To skip the long model-fitting step (~1–3 hours per model), load the `.rds` files directly in scripts 03 and 04:

```r
# In 03_pairwise_harm_ratings.R — already set correctly
m <- readRDS("models/brms_harm_model.rds")

# In 04_pairwise_asr.R — already set correctly
m <- readRDS("models/asr_brms_model.rds")
```

### Running from scratch

Set your working directory to the repo root and run scripts in order:

```r
setwd("/path/to/repo")
source("analysis/05_irr.R")               # IRR (no model fitting required)
source("analysis/01_brms_harm_ratings.R") # Fits or loads harm model (~1–3 hrs)
source("analysis/02_brms_asr.R")          # Fits or loads ASR model (~1–3 hrs)
source("analysis/03_pairwise_harm_ratings.R")
source("analysis/04_pairwise_asr.R")
```

## Reference Levels

All models use the following reference levels for interpretation of coefficients:

| Predictor | Reference level |
|---|---|
| `model` | Claude Sonnet 4.5 |
| `modality` | multimodal |
| `language` | en-US |
| `technique` | adding_noise |
| `execution` | embedded_text |
| `harm_category` | disinformation |
| `participant_gender` | M |
| `participant_age` | 18–34 |

## Prompts

The 363 jailbreak prompt scenarios used in this study are not included in this repository. Beyond dual-use concerns inherent to publishing jailbreak stimuli, releasing the prompt set would risk benchmark contamination: if prompts enter post-training pipelines for future model generations, the benchmark loses its validity as a tool for longitudinal safety comparison. Preserving the integrity of the benchmark for future generational evaluation is a primary motivation for withholding the original prompt data. Researchers seeking access for legitimate evaluation purposes may contact the authors directly.

## Licence

Data and code are provided for reproducibility purposes under the terms of the paper's review process. Please cite the paper if you use this material.
