# =============================================================================
# Pairwise Comparisons: Attack Success Rate (ASR) Model
#
# Emmeans contrasts from the fitted Bayesian Bernoulli GLMM (brms).
# All comparisons use mode = "latent" (log-odds scale).
# Odds ratios are computed by exponentiating.
# Credible intervals are 95% highest posterior density intervals (HPD).
#
# Requires: models/asr_brms_model.rds (fitted by 02_brms_asr.R)
# Packages: brms, emmeans
#
# Outputs:
#   pairwise_asr_models_overall.csv        — all model pairs, marginalised
#   pairwise_asr_models_by_language.csv    — model pairs × language
#   pairwise_asr_models_by_modality.csv    — model pairs × modality
#   pairwise_asr_models_by_lang_mod.csv    — model pairs × language × modality
#   pairwise_asr_technique_overall.csv     — technique pairs, marginalised
#   pairwise_asr_technique_by_language.csv — technique pairs × language
#   pairwise_asr_execution_overall.csv     — execution pairs, marginalised
#   pairwise_asr_execution_by_language.csv — execution pairs × language
#   pairwise_asr_harm_overall.csv          — harm category pairs, marginalised
#   pairwise_asr_harm_by_language.csv      — harm category pairs × language
#   pairwise_asr_language_by_model.csv     — language contrast within each model
#   pairwise_asr_language_by_technique.csv — language contrast within each technique
#   pairwise_asr_language_by_execution.csv — language contrast within each execution
#   pairwise_asr_language_by_harm.csv      — language contrast within each harm category
#   pairwise_asr_summary.txt               — formatted summary of all contrasts
# =============================================================================

library(brms)
library(emmeans)

# --- 1. Load model -----------------------------------------------------------
cat("Loading ASR model...\n")
m <- readRDS("models/asr_brms_model.rds")
cat("Model loaded.\n\n")

# --- 2. Helper ---------------------------------------------------------------
tidy_pairs <- function(pairs_obj) {
  df <- as.data.frame(summary(pairs_obj, infer = TRUE))

  if ("lower.HPD" %in% names(df)) names(df)[names(df) == "lower.HPD"] <- "ci_lower_95"
  if ("upper.HPD" %in% names(df)) names(df)[names(df) == "upper.HPD"] <- "ci_upper_95"
  if ("estimate"  %in% names(df)) names(df)[names(df) == "estimate"]  <- "estimate_logodds"

  df$odds_ratio     <- exp(df$estimate_logodds)
  df$or_ci_lower_95 <- exp(df$ci_lower_95)
  df$or_ci_upper_95 <- exp(df$ci_upper_95)
  df$credible       <- ifelse(df$ci_lower_95 > 0 | df$ci_upper_95 < 0, "YES", "no")
  df
}

run_pairs <- function(formula_str, data_model, file_out, label = "") {
  cat("Computing:", label, "\n")
  emm <- emmeans(data_model, as.formula(formula_str), mode = "latent")
  pw  <- pairs(emm, reverse = TRUE)
  df  <- tidy_pairs(pw)
  write.csv(df, file_out, row.names = FALSE)
  cat("  Saved:", file_out, "\n")
  invisible(list(emm = emm, pairs = pw, df = df))
}

# --- 3. Model pairwise comparisons -------------------------------------------
r_mod_overall <- run_pairs("~ model",
                            m, "pairwise_asr_models_overall.csv",
                            "model pairs (overall)")

r_mod_lang    <- run_pairs("~ model | language",
                            m, "pairwise_asr_models_by_language.csv",
                            "model pairs × language")

r_mod_modal   <- run_pairs("~ model | modality",
                            m, "pairwise_asr_models_by_modality.csv",
                            "model pairs × modality")

r_mod_langmod <- run_pairs("~ model | language * modality",
                            m, "pairwise_asr_models_by_lang_mod.csv",
                            "model pairs × language × modality")

# --- 4. Technique pairwise comparisons ---------------------------------------
r_tech_overall <- run_pairs("~ technique",
                              m, "pairwise_asr_technique_overall.csv",
                              "technique pairs (overall)")

r_tech_lang    <- run_pairs("~ technique | language",
                              m, "pairwise_asr_technique_by_language.csv",
                              "technique pairs × language")

# --- 5. Execution pairwise comparisons ---------------------------------------
r_exec_overall <- run_pairs("~ execution",
                              m, "pairwise_asr_execution_overall.csv",
                              "execution pairs (overall)")

r_exec_lang    <- run_pairs("~ execution | language",
                              m, "pairwise_asr_execution_by_language.csv",
                              "execution pairs × language")

# --- 6. Harm category pairwise comparisons -----------------------------------
r_harm_overall <- run_pairs("~ harm_category",
                              m, "pairwise_asr_harm_overall.csv",
                              "harm category pairs (overall)")

r_harm_lang    <- run_pairs("~ harm_category | language",
                              m, "pairwise_asr_harm_by_language.csv",
                              "harm category pairs × language")

# --- 7. Language contrasts within each predictor -----------------------------
# These directly answer: "does language group change ASR for each level of X?"

r_lang_model <- run_pairs("~ language | model",
                            m, "pairwise_asr_language_by_model.csv",
                            "language contrast within each model")

r_lang_tech  <- run_pairs("~ language | technique",
                            m, "pairwise_asr_language_by_technique.csv",
                            "language contrast within each technique")

r_lang_exec  <- run_pairs("~ language | execution",
                            m, "pairwise_asr_language_by_execution.csv",
                            "language contrast within each execution type")

r_lang_harm  <- run_pairs("~ language | harm_category",
                            m, "pairwise_asr_language_by_harm.csv",
                            "language contrast within each harm category")

# --- 8. Save full formatted summary ------------------------------------------
sink("pairwise_asr_summary.txt")

cat("=========================================================\n")
cat("PAIRWISE COMPARISONS — Bayesian ASR model\n")
cat("Family: Bernoulli(logit) | Latent log-odds scale\n")
cat("OR = exp(estimate) | Credible = CI excludes 0\n")
cat("=========================================================\n\n")

sections <- list(
  "MODEL PAIRS — overall"                    = r_mod_overall$pairs,
  "MODEL PAIRS — by language"                = r_mod_lang$pairs,
  "MODEL PAIRS — by modality"                = r_mod_modal$pairs,
  "MODEL PAIRS — by language × modality"     = r_mod_langmod$pairs,
  "TECHNIQUE PAIRS — overall"                = r_tech_overall$pairs,
  "TECHNIQUE PAIRS — by language"            = r_tech_lang$pairs,
  "EXECUTION PAIRS — overall"                = r_exec_overall$pairs,
  "EXECUTION PAIRS — by language"            = r_exec_lang$pairs,
  "HARM CATEGORY PAIRS — overall"            = r_harm_overall$pairs,
  "HARM CATEGORY PAIRS — by language"        = r_harm_lang$pairs,
  "LANGUAGE CONTRAST — within each model"    = r_lang_model$pairs,
  "LANGUAGE CONTRAST — within each technique"= r_lang_tech$pairs,
  "LANGUAGE CONTRAST — within each execution"= r_lang_exec$pairs,
  "LANGUAGE CONTRAST — within each harm cat."= r_lang_harm$pairs
)

for (title in names(sections)) {
  cat("\n---", title, "---\n")
  print(summary(sections[[title]], infer = TRUE))
}

sink()
cat("\nSaved: pairwise_asr_summary.txt\n")

# --- 9. Quick credible-effects console summary -------------------------------
cat("\n=========================================================\n")
cat("CREDIBLE LANGUAGE CONTRASTS AT A GLANCE\n")
cat("(es-MX vs. en-US within each level — CI excludes 0)\n")
cat("=========================================================\n")

print_credible <- function(df, group_col, label) {
  credible_rows <- df[df$credible == "YES", ]
  if (nrow(credible_rows) == 0) {
    cat("\n", label, ": no credible language contrasts\n")
  } else {
    cat("\n", label, ":\n", sep = "")
    cols <- intersect(c(group_col, "contrast", "estimate_logodds",
                        "ci_lower_95", "ci_upper_95", "odds_ratio", "credible"),
                      names(credible_rows))
    print(credible_rows[, cols])
  }
}

print_credible(r_lang_model$df, "model",         "Model")
print_credible(r_lang_tech$df,  "technique",     "Technique")
print_credible(r_lang_exec$df,  "execution",     "Execution")
print_credible(r_lang_harm$df,  "harm_category", "Harm category")

cat("\n=== Done. All pairwise files saved. ===\n")
