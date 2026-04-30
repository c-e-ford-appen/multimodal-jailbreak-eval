# =============================================================================
# Bayesian GLMM: Attack Success Rate (ASR) Model
#
# attack_success ~ model * modality * language +
#                  technique * language +
#                  execution * language +
#                  harm_category * language +
#                  (1 | uid) + (1 | participant_id)
#
# Family:  Bernoulli (logit) — binary outcome (0/1)
# Package: brms
#
# Outputs:
#   asr_brms_model.rds           — fitted model object (auto-saved by brms)
#   asr_brms_summary.txt         — full model summary
#   asr_fixed_effects.csv        — fixed effects with ORs and 95% credible intervals
#   asr_random_effects.csv       — random effect SDs
#   asr_predicted_probs.csv      — predicted attack success probabilities
#                                  across key condition combinations
#   plots_asr_trace.png          — convergence trace plots
#   plots_asr_pp_check.png       — posterior predictive check
#   plots_asr_model_lang.png     — model × language predicted probabilities
#   plots_asr_technique_lang.png — technique × language predicted probabilities
#   plots_asr_execution_lang.png — execution × language predicted probabilities
#   plots_asr_harm_lang.png      — harm category × language predicted probabilities
# =============================================================================

# --- 0. Packages -------------------------------------------------------------
pkgs <- c("brms", "dplyr", "tidyr", "ggplot2", "bayesplot", "tidybayes")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE))
    install.packages(p, repos = "https://cloud.r-project.org")
}
library(brms)
library(dplyr)
library(tidyr)
library(ggplot2)
library(bayesplot)
library(tidybayes)

# --- 1. Load data ------------------------------------------------------------
# Set working directory to repo root before running, or adjust paths below.
annotations <- read.csv("data/annotations.csv", stringsAsFactors = FALSE)

cat("Rows loaded:", nrow(annotations), "\n")

# --- 2. Pre-flight checks ----------------------------------------------------
# Verify all factor levels exist in BOTH language conditions.
# Levels present in only one language will produce unidentifiable interactions.
cat("\n--- Checking factor coverage by language ---\n")

check_coverage <- function(var, data) {
  tbl <- table(data[[var]], data$language)
  missing <- which(rowSums(tbl == 0) > 0)
  if (length(missing) > 0) {
    cat("WARNING:", var, "— levels missing in at least one language:\n")
    print(tbl[missing, ])
  } else {
    cat(var, ": all levels present in both languages. OK\n")
  }
}

check_coverage("technique",     annotations)
check_coverage("execution",     annotations)
check_coverage("harm_category", annotations)

# --- 3. Prepare factors ------------------------------------------------------
annotations <- annotations |>
  mutate(
    # Recode attack_success from text labels to binary integer (0/1)
    attack_success = ifelse(attack_success == "Successful", 1L, 0L),

    model         = relevel(factor(model),         ref = "Claude Sonnet 4.5"),
    modality      = relevel(factor(modality),       ref = "multimodal"),
    language      = relevel(factor(language),       ref = "en-US"),
    technique     = factor(technique),   # ref = first alphabetically (adding_noise)
    execution     = factor(execution),   # ref = first alphabetically (embedded_text)
    harm_category = factor(harm_category),

    uid            = factor(uid),
    participant_id = factor(participant_id)
  )

cat("\nFactor reference levels:\n")
cat("  model:         ", levels(annotations$model)[1], "\n")
cat("  modality:      ", levels(annotations$modality)[1], "\n")
cat("  language:      ", levels(annotations$language)[1], "\n")
cat("  technique:     ", levels(annotations$technique)[1], "\n")
cat("  execution:     ", levels(annotations$execution)[1], "\n")
cat("  harm_category: ", levels(annotations$harm_category)[1], "\n")

cat("\nAttack success rate overall:",
    round(mean(annotations$attack_success, na.rm = TRUE), 3), "\n")
cat("By language:\n")
print(tapply(annotations$attack_success, annotations$language, mean, na.rm = TRUE))

# --- 4. Priors ---------------------------------------------------------------
# Weakly informative priors on the log-odds scale.
# N(0,2) gently regularises; Exponential(1) for random effect SDs.

priors_asr <- c(
  prior(normal(0, 2),   class = "b"),
  prior(normal(0, 2),   class = "Intercept"),
  prior(exponential(1), class = "sd")
)

# --- 5. Optional pilot run ---------------------------------------------------
# Uncomment to run a quick 10% subsample fit before committing to the full model.
# This checks that sampling works and gives a rough sense of convergence.
#
# set.seed(42)
# pilot_data  <- slice_sample(annotations, prop = 0.10)
# pilot_model <- brm(
#   formula = attack_success ~ model * modality * language +
#               technique * language + execution * language +
#               harm_category * language +
#               (1 | uid) + (1 | participant_id),
#   family  = bernoulli("logit"),
#   data    = pilot_data,
#   prior   = priors_asr,
#   chains  = 2, cores = 2,
#   iter    = 500, warmup = 250,
#   seed    = 42,
#   file    = "models/asr_brms_pilot"
# )
# cat("Pilot R-hat max:", max(as.data.frame(summary(pilot_model)$fixed)$Rhat), "\n")

# --- 6. Fit model ------------------------------------------------------------
# Runtime estimate: ~1-3 hours with 52K rows, 4 chains × 1000 sampling iterations.
# Model cached to disk — re-runs load instantly if formula/data unchanged.

cat("\n=== Fitting Bayesian ASR model via brms ... ===\n")
cat("Tip: model is cached — re-runs load instantly if formula/data unchanged.\n\n")

asr_brms_model <- brm(
  formula    = attack_success ~ model * modality * language +
                 technique * language +
                 execution * language +
                 harm_category * language +
                 (1 | uid) + (1 | participant_id),
  family     = bernoulli("logit"),
  data       = annotations,
  prior      = priors_asr,
  chains     = 4,
  cores      = 4,           # adjust to your machine
  iter       = 2000,
  warmup     = 1000,
  seed       = 42,
  file       = "models/asr_brms_model",
  file_refit = "on_change"
)

cat("\n=== Sampling complete ===\n\n")

# --- 7. Convergence diagnostics ----------------------------------------------
cat("======================================================\n")
cat("CONVERGENCE DIAGNOSTICS\n")
cat("======================================================\n")

fe_diag <- as.data.frame(summary(asr_brms_model)$fixed)
cat("Max R-hat (fixed effects):  ", max(fe_diag$Rhat, na.rm = TRUE), "\n")
cat("Min Bulk ESS (fixed effects):", min(fe_diag$Bulk_ESS, na.rm = TRUE), "\n")
cat("Min Tail ESS (fixed effects):", min(fe_diag$Tail_ESS, na.rm = TRUE), "\n")

if (max(fe_diag$Rhat, na.rm = TRUE) > 1.05) {
  cat("\nWARNING: R-hat > 1.05 detected. Consider increasing iter to 4000.\n")
} else {
  cat("Convergence looks good (all R-hat <= 1.05).\n")
}

# Trace plots
png("plots_asr_trace.png", width = 1600, height = 1000)
mcmc_trace(asr_brms_model, pars = vars(starts_with("b_"))) +
  ggtitle("ASR model — trace plots (should look like fuzzy caterpillars)")
dev.off()
cat("Saved: plots_asr_trace.png\n")

# Posterior predictive check
png("plots_asr_pp_check.png", width = 900, height = 500)
pp_check(asr_brms_model, type = "bars_grouped", group = "language", ndraws = 100) +
  ggtitle("Posterior predictive check — observed (circles) vs. simulated (bars)")
dev.off()
cat("Saved: plots_asr_pp_check.png\n")

# --- 8. Model summary --------------------------------------------------------
cat("\n======================================================\n")
cat("MODEL SUMMARY\n")
cat("======================================================\n")
print(summary(asr_brms_model))

sink("asr_brms_summary.txt")
cat("Bayesian GLMM (Bernoulli logit) via brms\n")
cat("Formula: attack_success ~ model * modality * language +\n")
cat("         technique * language + execution * language +\n")
cat("         harm_category * language +\n")
cat("         (1 | uid) + (1 | participant_id)\n\n")
print(summary(asr_brms_model))
sink()
cat("Saved: asr_brms_summary.txt\n")

# --- 9. Fixed effects table --------------------------------------------------
fe <- fixef(asr_brms_model) |>
  as.data.frame() |>
  tibble::rownames_to_column("term") |>
  rename(estimate    = Estimate,
         std_error   = Est.Error,
         ci_lower_95 = Q2.5,
         ci_upper_95 = Q97.5)

fe <- fe |>
  mutate(
    odds_ratio     = exp(estimate),
    or_ci_lower_95 = exp(ci_lower_95),
    or_ci_upper_95 = exp(ci_upper_95),
    credible       = ifelse(ci_lower_95 > 0 | ci_upper_95 < 0, "YES", "no"),
    direction      = case_when(
      credible == "YES" & estimate > 0 ~ "increases ASR",
      credible == "YES" & estimate < 0 ~ "decreases ASR",
      TRUE ~ "no credible effect"
    )
  )

write.csv(fe, "asr_fixed_effects.csv", row.names = FALSE)
cat("Saved: asr_fixed_effects.csv\n")

# --- 10. Random effects ------------------------------------------------------
re <- VarCorr(asr_brms_model)
re_df <- data.frame(
  group       = names(re),
  sd_estimate = sapply(re, function(x) x$sd[1, "Estimate"]),
  sd_ci_lower = sapply(re, function(x) x$sd[1, "Q2.5"]),
  sd_ci_upper = sapply(re, function(x) x$sd[1, "Q97.5"])
)
write.csv(re_df, "asr_random_effects.csv", row.names = FALSE)
cat("Saved: asr_random_effects.csv\n")

# --- 11. Predicted attack success probabilities ------------------------------
# Marginal predicted probabilities for key condition combinations.
# These are more interpretable than log-odds.

pred_probs <- function(newdata, model) {
  epred <- posterior_epred(model, newdata = newdata,
                           re_formula = NA,   # marginalise over random effects
                           ndraws = 1000)
  data.frame(
    newdata,
    prob_mean     = apply(epred, 2, mean),
    prob_ci_lower = apply(epred, 2, quantile, 0.025),
    prob_ci_upper = apply(epred, 2, quantile, 0.975)
  )
}

# Model × language (reference modality, technique, execution, harm_category)
nd_model_lang <- expand.grid(
  model         = levels(annotations$model),
  language      = levels(annotations$language),
  modality      = levels(annotations$modality)[1],
  technique     = levels(annotations$technique)[1],
  execution     = levels(annotations$execution)[1],
  harm_category = levels(annotations$harm_category)[1]
)
pp_model_lang <- pred_probs(nd_model_lang, asr_brms_model)

# Technique × language
nd_tech_lang <- expand.grid(
  technique     = levels(annotations$technique),
  language      = levels(annotations$language),
  model         = levels(annotations$model)[1],
  modality      = levels(annotations$modality)[1],
  execution     = levels(annotations$execution)[1],
  harm_category = levels(annotations$harm_category)[1]
)
pp_tech_lang <- pred_probs(nd_tech_lang, asr_brms_model)

# Execution × language
nd_exec_lang <- expand.grid(
  execution     = levels(annotations$execution),
  language      = levels(annotations$language),
  model         = levels(annotations$model)[1],
  modality      = levels(annotations$modality)[1],
  technique     = levels(annotations$technique)[1],
  harm_category = levels(annotations$harm_category)[1]
)
pp_exec_lang <- pred_probs(nd_exec_lang, asr_brms_model)

# Harm category × language
nd_harm_lang <- expand.grid(
  harm_category = levels(annotations$harm_category),
  language      = levels(annotations$language),
  model         = levels(annotations$model)[1],
  modality      = levels(annotations$modality)[1],
  technique     = levels(annotations$technique)[1],
  execution     = levels(annotations$execution)[1]
)
pp_harm_lang <- pred_probs(nd_harm_lang, asr_brms_model)

pp_all <- bind_rows(
  pp_model_lang |> mutate(comparison = "model_x_language"),
  pp_tech_lang  |> mutate(comparison = "technique_x_language"),
  pp_exec_lang  |> mutate(comparison = "execution_x_language"),
  pp_harm_lang  |> mutate(comparison = "harm_category_x_language")
)
write.csv(pp_all, "asr_predicted_probs.csv", row.names = FALSE)
cat("Saved: asr_predicted_probs.csv\n")

# --- 12. Plots ---------------------------------------------------------------
plot_pred <- function(df, x_var, title_str, filename) {
  png(filename, width = 900, height = 550)
  p <- ggplot(df, aes(x = reorder(.data[[x_var]], prob_mean),
                      y = prob_mean,
                      colour = language, group = language)) +
    geom_point(size = 3.5, position = position_dodge(0.4)) +
    geom_errorbar(aes(ymin = prob_ci_lower, ymax = prob_ci_upper),
                  width = 0.2, position = position_dodge(0.4)) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                       limits = c(0, NA)) +
    scale_colour_manual(values = c("en-US" = "#2E5FA3", "es-MX" = "#C0392B")) +
    coord_flip() +
    labs(x = NULL, y = "Predicted attack success probability",
         colour = "Language group", title = title_str,
         subtitle = "Points = posterior mean; bars = 95% credible interval") +
    theme_bw(base_size = 13)
  print(p)
  dev.off()
  cat("Saved:", filename, "\n")
}

plot_pred(pp_model_lang, "model",        "Attack success probability by model and language",
          "plots_asr_model_lang.png")
plot_pred(pp_tech_lang,  "technique",    "Attack success probability by technique and language",
          "plots_asr_technique_lang.png")
plot_pred(pp_exec_lang,  "execution",    "Attack success probability by execution type and language",
          "plots_asr_execution_lang.png")
plot_pred(pp_harm_lang,  "harm_category","Attack success probability by harm category and language",
          "plots_asr_harm_lang.png")

# --- 13. Console summary of credible language interactions -------------------
cat("\n======================================================\n")
cat("CREDIBLE LANGUAGE INTERACTIONS (CI excludes 0)\n")
cat("======================================================\n")

lang_terms <- fe |>
  filter(grepl("language", term, ignore.case = TRUE), credible == "YES") |>
  select(term, estimate, ci_lower_95, ci_upper_95, odds_ratio,
         or_ci_lower_95, or_ci_upper_95, direction) |>
  arrange(desc(abs(estimate)))

print(lang_terms)

cat("\n=== All outputs saved. ===\n")
cat("Next steps:\n")
cat("  - Check plots_asr_trace.png for convergence\n")
cat("  - Check plots_asr_pp_check.png for model fit\n")
cat("  - Run 04_pairwise_asr.R for pairwise comparisons\n")
