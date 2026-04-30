# =============================================================================
# Bayesian CLMM: Harm Rating Model
#
# harm_rating ~ model * modality * language +
#               participant_gender + participant_age +
#               (1 | uid) + (1 | participant_id)
#
# Family:  Cumulative (logit) — Bayesian ordered regression
# Package: brms (Bayesian Regression Models using Stan)
#
# Advantages over frequentist CLMM (see 00_frequentist_clmm.R):
#   - Full posterior distributions instead of point estimates + p-values
#   - Rich diagnostics (R-hat, ESS, posterior predictive checks)
#   - Model cached to disk — re-running loads instantly if unchanged
#
# Outputs:
#   brms_harm_model.rds        — fitted model object (auto-saved by brms)
#   brms_summary.txt           — full model summary
#   brms_fixed_effects.csv     — fixed effects with ORs and 95% credible intervals
#   brms_thresholds.csv        — ordinal threshold intercepts
#   brms_random_effects.csv    — random effect SDs
#   plots_trace.png            — convergence trace plots
#   plots_pp_check.png         — posterior predictive check
#   plots_fixed_effects.png    — forest plot of main effects
#   plots_conditional_model.png — predicted probabilities by model
# =============================================================================

# --- 0. Packages -------------------------------------------------------------
pkgs <- c("brms", "dplyr", "tidyr", "ggplot2", "bayesplot")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE))
    install.packages(p, repos = "https://cloud.r-project.org")
}
library(brms)
library(dplyr)
library(ggplot2)
library(bayesplot)

# --- 1. Load data ------------------------------------------------------------
# Set working directory to repo root before running, or adjust paths below.
annotations <- read.csv("data/annotations.csv", stringsAsFactors = FALSE)

cat("Rows loaded:", nrow(annotations), "\n")

# --- 2. Prepare factors ------------------------------------------------------
harm_levels <- c(
  "Not at all harmful",
  "Mildly harmful",
  "Moderately harmful",
  "Very harmful",
  "Extremely harmful"
)

annotations <- annotations |>
  mutate(
    harm_rating = factor(harm_rating, levels = harm_levels, ordered = TRUE),

    # Reference levels (first level = reference in brms)
    model              = relevel(factor(model),              ref = "Claude Sonnet 4.5"),
    modality           = relevel(factor(modality),           ref = "multimodal"),
    language           = relevel(factor(language),           ref = "en-US"),
    participant_gender = relevel(factor(participant_gender), ref = "M"),
    participant_age    = relevel(factor(participant_age),    ref = "18-34"),

    uid            = factor(uid),
    participant_id = factor(participant_id)
  )

cat("harm_rating levels:", paste(levels(annotations$harm_rating), collapse = " < "), "\n")

# --- 3. Set priors -----------------------------------------------------------
# Weakly informative priors on the log-odds scale.
# Coefficients > 2 are very large effects; N(0,2) gently regularises.
# Exponential(1) for random effect SDs keeps them from blowing up.

priors <- c(
  prior(normal(0, 2),   class = "b"),          # fixed effects
  prior(normal(0, 2),   class = "Intercept"),  # threshold intercepts
  prior(exponential(1), class = "sd")          # random effect SDs
)

# --- 4. Fit Bayesian CLMM ----------------------------------------------------
# NOTE: First run compiles Stan code (~1-2 min). Subsequent runs load from
# cache instantly (brms checks if formula/data have changed via file_refit).
#
# Runtime estimate: ~1-3 hours on a modern laptop with 4 cores.
# Delete 'models/brms_harm_model.rds' to force a full refit.

cat("\n=== Fitting Bayesian CLMM via brms ... ===\n")
cat("Tip: First run compiles Stan (~1-2 min), then sampling begins.\n")
cat("     Model is cached to 'models/brms_harm_model' — re-runs load instantly.\n\n")

brms_harm_model <- brm(
  formula    = harm_rating ~ model * modality * language +
                 participant_gender + participant_age +
                 (1 | uid) + (1 | participant_id),
  family     = cumulative("logit"),
  data       = annotations,
  prior      = priors,
  chains     = 4,           # 4 independent chains (run in parallel)
  cores      = 4,           # adjust to your machine
  iter       = 2000,        # 1000 warmup + 1000 sampling per chain
  warmup     = 1000,
  seed       = 42,
  file       = "models/brms_harm_model",   # caches fitted model
  file_refit = "on_change"                 # only refits if formula/data change
)

cat("\n=== Sampling complete ===\n\n")

# --- 5. Convergence diagnostics ----------------------------------------------
cat("======================================================\n")
cat("CONVERGENCE DIAGNOSTICS\n")
cat("======================================================\n")

# R-hat < 1.01 and ESS > 400 indicate good convergence
diag_summary <- as.data.frame(summary(brms_harm_model)$fixed)
cat("\nMax R-hat (fixed effects):", max(diag_summary$Rhat, na.rm = TRUE), "\n")
cat("Min Bulk ESS:", min(diag_summary$Bulk_ESS, na.rm = TRUE), "\n")
cat("Min Tail ESS:", min(diag_summary$Tail_ESS, na.rm = TRUE), "\n")

if (max(diag_summary$Rhat, na.rm = TRUE) > 1.05) {
  cat("\nWARNING: R-hat > 1.05 detected. Consider increasing iter to 4000.\n")
} else {
  cat("Convergence looks good (all R-hat <= 1.05).\n")
}

# Trace plots — should look like fuzzy caterpillars with good mixing
png("plots_trace.png", width = 1400, height = 900)
mcmc_trace(brms_harm_model, pars = vars(starts_with("b_"))) +
  ggtitle("Trace plots — should look like 'fuzzy caterpillars'")
dev.off()
cat("Saved: plots_trace.png\n")

# --- 6. Posterior predictive check ------------------------------------------
# Does the model reproduce the observed distribution of harm_rating?
png("plots_pp_check.png", width = 800, height = 500)
pp_check(brms_harm_model, type = "bars", ndraws = 100) +
  ggtitle("Posterior predictive check — bars should overlap circles")
dev.off()
cat("Saved: plots_pp_check.png\n")

# --- 7. Full model summary ---------------------------------------------------
cat("\n======================================================\n")
cat("MODEL SUMMARY\n")
cat("======================================================\n")
print(summary(brms_harm_model))

sink("brms_summary.txt")
cat("Bayesian CLMM via brms\n")
cat("Formula: harm_rating ~ model * modality * language + participant_gender + participant_age\n")
cat("         + (1 | uid) + (1 | participant_id)\n")
cat("Family:  cumulative(logit)\n\n")
print(summary(brms_harm_model))
sink()
cat("Saved: brms_summary.txt\n")

# --- 8. Extract and save fixed effects ---------------------------------------
# Estimate = posterior mean (log-odds)
# Est.Error = posterior SD
# Q2.5 / Q97.5 = 95% credible interval
# If CI excludes 0 → credible evidence of an effect

fe <- fixef(brms_harm_model) |>
  as.data.frame() |>
  tibble::rownames_to_column("term") |>
  rename(estimate    = Estimate,
         std_error   = Est.Error,
         ci_lower_95 = Q2.5,
         ci_upper_95 = Q97.5) |>
  filter(!grepl("^Intercept\\[", term))   # thresholds saved separately below

# Exponentiate to cumulative odds ratios
fe$odds_ratio     <- exp(fe$estimate)
fe$or_ci_lower_95 <- exp(fe$ci_lower_95)
fe$or_ci_upper_95 <- exp(fe$ci_upper_95)
fe$credible       <- ifelse(fe$ci_lower_95 > 0 | fe$ci_upper_95 < 0, "YES", "no")

write.csv(fe, "brms_fixed_effects.csv", row.names = FALSE)
cat("Saved: brms_fixed_effects.csv\n")

# Threshold intercepts (cutpoints between harm levels)
thresholds <- fixef(brms_harm_model) |>
  as.data.frame() |>
  tibble::rownames_to_column("term") |>
  filter(grepl("^Intercept\\[", term))
write.csv(thresholds, "brms_thresholds.csv", row.names = FALSE)
cat("Saved: brms_thresholds.csv\n")

# --- 9. Random effects -------------------------------------------------------
re <- VarCorr(brms_harm_model)
re_df <- data.frame(
  group       = names(re),
  sd_estimate = sapply(re, function(x) x$sd[1, "Estimate"]),
  sd_ci_lower = sapply(re, function(x) x$sd[1, "Q2.5"]),
  sd_ci_upper = sapply(re, function(x) x$sd[1, "Q97.5"])
)
write.csv(re_df, "brms_random_effects.csv", row.names = FALSE)
cat("Saved: brms_random_effects.csv\n")

# --- 10. Forest plot of main effects -----------------------------------------
fe_plot <- fe |>
  filter(!grepl(":", term)) |>   # main effects only (remove interactions for clarity)
  mutate(term = gsub("^(model|modality|language|participant_gender|participant_age)", "", term))

png("plots_fixed_effects.png", width = 900, height = 600)
ggplot(fe_plot, aes(x = odds_ratio, y = reorder(term, odds_ratio))) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = or_ci_lower_95, xmax = or_ci_upper_95), height = 0.2) +
  scale_x_log10() +
  labs(x = "Cumulative Odds Ratio (log scale)", y = NULL,
       title = "Fixed Effects — Main Effects Only",
       subtitle = "Points = posterior mean OR; bars = 95% credible interval\nOR > 1: higher harm ratings vs. reference; OR < 1: lower harm ratings") +
  theme_bw(base_size = 13)
dev.off()
cat("Saved: plots_fixed_effects.png\n")

# --- 11. Conditional effects (predicted probabilities by model) ---------------
ce_model <- conditional_effects(brms_harm_model, effects = "model",
                                categorical = TRUE)
png("plots_conditional_model.png", width = 1000, height = 600)
plot(ce_model)[[1]] +
  labs(title = "Predicted probability of each harm rating by Model",
       x = "Model", y = "Predicted probability", fill = "Harm rating") +
  theme_bw(base_size = 12)
dev.off()
cat("Saved: plots_conditional_model.png\n")

cat("\n=== All outputs saved. See brms_summary.txt for full results. ===\n")
cat("Next: run 03_pairwise_harm_ratings.R for pairwise model comparisons.\n")
