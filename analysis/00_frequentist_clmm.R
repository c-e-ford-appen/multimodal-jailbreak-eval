# =============================================================================
# Frequentist CLMM: Harm Rating Model (Supplementary)
#
# harm_rating ~ model * modality * language +
#               participant_gender + participant_age +
#               (1 | uid) + (1 | participant_id)
#
# Package: ordinal
#
# This script provides the frequentist equivalent of 01_brms_harm_ratings.R
# using the ordinal package's clmm() function. Results are included as a
# supplementary check; the primary analysis uses the Bayesian model.
#
# Outputs:
#   clmm_summary.txt           — full model summary
#   clmm_coefficients.csv      — fixed-effect coefficients with ORs
#   clmm_random_effects.csv    — random effect variances and SDs
#   clmm_model.rds             — saved model object
# =============================================================================

# --- 0. Packages -------------------------------------------------------------
if (!requireNamespace("ordinal", quietly = TRUE))
  install.packages("ordinal", repos = "https://cloud.r-project.org")
library(ordinal)

# --- 1. Load data ------------------------------------------------------------
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

annotations$harm_rating <- factor(
  annotations$harm_rating,
  levels  = harm_levels,
  ordered = TRUE
)

annotations$model <- relevel(factor(annotations$model),
                              ref = "Claude Sonnet 4.5")
annotations$modality <- relevel(factor(annotations$modality),
                                ref = "multimodal")
annotations$language <- relevel(factor(annotations$language),
                                ref = "en-US")
annotations$participant_gender <- relevel(factor(annotations$participant_gender),
                                          ref = "M")
annotations$participant_age <- relevel(factor(annotations$participant_age),
                                       ref = "18-34")

annotations$uid            <- factor(annotations$uid)
annotations$participant_id <- factor(annotations$participant_id)

cat("\nharm_rating level order:\n")
print(levels(annotations$harm_rating))
cat("\nModel factor levels (reference first):\n")
cat("  model:             ", levels(annotations$model), "\n")
cat("  modality:          ", levels(annotations$modality), "\n")
cat("  language:          ", levels(annotations$language), "\n")
cat("  participant_gender:", levels(annotations$participant_gender), "\n")
cat("  participant_age:   ", levels(annotations$participant_age), "\n")

# --- 3. Fit CLMM -------------------------------------------------------------
cat("\n=== Fitting CLMM ... (this may take several minutes) ===\n\n")

set.seed(42)

clmm_model <- clmm(
  harm_rating ~ model * modality * language + participant_gender + participant_age +
    (1 | uid) + (1 | participant_id),
  data    = annotations,
  control = clmm.control(
    maxIter   = 20000,
    gradTol   = 1e-7,
    innerCtrl = "warnOnly",
    trace     = TRUE
  )
)

cat("\n=== Model fitting complete ===\n\n")

# --- 4. Display results ------------------------------------------------------
cat("======================================================\n")
cat("MODEL SUMMARY\n")
cat("======================================================\n")
print(summary(clmm_model))

# --- 5. Save results ---------------------------------------------------------
sink("clmm_summary.txt")
cat("CLMM: harm_rating ~ model * modality * language + participant_gender + participant_age\n")
cat("      + (1 | uid) + (1 | participant_id)\n\n")
print(summary(clmm_model))
sink()
cat("Saved: clmm_summary.txt\n")

# Fixed-effect coefficients
coef_df <- as.data.frame(coef(summary(clmm_model)))
coef_df$term <- rownames(coef_df)
rownames(coef_df) <- NULL
coef_df <- coef_df[, c("term", "Estimate", "Std. Error", "z value", "Pr(>|z|)")]
names(coef_df) <- c("term", "estimate", "std_error", "z_value", "p_value")

threshold_rows <- grepl("\\|", coef_df$term)
coef_df$odds_ratio <- NA
coef_df$odds_ratio[!threshold_rows] <- exp(coef_df$estimate[!threshold_rows])

write.csv(coef_df, "clmm_coefficients.csv", row.names = FALSE)
cat("Saved: clmm_coefficients.csv\n")

# Random effects variances
re_var <- VarCorr(clmm_model)
re_df  <- data.frame(
  group    = names(re_var),
  variance = sapply(re_var, function(x) x[1]),
  std_dev  = sapply(re_var, function(x) sqrt(x[1]))
)
write.csv(re_df, "clmm_random_effects.csv", row.names = FALSE)
cat("Saved: clmm_random_effects.csv\n")

saveRDS(clmm_model, "clmm_model.rds")
cat("Saved: clmm_model.rds\n")

# --- 6. Quick diagnostics ----------------------------------------------------
cat("\n--- Convergence check ---\n")
cat("Convergence code:", clmm_model$convergence, "\n")
cat("(0 = converged successfully)\n")

cat("\n--- Random effect variances ---\n")
print(re_df)

cat("\n--- Log-likelihood ---\n")
cat("logLik:", logLik(clmm_model), "\n")

cat("\n=== Done ===\n")
