# =============================================================================
# Pairwise Comparisons: Harm Rating Model
#
# Emmeans contrasts from the fitted Bayesian CLMM (brms).
# All comparisons use mode = "latent" (log-odds scale).
# Odds ratios are computed by exponentiating.
# Credible intervals are 95% highest posterior density intervals (HPD).
#
# Requires: models/brms_harm_model.rds (fitted by 01_brms_harm_ratings.R)
# Packages: brms, emmeans
#
# Outputs:
#   pairwise_models_overall.csv           — all model pairs, marginalised
#   pairwise_models_by_modality.csv       — model pairs × modality
#   pairwise_models_by_language.csv       — model pairs × language
#   pairwise_models_by_lang_mod.csv       — model pairs × language × modality
#   pairwise_age.csv                      — age group comparisons
#   pairwise_modality_by_model.csv        — text-only vs. multimodal within each model
#   pairwise_modality_by_model_lang.csv   — modality contrast within each model × language
#   pairwise_language_by_model.csv        — es-MX vs. en-US within each model
#   pairwise_language_by_model_mod.csv    — language contrast within each model × modality
#   pairwise_summary.txt                  — formatted summary of all contrasts
# =============================================================================

library(brms)
library(emmeans)

# --- 1. Load saved model -----------------------------------------------------
cat("Loading model...\n")
m <- readRDS("models/brms_harm_model.rds")
cat("Model loaded.\n\n")

# emmeans on mode = "latent" collapses the ordinal thresholds into a single
# latent score, then we exponentiate to get odds ratios.

# --- 2. Helper: tidy emmeans contrast output ---------------------------------
tidy_contrasts <- function(em_obj, label = "") {
  sm <- summary(em_obj, infer = TRUE)
  df <- as.data.frame(sm)

  names(df)[names(df) == "estimate"]   <- "estimate_logodds"
  names(df)[names(df) == "lower.HPD"] <- "ci_lower_95"
  names(df)[names(df) == "upper.HPD"] <- "ci_upper_95"

  df$odds_ratio     <- exp(df$estimate_logodds)
  df$or_ci_lower_95 <- exp(df$ci_lower_95)
  df$or_ci_upper_95 <- exp(df$ci_upper_95)
  df$credible       <- ifelse(df$ci_lower_95 > 0 | df$ci_upper_95 < 0, "YES", "no")

  if (nchar(label) > 0) df$comparison_group <- label
  df
}

# --- 3. Overall pairwise model comparisons -----------------------------------
cat("Computing overall pairwise model comparisons...\n")

emm_model <- emmeans(m, ~ model, mode = "latent")
pw_overall <- pairs(emm_model, reverse = TRUE)
df_overall <- tidy_contrasts(pw_overall)

write.csv(df_overall, "pairwise_models_overall.csv", row.names = FALSE)
cat("Saved: pairwise_models_overall.csv\n\n")

# --- 4. Pairwise by modality -------------------------------------------------
cat("Computing pairwise comparisons by modality...\n")

emm_mod <- emmeans(m, ~ model | modality, mode = "latent")
pw_mod   <- pairs(emm_mod, reverse = TRUE)
df_mod   <- as.data.frame(summary(pw_mod, infer = TRUE))
df_mod$odds_ratio     <- exp(df_mod$estimate)
df_mod$or_ci_lower_95 <- exp(df_mod$lower.HPD)
df_mod$or_ci_upper_95 <- exp(df_mod$upper.HPD)
df_mod$credible       <- ifelse(df_mod$lower.HPD > 0 | df_mod$upper.HPD < 0, "YES", "no")

write.csv(df_mod, "pairwise_models_by_modality.csv", row.names = FALSE)
cat("Saved: pairwise_models_by_modality.csv\n\n")

# --- 5. Pairwise by language -------------------------------------------------
cat("Computing pairwise comparisons by language...\n")

emm_lang <- emmeans(m, ~ model | language, mode = "latent")
pw_lang  <- pairs(emm_lang, reverse = TRUE)
df_lang  <- as.data.frame(summary(pw_lang, infer = TRUE))
df_lang$odds_ratio     <- exp(df_lang$estimate)
df_lang$or_ci_lower_95 <- exp(df_lang$lower.HPD)
df_lang$or_ci_upper_95 <- exp(df_lang$upper.HPD)
df_lang$credible       <- ifelse(df_lang$lower.HPD > 0 | df_lang$upper.HPD < 0, "YES", "no")

write.csv(df_lang, "pairwise_models_by_language.csv", row.names = FALSE)
cat("Saved: pairwise_models_by_language.csv\n\n")

# --- 6. Pairwise by language × modality -------------------------------------
cat("Computing pairwise comparisons by language × modality...\n")

emm_langmod <- emmeans(m, ~ model | language * modality, mode = "latent")
pw_langmod  <- pairs(emm_langmod, reverse = TRUE)
df_langmod  <- as.data.frame(summary(pw_langmod, infer = TRUE))
df_langmod$odds_ratio     <- exp(df_langmod$estimate)
df_langmod$or_ci_lower_95 <- exp(df_langmod$lower.HPD)
df_langmod$or_ci_upper_95 <- exp(df_langmod$upper.HPD)
df_langmod$credible       <- ifelse(df_langmod$lower.HPD > 0 | df_langmod$upper.HPD < 0, "YES", "no")

write.csv(df_langmod, "pairwise_models_by_lang_mod.csv", row.names = FALSE)
cat("Saved: pairwise_models_by_lang_mod.csv\n\n")

# --- 7. Age group pairwise ---------------------------------------------------
cat("Computing age group pairwise comparisons...\n")

emm_age <- emmeans(m, ~ participant_age, mode = "latent")
pw_age  <- pairs(emm_age, reverse = TRUE)
df_age  <- as.data.frame(summary(pw_age, infer = TRUE))
df_age$odds_ratio     <- exp(df_age$estimate)
df_age$or_ci_lower_95 <- exp(df_age$lower.HPD)
df_age$or_ci_upper_95 <- exp(df_age$upper.HPD)
df_age$credible       <- ifelse(df_age$lower.HPD > 0 | df_age$upper.HPD < 0, "YES", "no")

write.csv(df_age, "pairwise_age.csv", row.names = FALSE)
cat("Saved: pairwise_age.csv\n\n")

# --- 8. Modality contrasts within each model ---------------------------------
# The main modality effect was not credible overall, but the Pixtral interaction
# was. These contrasts show text-only vs. multimodal within each individual model.
cat("Computing modality contrasts within each model...\n")

emm_mod_bymodel <- emmeans(m, ~ modality | model, mode = "latent")
pw_mod_bymodel  <- pairs(emm_mod_bymodel, reverse = TRUE)
df_mod_bymodel  <- as.data.frame(summary(pw_mod_bymodel, infer = TRUE))
df_mod_bymodel$odds_ratio     <- exp(df_mod_bymodel$estimate)
df_mod_bymodel$or_ci_lower_95 <- exp(df_mod_bymodel$lower.HPD)
df_mod_bymodel$or_ci_upper_95 <- exp(df_mod_bymodel$upper.HPD)
df_mod_bymodel$credible       <- ifelse(df_mod_bymodel$lower.HPD > 0 | df_mod_bymodel$upper.HPD < 0, "YES", "no")

write.csv(df_mod_bymodel, "pairwise_modality_by_model.csv", row.names = FALSE)
cat("Saved: pairwise_modality_by_model.csv\n")

# Modality within each model × language cell
emm_mod_bylangmod <- emmeans(m, ~ modality | model * language, mode = "latent")
pw_mod_bylangmod  <- pairs(emm_mod_bylangmod, reverse = TRUE)
df_mod_bylangmod  <- as.data.frame(summary(pw_mod_bylangmod, infer = TRUE))
df_mod_bylangmod$odds_ratio     <- exp(df_mod_bylangmod$estimate)
df_mod_bylangmod$or_ci_lower_95 <- exp(df_mod_bylangmod$lower.HPD)
df_mod_bylangmod$or_ci_upper_95 <- exp(df_mod_bylangmod$upper.HPD)
df_mod_bylangmod$credible       <- ifelse(df_mod_bylangmod$lower.HPD > 0 | df_mod_bylangmod$upper.HPD < 0, "YES", "no")

write.csv(df_mod_bylangmod, "pairwise_modality_by_model_lang.csv", row.names = FALSE)
cat("Saved: pairwise_modality_by_model_lang.csv\n\n")

# --- 9. Language contrasts within each model ---------------------------------
# The main language effect was not credible, but several model × language
# interactions were. These contrasts identify where the signal sits.
cat("Computing language contrasts within each model...\n")

emm_lang_bymodel <- emmeans(m, ~ language | model, mode = "latent")
pw_lang_bymodel  <- pairs(emm_lang_bymodel, reverse = TRUE)
df_lang_bymodel  <- as.data.frame(summary(pw_lang_bymodel, infer = TRUE))
df_lang_bymodel$odds_ratio     <- exp(df_lang_bymodel$estimate)
df_lang_bymodel$or_ci_lower_95 <- exp(df_lang_bymodel$lower.HPD)
df_lang_bymodel$or_ci_upper_95 <- exp(df_lang_bymodel$upper.HPD)
df_lang_bymodel$credible       <- ifelse(df_lang_bymodel$lower.HPD > 0 | df_lang_bymodel$upper.HPD < 0, "YES", "no")

write.csv(df_lang_bymodel, "pairwise_language_by_model.csv", row.names = FALSE)
cat("Saved: pairwise_language_by_model.csv\n")

# Language within each model × modality cell
emm_lang_bymodmod <- emmeans(m, ~ language | model * modality, mode = "latent")
pw_lang_bymodmod  <- pairs(emm_lang_bymodmod, reverse = TRUE)
df_lang_bymodmod  <- as.data.frame(summary(pw_lang_bymodmod, infer = TRUE))
df_lang_bymodmod$odds_ratio     <- exp(df_lang_bymodmod$estimate)
df_lang_bymodmod$or_ci_lower_95 <- exp(df_lang_bymodmod$lower.HPD)
df_lang_bymodmod$or_ci_upper_95 <- exp(df_lang_bymodmod$upper.HPD)
df_lang_bymodmod$credible       <- ifelse(df_lang_bymodmod$lower.HPD > 0 | df_lang_bymodmod$upper.HPD < 0, "YES", "no")

write.csv(df_lang_bymodmod, "pairwise_language_by_model_mod.csv", row.names = FALSE)
cat("Saved: pairwise_language_by_model_mod.csv\n\n")

# --- 10. Save full formatted summary -----------------------------------------
sink("pairwise_summary.txt")

cat("=========================================================\n")
cat("PAIRWISE MODEL COMPARISONS — brms cumulative(logit) model\n")
cat("All comparisons on the latent log-odds scale\n")
cat("OR = exp(estimate); CI excludes 0 → labelled as credible\n")
cat("=========================================================\n\n")

cat("--- OVERALL (marginalised over modality, language, gender, age) ---\n")
print(summary(pw_overall, infer = TRUE))

cat("\n--- BY MODALITY ---\n")
print(summary(pw_mod, infer = TRUE))

cat("\n--- BY LANGUAGE ---\n")
print(summary(pw_lang, infer = TRUE))

cat("\n--- BY LANGUAGE x MODALITY ---\n")
print(summary(pw_langmod, infer = TRUE))

cat("\n--- AGE GROUPS ---\n")
print(summary(pw_age, infer = TRUE))

cat("\n--- MODALITY (text-only vs. multimodal) WITHIN EACH MODEL ---\n")
print(summary(pw_mod_bymodel, infer = TRUE))

cat("\n--- MODALITY WITHIN EACH MODEL × LANGUAGE CELL ---\n")
print(summary(pw_mod_bylangmod, infer = TRUE))

cat("\n--- LANGUAGE (es-MX vs. en-US) WITHIN EACH MODEL ---\n")
print(summary(pw_lang_bymodel, infer = TRUE))

cat("\n--- LANGUAGE WITHIN EACH MODEL × MODALITY CELL ---\n")
print(summary(pw_lang_bymodmod, infer = TRUE))

sink()
cat("Saved: pairwise_summary.txt\n\n")

# --- 11. Quick console summary -----------------------------------------------
cat("=========================================================\n")
cat("OVERALL PAIRWISE — odds ratios at a glance\n")
cat("=========================================================\n")
cat(sprintf("%-40s  OR      95%% CI              Credible?\n", "Contrast"))
cat(strrep("-", 75), "\n")
for (i in seq_len(nrow(df_overall))) {
  cat(sprintf("%-40s  %5.2f   [%5.2f, %5.2f]      %s\n",
    df_overall$contrast[i],
    df_overall$odds_ratio[i],
    df_overall$or_ci_lower_95[i],
    df_overall$or_ci_upper_95[i],
    df_overall$credible[i]
  ))
}

cat("\n=== Done. All pairwise files saved. ===\n")
