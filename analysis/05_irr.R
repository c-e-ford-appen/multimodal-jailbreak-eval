# =============================================================================
# Inter-Rater Reliability (IRR)
#
# Computes:
#   1. Krippendorff's alpha (ordinal) — overall, by modality, by language, by model
#   2. Gwet's AC2 (quadratic weights, k = 6 categories) — overall, by language,
#      by model, by modality, and by model × language
#   3. Confusion matrices (disagreement heatmaps) — overall, by language, by model,
#      and by model × language
#
# Note: Gwet's AC2 is the primary IRR measure (robust to prevalence paradox).
# Krippendorff's alpha is included as a secondary measure.
#
# Requires: data/annotations.csv
# Packages: irr, irrCAC, dplyr, tidyr, ggplot2
#
# Outputs:
#   irr_kripp_overall.csv                      — Krippendorff's alpha overall
#   irr_kripp_by_modality.csv                  — alpha by modality
#   irr_kripp_by_language.csv                  — alpha by language
#   irr_kripp_by_model.csv                     — alpha by model
#   irr_kripp_by_model_modality.csv            — alpha by model × modality
#   irr_kripp_by_model_language.csv            — alpha by model × language
#   irr_gwet_summary.txt                       — Gwet's AC2 results (console)
#   confusion_overall.png                      — disagreement heatmap, overall
#   confusion_enUS.png / confusion_esMX.png    — by language group
#   confusion_<model>.png                      — by model
#   confusion_<model>_<language>.png           — by model × language
# =============================================================================

# --- 0. Packages -------------------------------------------------------------
pkgs <- c("irr", "irrCAC", "dplyr", "tidyr", "ggplot2")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE))
    install.packages(p, repos = "https://cloud.r-project.org")
}
library(irr)
library(irrCAC)
library(dplyr)
library(tidyr)
library(ggplot2)

# --- 1. Load data ------------------------------------------------------------
annotations <- read.csv("data/annotations.csv", stringsAsFactors = FALSE)

cat("Rows loaded:", nrow(annotations), "\n")
cat("Unique participants:", length(unique(annotations$participant_id)), "\n")
cat("Unique items (uid):", length(unique(annotations$uid)), "\n")

# --- 2. Create item ID -------------------------------------------------------
# Each unique combination of uid + model + modality + language is one rateable
# "item". Language must be included because en-US and es-MX raters evaluated
# different language versions of the same prompt, so uid × model × modality
# alone is not unique across the full dataset.
annotations <- annotations |>
  mutate(item_id = paste(uid, model, modality, language, sep = "_"))

cat("Unique items (uid × model × modality × language):", length(unique(annotations$item_id)), "\n\n")

# --- 3. Helper: build rater matrix from a subset ----------------------------
make_matrix <- function(df) {
  df |>
    select(item_id, participant_id, rating_default) |>
    pivot_wider(id_cols     = item_id,
                names_from  = participant_id,
                values_from = rating_default) |>
    select(-item_id) |>
    as.matrix()
}

# --- 4. Helper: Krippendorff's alpha from a matrix --------------------------
compute_kripp <- function(mat) {
  kripp.alpha(t(mat), method = "ordinal")$value
}

# --- 5. Helper: percentage exact agreement -----------------------------------
pct_agreement <- function(mat) {
  total_pairs <- 0
  agree_pairs <- 0
  for (i in seq_len(nrow(mat))) {
    ratings <- mat[i, ][!is.na(mat[i, ])]
    if (length(ratings) > 1) {
      pairs <- combn(ratings, 2)
      total_pairs <- total_pairs + ncol(pairs)
      agree_pairs <- agree_pairs + sum(pairs[1, ] == pairs[2, ])
    }
  }
  if (total_pairs == 0) return(NA)
  agree_pairs / total_pairs
}

# --- 6. Quadratic weight matrix (k = 6 categories: 0–5) ---------------------
k <- 6
w <- matrix(0, nrow = k, ncol = k)
for (i in 0:(k - 1)) {
  for (j in 0:(k - 1)) {
    w[i + 1, j + 1] <- 1 - ((i - j) / (k - 1))^2
  }
}

# =============================================================================
# KRIPPENDORFF'S ALPHA
# =============================================================================

# --- 7. Overall Krippendorff's alpha -----------------------------------------
cat("=== KRIPPENDORFF'S ALPHA ===\n\n")

mat_overall <- make_matrix(annotations)
alpha_overall <- compute_kripp(mat_overall)
cat("Overall alpha (ordinal):", round(alpha_overall, 4), "\n\n")
write.csv(data.frame(scope = "overall", alpha = alpha_overall),
          "irr_kripp_overall.csv", row.names = FALSE)

# --- 8. Alpha by modality ----------------------------------------------------
alpha_by_modality <- annotations |>
  group_by(modality) |>
  group_modify(~ {
    mat <- make_matrix(.x)
    tibble(alpha = compute_kripp(mat))
  })

cat("Alpha by modality:\n")
print(alpha_by_modality)
write.csv(alpha_by_modality, "irr_kripp_by_modality.csv", row.names = FALSE)

# --- 9. Alpha by language ----------------------------------------------------
alpha_by_language <- annotations |>
  group_by(language) |>
  group_modify(~ {
    mat <- make_matrix(.x)
    tibble(alpha = compute_kripp(mat))
  })

cat("\nAlpha by language:\n")
print(alpha_by_language)
write.csv(alpha_by_language, "irr_kripp_by_language.csv", row.names = FALSE)

# --- 10. Alpha by model -------------------------------------------------------
alpha_by_model <- annotations |>
  group_by(model) |>
  group_modify(~ {
    mat <- make_matrix(.x)
    tibble(alpha = compute_kripp(mat))
  })

cat("\nAlpha by model:\n")
print(alpha_by_model)
write.csv(alpha_by_model, "irr_kripp_by_model.csv", row.names = FALSE)

# --- 11. Alpha by model × modality -------------------------------------------
alpha_by_model_modality <- annotations |>
  group_by(model, modality) |>
  group_modify(~ {
    mat <- make_matrix(.x)
    tibble(alpha = compute_kripp(mat))
  })

cat("\nAlpha by model × modality:\n")
print(alpha_by_model_modality)
write.csv(alpha_by_model_modality, "irr_kripp_by_model_modality.csv", row.names = FALSE)

# --- 12. Alpha by model × language -------------------------------------------
alpha_by_model_language <- annotations |>
  group_by(model, language) |>
  group_modify(~ {
    mat <- make_matrix(.x)
    tibble(alpha = compute_kripp(mat))
  })

cat("\nAlpha by model × language:\n")
print(alpha_by_model_language)
write.csv(alpha_by_model_language, "irr_kripp_by_model_language.csv", row.names = FALSE)

# =============================================================================
# GWET'S AC2 (quadratic weights)
# =============================================================================
cat("\n\n=== GWET'S AC2 (quadratic weights, k = 6) ===\n")

languages <- c("en-US", "es-MX")
models    <- unique(annotations$model)

sink("irr_gwet_summary.txt")
cat("Gwet's AC2 — quadratic weights, k = 6 categories\n")
cat("Primary IRR measure (robust to prevalence/marginal-homogeneity paradox)\n\n")

# Overall
cat("=== OVERALL ===\n")
mat_overall <- make_matrix(annotations)
res_overall <- gwet.ac1.raw(mat_overall, weights = w)
print(res_overall$est)
cat("% exact agreement:", round(pct_agreement(mat_overall) * 100, 2), "%\n\n")

# By language group
cat("=== BY LANGUAGE GROUP ===\n")
for (lang in languages) {
  cat("\nLanguage:", lang, "\n")
  df_sub  <- annotations |> filter(language == lang)
  mat     <- make_matrix(df_sub)
  result  <- gwet.ac1.raw(mat, weights = w)
  coeff   <- result$est
  cat("AC2 =", coeff$coeff.val,
      "| 95% CI = [", coeff$conf.int[1], ",", coeff$conf.int[2], "]\n")
  cat("% exact agreement:", round(pct_agreement(mat) * 100, 2), "%\n")
}

# By model
cat("\n=== BY MODEL ===\n")
for (m in models) {
  cat("\nModel:", m, "\n")
  df_sub <- annotations |> filter(model == m)
  mat    <- make_matrix(df_sub)
  if (nrow(mat) < 2) { cat("Too few items.\n"); next }
  result <- gwet.ac1.raw(mat, weights = w)
  coeff  <- result$est
  cat("AC2 =", coeff$coeff.val,
      "| 95% CI = [", coeff$conf.int[1], ",", coeff$conf.int[2], "]\n")
  cat("% exact agreement:", round(pct_agreement(mat) * 100, 2), "%\n")
}

# By modality
cat("\n=== BY MODALITY ===\n")
for (mod in c("multimodal", "text-only")) {
  cat("\nModality:", mod, "\n")
  df_sub <- annotations |> filter(modality == mod)
  mat    <- make_matrix(df_sub)
  result <- gwet.ac1.raw(mat, weights = w)
  coeff  <- result$est
  cat("AC2 =", coeff$coeff.val,
      "| 95% CI = [", coeff$conf.int[1], ",", coeff$conf.int[2], "]\n")
  cat("% exact agreement:", round(pct_agreement(mat) * 100, 2), "%\n")
}

# By model × language
cat("\n=== BY MODEL x LANGUAGE ===\n")
for (m in models) {
  for (lang in languages) {
    cat("\nModel:", m, "| Language:", lang, "\n")
    df_sub <- annotations |> filter(model == m, language == lang)
    mat    <- make_matrix(df_sub)
    if (nrow(mat) < 2) { cat("Too few items.\n"); next }
    result <- gwet.ac1.raw(mat, weights = w)
    coeff  <- result$est
    cat("AC2 =", coeff$coeff.val,
        "| 95% CI = [", coeff$conf.int[1], ",", coeff$conf.int[2], "]\n")
    cat("% exact agreement:", round(pct_agreement(mat) * 100, 2), "%\n")
  }
}

sink()
cat("Saved: irr_gwet_summary.txt\n")

# Also print to console
cat("\nOverall Gwet's AC2:\n")
print(gwet.ac1.raw(make_matrix(annotations), weights = w)$est)

# =============================================================================
# CONFUSION MATRICES (disagreement heatmaps)
# =============================================================================
cat("\n=== CONFUSION MATRICES ===\n")

plot_confusion <- function(df, title = "Harm Rating Disagreement Matrix",
                           subtitle = "Excludes Exact Agreements",
                           filename = NULL) {
  ratings_long <- df |>
    mutate(
      item_id        = paste(uid, model, modality, language, sep = "_"),
      participant_id = as.character(participant_id)
    ) |>
    select(item_id, participant_id, rating_default)

  rating_pairs <- ratings_long |>
    rename(rater1 = participant_id, rating1 = rating_default) |>
    inner_join(
      ratings_long |> rename(rater2 = participant_id, rating2 = rating_default),
      by = "item_id", relationship = "many-to-many"
    ) |>
    filter(rater1 < rater2, !is.na(rating1), !is.na(rating2))

  confusion_props <- rating_pairs |>
    mutate(r_low  = pmin(rating1, rating2),
           r_high = pmax(rating1, rating2)) |>
    filter(r_low != r_high) |>
    count(r_low, r_high) |>
    complete(r_low = 0:5, r_high = 0:5, fill = list(n = 0)) |>
    mutate(prop = n / sum(n))

  p <- ggplot(confusion_props,
              aes(x = factor(r_high), y = factor(r_low), fill = prop)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = "#08306b",
                        limits = c(0, max(confusion_props$prop)),
                        name = "Proportion\nof Disagreements") +
    labs(title = title, subtitle = subtitle,
         x = "Rating by Annotator 2", y = "Rating by Annotator 1") +
    coord_fixed() +
    theme_bw(base_size = 12)

  if (!is.null(filename)) {
    ggsave(filename, p, width = 8, height = 6, dpi = 300)
    cat("Saved:", filename, "\n")
  }

  invisible(p)
}

# Overall
plot_confusion(annotations,
               title    = "Disagreement Matrix \u2014 Overall",
               filename = "confusion_overall.png")

# Per language group
for (lang in languages) {
  plot_confusion(
    annotations |> filter(language == lang),
    title    = paste("Disagreement Matrix \u2014", lang),
    filename = paste0("confusion_", gsub("-", "", lang), ".png")
  )
}

# Per model
for (m in models) {
  plot_confusion(
    annotations |> filter(model == m),
    title    = paste("Disagreement Matrix \u2014", m),
    filename = paste0("confusion_", gsub(" ", "_", m), ".png")
  )
}

# Per model × language
for (m in models) {
  for (lang in languages) {
    plot_confusion(
      annotations |> filter(model == m, language == lang),
      title    = paste0("Disagreement Matrix \u2014 ", m, " | ", lang),
      filename = paste0("confusion_", gsub(" ", "_", m), "_",
                        gsub("-", "", lang), ".png")
    )
  }
}

cat("\n=== All IRR outputs saved. ===\n")
