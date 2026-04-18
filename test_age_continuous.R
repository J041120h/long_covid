#!/usr/bin/env Rscript
# =============================================================================
# Age-as-Continuous vs Age-as-Categorical Testing
# =============================================================================
# For every gene previously flagged as differential in age-related analyses,
# test whether expression varies LINEARLY with continuous age (justifying
# continuous treatment) or shows NON-LINEAR structure (justifying the current
# categorical Young/Middle/Old bucketing).
#
# Inputs: existing DE gene lists + pseudobulk expression trees that contain
# "Age at enrollment" (continuous) and "age_cluster" (categorical) columns.
#
# Per (source, celltype, month-or-pooled, gene) we fit:
#   M0: expr ~ covariates                                   (baseline)
#   M1: expr ~ age_continuous + covariates                  (linear)
#   M2: expr ~ age_continuous + I(age_continuous^2) + ...   (quadratic)
#   Mc: expr ~ age_cluster + covariates                     (categorical)
#
# Decision per gene:
#   - p_quadratic >= 0.05 AND AIC(M1) <= AIC(Mc) + 2      -> "Linear"
#   - p_quadratic <  0.05 OR  AIC(Mc) <  AIC(M1) - 2      -> "Non-linear"
#   - both p_linear >= 0.05 AND p_categorical >= 0.05     -> "No age effect"
#
# Author: hjiang / Claude
# Output: /dcs07/hongkai/data/harry/result/long_covid/analysis/age_continous_testing
# =============================================================================

setup_env <- function(env_lib = "~/R_envs/differential_gene") {
  .libPaths(path.expand(env_lib))
  suppressPackageStartupMessages({
    library(data.table); library(dplyr); library(tidyr)
    library(ggplot2); library(ggrepel)
    library(future); library(future.apply)
  })
}

setup_parallel <- function(n_workers = NULL, memory_per_worker = "8GB") {
  if (is.null(n_workers)) n_workers <- max(1, parallel::detectCores() - 1)
  if (.Platform$OS.type == "unix") plan(multicore, workers = n_workers)
  else                              plan(multisession, workers = n_workers)
  mem <- as.numeric(gsub("[^0-9.]", "", memory_per_worker))
  if (grepl("GB", memory_per_worker, ignore.case = TRUE)) mem <- mem * 1024^3
  else if (grepl("MB", memory_per_worker, ignore.case = TRUE)) mem <- mem * 1024^2
  options(future.globals.maxSize = mem)
  message(sprintf("Parallel: %d workers, %s each", n_workers, memory_per_worker))
  invisible(n_workers)
}

# =============================================================================
# Source configurations — one entry per DE pipeline to scan.
# Each entry knows where its gene lists live and which pseudobulk tree carries
# the expression + continuous age for that pipeline.
# =============================================================================
build_sources <- function() {
  root_simple <- "/dcs07/hongkai/data/harry/result/long_covid/sample_pseudobulk_differential_analysis"
  root_decouple <- "/dcs07/hongkai/data/harry/result/long_covid/LC_recovered_decouple"

  list(
    # A. Cross-group categorical Age DE (per month, named celltypes)
    list(
      label     = "cross_group_Age",
      de_root   = file.path(root_simple, "step1/step1_cross_group/Age_Young_Middle_Old"),
      pb_root   = file.path(root_simple, "different_time_point"),
      tp_regex  = "^([0-9]+)_month$",  # tp dir name -> month number
      pb_tp_fmt = "%d_month",           # how to format month into pb subdir name
      layout    = "month_celltype"      # de_root/<month>/<celltype>/<contrast>/*.txt
    ),

    # B. Across-timepoint by Age DE (pooled per celltype, contrasts include age prefix)
    list(
      label     = "across_timepoint_by_Age",
      de_root   = file.path(root_simple, "step2/step2_across_timepoint/by_Age"),
      pb_root   = file.path(root_simple, "different_time_point"),
      tp_regex  = NA,  # no month folder; gene applies to pooled months
      pb_tp_fmt = "%d_month",
      layout    = "celltype_contrast"   # de_root/<celltype>/<contrast>/*.txt
    ),

    # C. Interaction LC x Age, step1 (per-month, leiden-numbered celltypes)
    list(
      label     = "interaction_step1_leiden_0.25_LC_x_Age",
      de_root   = file.path(root_decouple, "leiden_0.25/interaction_step1/step1_interaction/LC_x_Age"),
      pb_root   = file.path(root_decouple, "leiden_0.25/merged"),
      tp_regex  = "^([0-9]+)$",
      pb_tp_fmt = "%d",
      layout    = "month_celltype"
    ),
    list(
      label     = "interaction_step1_leiden_1_LC_x_Age",
      de_root   = file.path(root_decouple, "leiden_1/interaction_step1/step1_interaction/LC_x_Age"),
      pb_root   = file.path(root_decouple, "leiden_1/merged"),
      tp_regex  = "^([0-9]+)$",
      pb_tp_fmt = "%d",
      layout    = "month_celltype"
    )
  )
}

# =============================================================================
# DE gene collection
# =============================================================================
# Read one gene list file (one gene per line; skip blanks + ENSG)
read_gene_list <- function(path) {
  if (!file.exists(path)) return(character(0))
  g <- tryCatch(readLines(path, warn = FALSE), error = function(e) character(0))
  g <- trimws(g)
  g <- g[nchar(g) > 0]
  g[!grepl("^ENSG", g)]
}

# Enumerate all (source, celltype, month) DE gene-list files across all sources
collect_de_gene_tasks <- function(sources) {
  tasks <- list()

  for (src in sources) {
    if (!dir.exists(src$de_root)) {
      message("  Skipping (missing): ", src$label, " -> ", src$de_root)
      next
    }

    if (src$layout == "month_celltype") {
      tp_dirs <- list.dirs(src$de_root, recursive = FALSE, full.names = FALSE)
      for (tp_dir in tp_dirs) {
        m <- regmatches(tp_dir, regexec(src$tp_regex, tp_dir))[[1]]
        if (length(m) < 2) next
        month_num <- suppressWarnings(as.integer(m[2]))
        if (is.na(month_num)) next

        ct_dirs <- list.dirs(file.path(src$de_root, tp_dir), recursive = FALSE, full.names = FALSE)
        for (ct in ct_dirs) {
          contrast_dirs <- list.dirs(file.path(src$de_root, tp_dir, ct),
                                     recursive = FALSE, full.names = FALSE)
          genes_union <- character(0)
          for (cdir in contrast_dirs) {
            gl <- file.path(src$de_root, tp_dir, ct, cdir, "significant_genes_FDR_0.05_all.txt")
            genes_union <- c(genes_union, read_gene_list(gl))
          }
          genes_union <- unique(genes_union)
          if (length(genes_union) == 0) next
          tasks[[length(tasks) + 1]] <- list(
            source_label = src$label, celltype = ct, month = month_num,
            genes = genes_union, pb_root = src$pb_root,
            pb_tp_fmt = src$pb_tp_fmt, layout = src$layout
          )
        }
      }
    } else if (src$layout == "celltype_contrast") {
      ct_dirs <- list.dirs(src$de_root, recursive = FALSE, full.names = FALSE)
      for (ct in ct_dirs) {
        contrast_dirs <- list.dirs(file.path(src$de_root, ct), recursive = FALSE, full.names = FALSE)
        genes_union <- character(0)
        for (cdir in contrast_dirs) {
          gl <- file.path(src$de_root, ct, cdir, "significant_genes_FDR_0.05_all.txt")
          genes_union <- c(genes_union, read_gene_list(gl))
        }
        genes_union <- unique(genes_union)
        if (length(genes_union) == 0) next
        # For pooled-month tasks, month = NA (use all months from pb tree)
        tasks[[length(tasks) + 1]] <- list(
          source_label = src$label, celltype = ct, month = NA_integer_,
          genes = genes_union, pb_root = src$pb_root,
          pb_tp_fmt = src$pb_tp_fmt, layout = src$layout
        )
      }
    }
  }

  tasks
}

# =============================================================================
# Load pseudobulk expression + metadata for a task
# If month is NA: pool all month folders for that celltype
# =============================================================================
.detect_id_col <- function(df) {
  cn <- colnames(df)
  hit <- intersect(c("sample","sample_id","Sample","SampleID","id","ID"), cn)
  if (length(hit) > 0) return(hit[1])
  if (!is.numeric(df[[1]]) && !is.integer(df[[1]])) return(cn[1])
  NULL
}

load_one_pb <- function(pb_folder) {
  ef <- file.path(pb_folder, "pseudobulk_expression.csv")
  mf <- file.path(pb_folder, "pseudobulk_metadata.csv")
  if (!file.exists(ef) || !file.exists(mf)) return(NULL)

  expr_df <- tryCatch(as.data.frame(fread(ef)), error = function(e) NULL)
  meta_df <- tryCatch(as.data.frame(fread(mf)), error = function(e) NULL)
  if (is.null(expr_df) || is.null(meta_df)) return(NULL)

  eid <- .detect_id_col(expr_df); mid <- .detect_id_col(meta_df)
  if (is.null(eid) || is.null(mid)) return(NULL)

  rownames(expr_df) <- make.unique(as.character(expr_df[[eid]])); expr_df[[eid]] <- NULL
  rownames(meta_df) <- make.unique(as.character(meta_df[[mid]])); meta_df[[mid]] <- NULL

  expr_mat <- as.matrix(expr_df); mode(expr_mat) <- "numeric"
  common <- intersect(rownames(expr_mat), rownames(meta_df))
  if (length(common) < 3) return(NULL)

  list(expr = expr_mat[common, , drop = FALSE], meta = meta_df[common, , drop = FALSE])
}

load_task_pb <- function(task) {
  if (is.na(task$month)) {
    # Pool all month folders for this celltype
    tp_dirs <- list.dirs(task$pb_root, recursive = FALSE, full.names = FALSE)
    dat_list <- list()
    for (tp_dir in tp_dirs) {
      folder <- file.path(task$pb_root, tp_dir, task$celltype)
      if (!dir.exists(folder)) next
      d <- load_one_pb(folder)
      if (is.null(d)) next
      # Prefix rownames with month to keep uniqueness across months
      rownames(d$expr) <- paste0(tp_dir, "_", rownames(d$expr))
      rownames(d$meta) <- paste0(tp_dir, "_", rownames(d$meta))
      dat_list[[length(dat_list) + 1]] <- d
    }
    if (length(dat_list) == 0) return(NULL)
    common_genes <- Reduce(intersect, lapply(dat_list, function(d) colnames(d$expr)))
    if (length(common_genes) < 2) return(NULL)
    expr <- do.call(rbind, lapply(dat_list, function(d) d$expr[, common_genes, drop = FALSE]))
    # Merge metadata (keep columns in common)
    common_meta_cols <- Reduce(intersect, lapply(dat_list, function(d) colnames(d$meta)))
    meta <- do.call(rbind, lapply(dat_list, function(d) d$meta[, common_meta_cols, drop = FALSE]))
    return(list(expr = expr, meta = meta))
  }
  folder <- file.path(task$pb_root, sprintf(task$pb_tp_fmt, task$month), task$celltype)
  load_one_pb(folder)
}

# =============================================================================
# Pull continuous age + covariates from metadata. Returns a data.frame with:
#   age_num, age_cluster, lc, sex, bmi, month
# Any row with missing age_num is dropped; factors are dropped if <2 levels.
# =============================================================================
find_col_by_variants <- function(key, df) {
  if (is.null(df)) return(NULL)
  variants <- c(key, gsub(" ", "_", key), gsub(" ", ".", key),
                tolower(key), gsub(" ", "_", tolower(key)))
  for (name in variants) if (name %in% colnames(df)) return(name)
  NULL
}

extract_covariates <- function(meta) {
  age_col   <- find_col_by_variants("Age at enrollment", meta)
  clust_col <- find_col_by_variants("age_cluster", meta)
  lc_col    <- find_col_by_variants("LC/Recovered", meta)
  sex_col   <- find_col_by_variants("Sex", meta)
  bmi_col   <- find_col_by_variants("BMI category", meta)
  month_col <- find_col_by_variants("month", meta)

  if (is.null(age_col)) return(NULL)

  df <- data.frame(row.names = rownames(meta), stringsAsFactors = FALSE)
  df$age_num <- suppressWarnings(as.numeric(as.character(meta[[age_col]])))

  if (!is.null(clust_col)) {
    df$age_cluster <- factor(as.character(meta[[clust_col]]),
                             levels = c("Young", "Middle", "Old"))
  } else {
    df$age_cluster <- factor(rep(NA, nrow(meta)), levels = c("Young", "Middle", "Old"))
  }

  df$lc    <- if (!is.null(lc_col))    factor(as.character(meta[[lc_col]]))    else factor(NA)
  df$sex   <- if (!is.null(sex_col))   factor(as.character(meta[[sex_col]]))   else factor(NA)
  df$bmi   <- if (!is.null(bmi_col))   factor(as.character(meta[[bmi_col]]))   else factor(NA)
  df$month <- if (!is.null(month_col)) factor(as.character(meta[[month_col]])) else factor(NA)

  # Drop rows with NA continuous age
  df <- df[!is.na(df$age_num), , drop = FALSE]
  df
}

# Build RHS covariate formula string from whichever covariates actually vary
build_cov_rhs <- function(cov_df, min_per_level = 2) {
  parts <- c()
  for (k in c("lc", "sex", "bmi", "month")) {
    v <- cov_df[[k]]
    if (is.null(v)) next
    if (all(is.na(v))) next
    vf <- droplevels(factor(v))
    if (nlevels(vf) >= 2 && all(table(vf) >= min_per_level)) {
      parts <- c(parts, k)
    }
  }
  if (length(parts) == 0) return("1")
  paste(parts, collapse = " + ")
}

# Compute adjusted R^2 safely
adj_r2 <- function(fit) {
  s <- tryCatch(summary(fit), error = function(e) NULL)
  if (is.null(s) || is.null(s$adj.r.squared)) return(NA_real_)
  s$adj.r.squared
}

# =============================================================================
# Per-gene test: fit baseline / linear / quadratic / categorical models
# Returns a 1-row data.frame of statistics
# =============================================================================
test_one_gene <- function(y, cov_df, cov_rhs) {
  df <- cov_df
  df$y <- y
  keep <- is.finite(df$y) & !is.na(df$age_num)
  df <- df[keep, , drop = FALSE]
  n <- nrow(df)
  if (n < 6) return(NULL)

  # Models
  f_base <- as.formula(paste("y ~", cov_rhs))
  f_lin  <- as.formula(paste("y ~ age_num +", cov_rhs))
  f_quad <- as.formula(paste("y ~ age_num + I(age_num^2) +", cov_rhs))

  fit_base <- tryCatch(lm(f_base, data = df), error = function(e) NULL)
  fit_lin  <- tryCatch(lm(f_lin,  data = df), error = function(e) NULL)
  fit_quad <- tryCatch(lm(f_quad, data = df), error = function(e) NULL)

  # Categorical: requires >=2 age_cluster levels with data
  have_cat <- !is.null(df$age_cluster) &&
    nlevels(droplevels(df$age_cluster[!is.na(df$age_cluster)])) >= 2 &&
    sum(!is.na(df$age_cluster)) >= 6
  fit_cat <- NULL
  if (have_cat) {
    f_cat <- as.formula(paste("y ~ age_cluster +", cov_rhs))
    fit_cat <- tryCatch(lm(f_cat, data = df[!is.na(df$age_cluster), , drop = FALSE]),
                         error = function(e) NULL)
  }

  if (is.null(fit_lin)) return(NULL)

  # Linear age p-value (ANOVA adding age_num)
  p_linear <- tryCatch({
    av <- anova(fit_base, fit_lin)
    av$`Pr(>F)`[2]
  }, error = function(e) NA_real_)

  beta_lin <- tryCatch(coef(fit_lin)[["age_num"]], error = function(e) NA_real_)
  se_lin <- tryCatch({
    cf <- summary(fit_lin)$coefficients
    if ("age_num" %in% rownames(cf)) cf["age_num", "Std. Error"] else NA_real_
  }, error = function(e) NA_real_)

  # Quadratic non-linearity test
  p_quadratic <- NA_real_
  if (!is.null(fit_quad)) {
    p_quadratic <- tryCatch({
      av <- anova(fit_lin, fit_quad)
      av$`Pr(>F)`[2]
    }, error = function(e) NA_real_)
  }

  # Categorical test
  p_categorical <- NA_real_
  aic_cat <- NA_real_; r2_cat <- NA_real_
  if (!is.null(fit_cat)) {
    p_categorical <- tryCatch({
      # Same data subset used for fit_cat: rebuild a matching base
      df_cat <- df[!is.na(df$age_cluster), , drop = FALSE]
      fit_base_cat <- lm(f_base, data = df_cat)
      anova(fit_base_cat, fit_cat)$`Pr(>F)`[2]
    }, error = function(e) NA_real_)
    aic_cat <- tryCatch(AIC(fit_cat), error = function(e) NA_real_)
    r2_cat  <- adj_r2(fit_cat)
  }

  aic_lin  <- tryCatch(AIC(fit_lin),  error = function(e) NA_real_)
  aic_quad <- if (!is.null(fit_quad)) tryCatch(AIC(fit_quad), error = function(e) NA_real_) else NA_real_
  r2_lin   <- adj_r2(fit_lin)
  r2_quad  <- if (!is.null(fit_quad)) adj_r2(fit_quad) else NA_real_

  # Decision
  alpha <- 0.05
  age_effect_anywhere <-
    (!is.na(p_linear) && p_linear < alpha) ||
    (!is.na(p_categorical) && p_categorical < alpha) ||
    (!is.na(p_quadratic) && p_quadratic < alpha)

  decision <- if (!age_effect_anywhere) {
    "No_age_effect"
  } else if (!is.na(p_quadratic) && p_quadratic < alpha) {
    "Non_linear"
  } else if (!is.na(aic_cat) && !is.na(aic_lin) && aic_cat < aic_lin - 2) {
    "Non_linear"
  } else {
    "Linear"
  }

  data.frame(
    n_samples      = n,
    age_min        = min(df$age_num, na.rm = TRUE),
    age_max        = max(df$age_num, na.rm = TRUE),
    beta_linear    = beta_lin,
    se_linear      = se_lin,
    p_linear       = p_linear,
    p_quadratic    = p_quadratic,
    p_categorical  = p_categorical,
    aic_linear     = aic_lin,
    aic_quadratic  = aic_quad,
    aic_categorical= aic_cat,
    delta_aic_cat_minus_lin = aic_cat - aic_lin,
    adj_r2_linear  = r2_lin,
    adj_r2_quadratic = r2_quad,
    adj_r2_categorical = r2_cat,
    decision       = decision,
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Per-gene visualization: scatter vs continuous age with linear+quadratic fits,
# colored by LC, with categorical mean segments overlaid for comparison.
# =============================================================================
plot_one_gene <- function(gene, y, cov_df, task, decision, stats_row, outdir) {
  df <- cov_df
  df$y <- y
  df <- df[is.finite(df$y) & !is.na(df$age_num), , drop = FALSE]
  if (nrow(df) < 6) return(invisible(NULL))

  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  # Categorical means + SE for overlay
  cat_df <- NULL
  if (any(!is.na(df$age_cluster))) {
    cat_df <- df %>%
      filter(!is.na(age_cluster)) %>%
      group_by(age_cluster) %>%
      summarise(
        age_center = mean(age_num),
        age_min = min(age_num), age_max = max(age_num),
        mean_y = mean(y), se_y = sd(y) / sqrt(n()),
        .groups = "drop"
      )
  }

  title_main <- sprintf("%s | %s | month=%s | %s",
                        gene, task$celltype,
                        ifelse(is.na(task$month), "pooled", as.character(task$month)),
                        task$source_label)
  subtitle <- sprintf(
    "decision=%s | beta_lin=%.3g (p=%.2g) | p_quad=%.2g | p_cat=%.2g | AIC_cat-AIC_lin=%.2f | n=%d",
    decision,
    stats_row$beta_linear, stats_row$p_linear,
    stats_row$p_quadratic, stats_row$p_categorical,
    stats_row$delta_aic_cat_minus_lin, stats_row$n_samples
  )

  has_lc <- !all(is.na(df$lc)) && nlevels(droplevels(factor(df$lc))) >= 1
  p <- ggplot(df, aes(x = age_num, y = y))
  if (has_lc) {
    p <- p + geom_point(aes(color = lc), alpha = 0.8, size = 2.2)
    p <- p + scale_color_manual(values = c("Recovered" = "#4DBBD5", "LC" = "#E64B35"),
                                 na.value = "grey50")
  } else {
    p <- p + geom_point(alpha = 0.8, size = 2.2, color = "#4DBBD5")
  }
  # Linear + quadratic smooths (overall, ignoring LC)
  p <- p +
    geom_smooth(method = "lm", formula = y ~ x, se = TRUE,
                color = "black", linetype = "solid", linewidth = 0.8) +
    geom_smooth(method = "lm", formula = y ~ x + I(x^2), se = FALSE,
                color = "orange", linetype = "dashed", linewidth = 0.8)

  if (!is.null(cat_df) && nrow(cat_df) > 0) {
    p <- p +
      geom_errorbar(data = cat_df,
                    aes(x = age_center, ymin = mean_y - se_y, ymax = mean_y + se_y),
                    width = 1.2, color = "#8B0000", linewidth = 0.7,
                    inherit.aes = FALSE) +
      geom_point(data = cat_df,
                 aes(x = age_center, y = mean_y),
                 shape = 18, size = 4, color = "#8B0000",
                 inherit.aes = FALSE) +
      geom_text(data = cat_df,
                aes(x = age_center, y = mean_y,
                    label = as.character(age_cluster)),
                vjust = -1.2, color = "#8B0000", size = 3.4,
                inherit.aes = FALSE)
  }

  p <- p +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(face = "bold", size = 12),
          plot.subtitle = element_text(size = 9, color = "grey30"),
          legend.position = "right") +
    labs(title = title_main, subtitle = subtitle,
         x = "Age at enrollment (years)",
         y = "Normalized expression",
         color = "LC status",
         caption = "Black solid = linear fit | Orange dashed = quadratic fit | Red diamonds = age_cluster mean ± SE")

  fname <- paste0(gsub("[^A-Za-z0-9_.-]", "_", gene), ".png")
  tryCatch(ggsave(file.path(outdir, fname), p, width = 8, height = 5, dpi = 300),
           error = function(e) NULL)
  invisible(NULL)
}

# =============================================================================
# Run all gene tests for one task
# =============================================================================
run_task <- function(task, output_dir, max_plots_per_task = 60, debug = FALSE) {
  pb <- load_task_pb(task)
  if (is.null(pb)) return(list(success = FALSE, reason = "pseudobulk not loadable",
                               task = task))

  cov_df <- extract_covariates(pb$meta)
  if (is.null(cov_df) || nrow(cov_df) < 6)
    return(list(success = FALSE, reason = "no age-at-enrollment or too few samples",
                task = task))

  # Align samples
  common <- intersect(rownames(pb$expr), rownames(cov_df))
  if (length(common) < 6)
    return(list(success = FALSE, reason = "too few samples with continuous age",
                task = task))

  expr   <- pb$expr[common, , drop = FALSE]
  cov_df <- cov_df[common, , drop = FALSE]
  cov_rhs <- build_cov_rhs(cov_df)

  genes <- intersect(task$genes, colnames(expr))
  if (length(genes) == 0)
    return(list(success = FALSE, reason = "no DE genes present in pseudobulk matrix",
                task = task))

  # Test every gene
  per_gene <- list()
  for (g in genes) {
    y <- as.numeric(expr[, g])
    row <- tryCatch(test_one_gene(y, cov_df, cov_rhs), error = function(e) NULL)
    if (is.null(row)) next
    row$gene <- g
    per_gene[[g]] <- row
  }
  if (length(per_gene) == 0)
    return(list(success = FALSE, reason = "all gene tests failed", task = task))

  gene_df <- do.call(rbind, per_gene)
  gene_df <- gene_df %>%
    mutate(source_label = task$source_label,
           celltype     = task$celltype,
           month        = task$month) %>%
    select(source_label, celltype, month, gene, everything()) %>%
    arrange(decision, p_linear)

  # Save per-task CSV
  tp_tag <- if (is.na(task$month)) "pooled" else sprintf("month_%s", task$month)
  task_dir <- file.path(output_dir, "per_task",
                        task$source_label, task$celltype, tp_tag)
  dir.create(task_dir, recursive = TRUE, showWarnings = FALSE)
  fwrite(gene_df, file.path(task_dir, "gene_results.csv"))

  # Plot a bounded number of genes (prioritize those with an age effect)
  gene_df_for_plot <- gene_df %>%
    mutate(priority = dplyr::case_when(
      decision == "Linear"     ~ 1L,
      decision == "Non_linear" ~ 2L,
      TRUE                     ~ 3L
    )) %>%
    arrange(priority, p_linear) %>%
    head(max_plots_per_task)

  fig_dir <- file.path(task_dir, "figures")
  for (i in seq_len(nrow(gene_df_for_plot))) {
    g <- gene_df_for_plot$gene[i]
    plot_one_gene(g, as.numeric(expr[, g]), cov_df, task,
                  decision = gene_df_for_plot$decision[i],
                  stats_row = gene_df_for_plot[i, ],
                  outdir    = fig_dir)
  }

  list(success = TRUE, gene_df = gene_df, task = task)
}

# =============================================================================
# Cross-task summary generator
# =============================================================================
generate_summary <- function(all_gene_dfs, skipped, output_dir) {
  summary_dir <- file.path(output_dir, "summary")
  fig_dir     <- file.path(summary_dir, "figures")
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

  if (length(all_gene_dfs) == 0) {
    writeLines("No successful tasks; nothing to summarize.",
               file.path(summary_dir, "conclusion.txt"))
    return(invisible(NULL))
  }

  combined <- do.call(rbind, all_gene_dfs)
  fwrite(combined, file.path(summary_dir, "all_gene_results_combined.csv"))

  # Per-task decision counts
  task_counts <- combined %>%
    group_by(source_label, celltype, month, decision) %>%
    summarise(n = dplyr::n(), .groups = "drop") %>%
    tidyr::pivot_wider(names_from = decision, values_from = n, values_fill = 0L)
  fwrite(task_counts, file.path(summary_dir, "decision_counts_per_task.csv"))

  # Global decision counts
  global <- combined %>%
    group_by(decision) %>%
    summarise(n_genes = dplyr::n(), .groups = "drop") %>%
    mutate(pct = round(100 * n_genes / sum(n_genes), 2))
  fwrite(global, file.path(summary_dir, "decision_counts_global.csv"))

  # Per-gene consensus across all (source, celltype, month) contexts
  gene_consensus <- combined %>%
    group_by(gene) %>%
    summarise(
      n_contexts     = dplyr::n(),
      n_linear       = sum(decision == "Linear",       na.rm = TRUE),
      n_nonlinear    = sum(decision == "Non_linear",   na.rm = TRUE),
      n_no_effect    = sum(decision == "No_age_effect", na.rm = TRUE),
      median_p_lin   = median(p_linear, na.rm = TRUE),
      median_p_quad  = median(p_quadratic, na.rm = TRUE),
      median_delta_aic = median(delta_aic_cat_minus_lin, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(pct_linear    = n_linear    / n_contexts,
           pct_nonlinear = n_nonlinear / n_contexts) %>%
    arrange(desc(pct_linear), desc(n_contexts))
  fwrite(gene_consensus, file.path(summary_dir, "gene_consensus_across_contexts.csv"))

  # -- Figure 1: Global pie / bar of decisions --
  p1 <- ggplot(global, aes(x = reorder(decision, -n_genes), y = n_genes, fill = decision)) +
    geom_bar(stat = "identity", alpha = 0.85) +
    geom_text(aes(label = sprintf("%d\n(%.1f%%)", n_genes, pct)),
              vjust = -0.2, size = 4, fontface = "bold") +
    scale_fill_manual(values = c("Linear" = "#4DBBD5",
                                 "Non_linear" = "#E64B35",
                                 "No_age_effect" = "grey60")) +
    theme_bw(base_size = 12) +
    theme(legend.position = "none",
          plot.title = element_text(face = "bold")) +
    labs(title = "Age pattern of DE genes: continuous-linear vs categorical",
         subtitle = sprintf("Total gene x context tests: %d", sum(global$n_genes)),
         x = "Decision", y = "Number of gene x context tests") +
    ylim(0, max(global$n_genes) * 1.15)
  ggsave(file.path(fig_dir, "global_decision_bar.png"),
         p1, width = 8, height = 5, dpi = 300)

  # -- Figure 2: per-task stacked bar --
  stacked_df <- combined %>%
    group_by(source_label, celltype, month, decision) %>%
    summarise(n = dplyr::n(), .groups = "drop") %>%
    mutate(task_label = sprintf("%s | %s | %s", source_label, celltype,
                                ifelse(is.na(month), "pooled", as.character(month))))

  # Cap the number of tasks per plot to keep figures readable
  task_totals <- stacked_df %>%
    group_by(task_label) %>% summarise(total = sum(n), .groups = "drop") %>%
    arrange(desc(total)) %>% head(60)
  stacked_df <- stacked_df %>% filter(task_label %in% task_totals$task_label)

  if (nrow(stacked_df) > 0) {
    p2 <- ggplot(stacked_df,
                 aes(x = reorder(task_label, n, FUN = sum), y = n, fill = decision)) +
      geom_bar(stat = "identity", position = "stack", alpha = 0.85) +
      scale_fill_manual(values = c("Linear" = "#4DBBD5",
                                   "Non_linear" = "#E64B35",
                                   "No_age_effect" = "grey60")) +
      coord_flip() +
      theme_bw(base_size = 10) +
      theme(plot.title = element_text(face = "bold"),
            axis.text.y = element_text(size = 8)) +
      labs(title = "Per-task distribution of age-pattern decisions",
           subtitle = "Top 60 tasks by gene count (source | celltype | month)",
           x = NULL, y = "Genes", fill = "Decision")
    ggsave(file.path(fig_dir, "per_task_decision_stacked_bar.png"),
           p2, width = 12, height = max(6, 3 + nrow(task_totals) * 0.2), dpi = 300)
  }

  # -- Figure 3: p_linear vs p_quadratic scatter (shows where non-linearity concentrates) --
  scatter_df <- combined %>%
    filter(!is.na(p_linear), !is.na(p_quadratic)) %>%
    mutate(p_linear    = pmin(pmax(p_linear, 1e-300), 1),
           p_quadratic = pmin(pmax(p_quadratic, 1e-300), 1))
  if (nrow(scatter_df) > 0) {
    p3 <- ggplot(scatter_df,
                 aes(x = -log10(p_linear), y = -log10(p_quadratic), color = decision)) +
      geom_point(alpha = 0.55, size = 1.4) +
      geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "#E64B35") +
      geom_vline(xintercept = -log10(0.05), linetype = "dashed", color = "#4DBBD5") +
      scale_color_manual(values = c("Linear" = "#4DBBD5",
                                    "Non_linear" = "#E64B35",
                                    "No_age_effect" = "grey60")) +
      theme_bw(base_size = 12) +
      theme(plot.title = element_text(face = "bold"),
            legend.position = "right") +
      labs(title = "Linear vs non-linear age signal",
           subtitle = "Top-left = significant linear trend, top-right = additionally non-linear",
           x = "-log10(p linear age)",
           y = "-log10(p quadratic term)",
           color = "Decision")
    ggsave(file.path(fig_dir, "scatter_pLinear_vs_pQuadratic.png"),
           p3, width = 8, height = 6, dpi = 300)
  }

  # -- Figure 4: delta AIC distribution (categorical - linear) --
  aic_df <- combined %>% filter(is.finite(delta_aic_cat_minus_lin))
  if (nrow(aic_df) > 0) {
    p4 <- ggplot(aic_df, aes(x = delta_aic_cat_minus_lin, fill = decision)) +
      geom_histogram(bins = 60, alpha = 0.85, color = "white") +
      geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "grey40") +
      geom_vline(xintercept = 0, color = "black") +
      scale_fill_manual(values = c("Linear" = "#4DBBD5",
                                   "Non_linear" = "#E64B35",
                                   "No_age_effect" = "grey60")) +
      theme_bw(base_size = 12) +
      theme(plot.title = element_text(face = "bold")) +
      labs(title = "AIC(categorical) - AIC(linear)",
           subtitle = "< -2: categorical clearly better | > 2: linear clearly better",
           x = "Delta AIC (categorical - linear)", y = "Count", fill = "Decision")
    ggsave(file.path(fig_dir, "delta_aic_histogram.png"),
           p4, width = 8, height = 5, dpi = 300)
  }

  # -- Written conclusion --
  n_total <- sum(global$n_genes)
  n_lin   <- sum(global$n_genes[global$decision == "Linear"])
  n_nonlin <- sum(global$n_genes[global$decision == "Non_linear"])
  n_none   <- sum(global$n_genes[global$decision == "No_age_effect"])
  pct_lin <- if (n_total > 0) 100 * n_lin / n_total else 0
  pct_nonlin <- if (n_total > 0) 100 * n_nonlin / n_total else 0

  verdict <- if (pct_lin >= 60 && pct_nonlin <= 20) {
    paste("RECOMMENDATION: Treat age as a CONTINUOUS linear predictor.",
          "Most differential genes show a linear trend with age; little",
          "evidence of non-linearity that would require categorical bucketing.")
  } else if (pct_nonlin >= 30) {
    paste("RECOMMENDATION: Keep age as a CATEGORICAL predictor (or use a",
          "flexible spline). A non-trivial fraction of genes show true",
          "non-linear age patterns that a purely linear term would miss.")
  } else {
    paste("RECOMMENDATION: Mixed signal. Neither pure-linear nor pure-categorical",
          "dominates. Consider a hybrid (e.g., spline on age with df=3) or run",
          "both models as sensitivity checks.")
  }

  lines <- c(
    paste(rep("=", 80), collapse = ""),
    "AGE AS CONTINUOUS vs CATEGORICAL — CONCLUSION",
    paste(rep("=", 80), collapse = ""),
    "",
    sprintf("Total gene x context tests: %d", n_total),
    sprintf("  Linear age pattern:     %d (%.1f%%)", n_lin,    pct_lin),
    sprintf("  Non-linear age pattern: %d (%.1f%%)", n_nonlin, pct_nonlin),
    sprintf("  No detectable effect:   %d (%.1f%%)", n_none, 100 * n_none / max(n_total, 1)),
    "",
    sprintf("Skipped tasks: %d", length(skipped)),
    "",
    paste(rep("-", 80), collapse = ""),
    verdict,
    paste(rep("-", 80), collapse = ""),
    "",
    "Decision rules applied per gene:",
    "  * p_quadratic < 0.05                 -> Non_linear",
    "  * AIC(categorical) < AIC(linear) - 2 -> Non_linear",
    "  * p_linear < 0.05 AND above not met  -> Linear",
    "  * p_linear >= 0.05 AND p_cat >= 0.05 -> No_age_effect",
    "",
    "Per-task CSVs:  per_task/<source>/<celltype>/<month>/gene_results.csv",
    "Per-gene plots: per_task/<source>/<celltype>/<month>/figures/<gene>.png",
    "Combined CSV:   summary/all_gene_results_combined.csv"
  )
  writeLines(lines, file.path(summary_dir, "conclusion.txt"))

  # Save skip log
  if (length(skipped) > 0) {
    skip_df <- do.call(rbind, lapply(skipped, function(x) {
      data.frame(source_label = x$task$source_label,
                 celltype = x$task$celltype,
                 month = x$task$month,
                 reason = x$reason,
                 stringsAsFactors = FALSE)
    }))
    fwrite(skip_df, file.path(summary_dir, "skipped_tasks.csv"))
  }

  invisible(NULL)
}

# =============================================================================
# Orchestrator
# =============================================================================
run_age_continuous_testing <- function(output_dir,
                                       env_lib = "~/R_envs/differential_gene",
                                       n_workers = NULL,
                                       memory_per_worker = "8GB",
                                       max_plots_per_task = 60) {
  setup_env(env_lib)
  setup_parallel(n_workers, memory_per_worker)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  sources <- build_sources()

  message("\n", strrep("=", 70))
  message("AGE CONTINUOUS vs CATEGORICAL TESTING")
  message("Output: ", output_dir)
  message(strrep("=", 70))

  message("\n-- Collecting DE gene tasks from all sources --")
  tasks <- collect_de_gene_tasks(sources)
  message(sprintf("  Built %d (source, celltype, month) tasks", length(tasks)))

  if (length(tasks) == 0) {
    message("No tasks found. Exiting.")
    return(invisible(NULL))
  }

  message("\n-- Running per-task tests in parallel --")
  results <- future_lapply(tasks, function(tk) {
    tryCatch(
      run_task(tk, output_dir = output_dir, max_plots_per_task = max_plots_per_task),
      error = function(e) list(success = FALSE,
                               reason  = paste("Error:", e$message),
                               task    = tk))
  }, future.seed = TRUE)

  gene_dfs <- list()
  skipped  <- list()
  for (r in results) {
    if (isTRUE(r$success) && !is.null(r$gene_df)) {
      gene_dfs[[length(gene_dfs) + 1]] <- r$gene_df
    } else {
      skipped[[length(skipped) + 1]] <- r
    }
  }

  message(sprintf("\n  Completed: %d | Skipped: %d", length(gene_dfs), length(skipped)))

  message("\n-- Generating overall summary --")
  generate_summary(gene_dfs, skipped, output_dir)

  plan(sequential)
  message("\nDone. Results in: ", output_dir)
  invisible(list(gene_dfs = gene_dfs, skipped = skipped))
}

# =============================================================================
# ENTRY POINT
# =============================================================================
if (sys.nframe() == 0) {
  run_age_continuous_testing(
    output_dir = "/dcs07/hongkai/data/harry/result/long_covid/analysis/age_continous_testing",
    env_lib   = "~/R_envs/differential_gene",
    n_workers = NULL,
    memory_per_worker = "8GB",
    max_plots_per_task = 60
  )
}
