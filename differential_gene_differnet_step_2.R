#!/usr/bin/env Rscript
# =============================================================================
# Step 2: Within-group across timepoints (pairwise) for each cell type
# Supports stratification by: LC/Recovered, BMI (Normal/Elevated), Sex (Male/Female), Age (Young/Middle/Old)
#
# Includes:
#   - Differential GENE expression (pseudobulk)
#   - Differential CELL PROPORTION analysis (limma on proportions)
#   - Two-tier significance: FDR+logFC and FDR-only
#
# STRICT MODE: If ANY batch covariate cannot be controlled, SKIP that
# analysis entirely.
# =============================================================================

# =========================================================================
# SHARED SETUP + HELPERS
# =========================================================================
setup_pipeline_env <- function(env_lib = "~/R_envs/differential_gene") {
  .libPaths(path.expand(env_lib))
  suppressPackageStartupMessages({
    library(data.table); library(dplyr); library(tidyr)
    library(ggplot2); library(limma); library(pheatmap)
    library(RColorBrewer); library(scales); library(ggrepel)
    library(future); library(future.apply)
  })
}

# =============================================================================
# Setup parallel processing
# =============================================================================
setup_parallel <- function(n_workers = NULL, memory_per_worker = "4GB") {
  if (is.null(n_workers)) {
    n_workers <- max(1, parallel::detectCores() - 1)
  }
  
  if (.Platform$OS.type == "unix") {
    plan(multicore, workers = n_workers)
  } else {
    plan(multisession, workers = n_workers)
  }
  
  mem_bytes <- as.numeric(gsub("[^0-9.]", "", memory_per_worker))
  unit <- gsub("[0-9.]", "", memory_per_worker)
  if (grepl("GB", unit, ignore.case = TRUE)) {
    mem_bytes <- mem_bytes * 1024^3
  } else if (grepl("MB", unit, ignore.case = TRUE)) {
    mem_bytes <- mem_bytes * 1024^2
  }
  options(future.globals.maxSize = mem_bytes)
  
  message(sprintf("Parallel processing enabled: %d workers", n_workers))
  invisible(n_workers)
}

# -- Colors -------------------------------------------------------------------
GROUP_COLORS <- c("Recovered" = "#4DBBD5", "LC" = "#E64B35")
BMI_COLORS   <- c("Normal" = "#00A087", "Elevated" = "#F39B7F")
SEX_COLORS   <- c("Female" = "#7570B3", "Male" = "#1B9E77")
AGE_COLORS   <- c("Young" = "#00BFC4", "Middle" = "#7CAE00", "Old" = "#F8766D")

# -- Discover structure -------------------------------------------------------
discover_structure <- function(parent_dir) {
  timepoints <- sort(list.dirs(parent_dir, recursive = FALSE, full.names = FALSE))
  if (length(timepoints) == 0) stop("No timepoint subdirectories in: ", parent_dir)
  celltype_map <- list()
  for (tp in timepoints)
    celltype_map[[tp]] <- sort(list.dirs(file.path(parent_dir, tp), recursive = FALSE, full.names = FALSE))
  all_celltypes <- sort(unique(unlist(celltype_map)))
  tp_order_fn <- function(x) { n <- as.numeric(gsub("[^0-9]", "", x)); ifelse(is.na(n), Inf, n) }
  tp_sorted <- timepoints[order(sapply(timepoints, tp_order_fn))]
  list(timepoints = timepoints, tp_sorted = tp_sorted, celltype_map = celltype_map,
       all_celltypes = all_celltypes)
}

tp_short <- function(tp_name) paste0("M", gsub("[^0-9]", "", tp_name))

# -- Data loading -------------------------------------------------------------
.detect_id_col <- function(df) {
  cn <- colnames(df)
  hit <- intersect(c("sample","sample_id","Sample","SampleID","id","ID"), cn)
  if (length(hit) > 0) return(hit[1])
  if (!is.numeric(df[[1]]) && !is.integer(df[[1]])) return(cn[1])
  NULL
}

load_celltype_data <- function(folder_path) {
  ef <- file.path(folder_path, "pseudobulk_expression.csv")
  mf <- file.path(folder_path, "pseudobulk_metadata.csv")
  if (!file.exists(ef) || !file.exists(mf)) return(NULL)
  expr_df <- as.data.frame(fread(ef)); meta_df <- as.data.frame(fread(mf))
  eid <- .detect_id_col(expr_df); mid <- .detect_id_col(meta_df)
  if (is.null(eid) || is.null(mid)) return(NULL)
  rownames(expr_df) <- make.unique(as.character(expr_df[[eid]])); expr_df[[eid]] <- NULL
  rownames(meta_df) <- make.unique(as.character(meta_df[[mid]])); meta_df[[mid]] <- NULL
  expr_mat <- as.matrix(expr_df); mode(expr_mat) <- "numeric"
  common <- intersect(rownames(expr_mat), rownames(meta_df))
  if (length(common) < 3) return(NULL)
  list(expr = expr_mat[common, , drop = FALSE], meta = meta_df[common, , drop = FALSE])
}

# =============================================================================
# Load cell type proportions data (matching Step 1 pattern)
# Format: rows = cell types, columns = samples, values = proportions
# Returns: list(prop_mat = samples x celltypes matrix, celltypes = character)
# =============================================================================
load_proportion_data <- function(prop_file) {
  if (!file.exists(prop_file)) return(NULL)
  
  prop_df <- as.data.frame(fread(prop_file))
  
  # First column is cell type names
  celltype_col <- colnames(prop_df)[1]
  celltypes <- as.character(prop_df[[celltype_col]])
  prop_df[[celltype_col]] <- NULL
  
  # Convert to numeric matrix: celltypes x samples
  prop_mat <- as.matrix(prop_df)
  mode(prop_mat) <- "numeric"
  rownames(prop_mat) <- celltypes
  
  # Transpose to samples x celltypes
  prop_mat <- t(prop_mat)
  
  if (nrow(prop_mat) < 3) return(NULL)
  
  list(prop_mat = prop_mat, celltypes = celltypes)
}

# =============================================================================
# Load metadata for proportion analysis (matching Step 1 pattern)
# Searches celltype subdirectories for a metadata file matching sample IDs
# =============================================================================
load_proportion_metadata <- function(parent_dir, tp, sample_ids) {
  ct_dirs <- list.dirs(file.path(parent_dir, tp), recursive = FALSE, full.names = TRUE)
  
  for (ct_dir in ct_dirs) {
    mf <- file.path(ct_dir, "pseudobulk_metadata.csv")
    if (!file.exists(mf)) next
    
    meta_df <- as.data.frame(fread(mf))
    mid <- .detect_id_col(meta_df)
    if (is.null(mid)) next
    
    rownames(meta_df) <- make.unique(as.character(meta_df[[mid]]))
    meta_df[[mid]] <- NULL
    
    common <- intersect(sample_ids, rownames(meta_df))
    if (length(common) >= 3) {
      return(meta_df[common, , drop = FALSE])
    }
  }
  
  NULL
}

# =============================================================================
# Unified helper: find column by possible name variants
# =============================================================================
find_col_by_variants <- function(key, df) {
  if (is.null(df)) return(NULL)
  variants <- c(key, gsub(" ", "_", key), gsub(" ", ".", key), 
                tolower(key), gsub(" ", "_", tolower(key)))
  for (name in variants) {
    if (name %in% colnames(df)) return(name)
  }
  NULL
}

# =============================================================================
# Group extraction functions
# =============================================================================
extract_group <- function(meta_df) {
  for (cn in colnames(meta_df)) {
    v <- meta_df[[cn]]
    if (is.character(v) || is.factor(v)) {
      lv <- unique(as.character(v))
      if (all(c("LC","Recovered") %in% lv) || all(c("LongCOVID","Recovered") %in% lv)) {
        g <- as.character(v); g[g == "LongCOVID"] <- "LC"
        return(factor(g, levels = c("Recovered", "LC")))
      }
    }
  }
  if (all(c("LC","Recovered") %in% colnames(meta_df))) {
    lc <- suppressWarnings(as.numeric(as.character(meta_df$LC)))
    rc <- suppressWarnings(as.numeric(as.character(meta_df$Recovered)))
    grp <- ifelse(lc == 1, "LC", ifelse(rc == 1, "Recovered", NA))
    if (any(is.na(grp))) return(NULL)
    return(factor(grp, levels = c("Recovered", "LC")))
  }
  NULL
}

extract_bmi_group <- function(meta_df) {
  bmi_cols <- c("BMI category", "BMI_category", "BMI.category", "bmi_category", "BMI")
  for (cn in bmi_cols) {
    col_found <- find_col_by_variants(cn, meta_df)
    if (!is.null(col_found)) {
      v <- as.character(meta_df[[col_found]])
      lv <- unique(v[!is.na(v)])
      if (all(c("Normal", "Elevated") %in% lv)) return(factor(v, levels = c("Normal", "Elevated")))
      if (all(c("Normal", "Obese") %in% lv)) return(factor(v, levels = c("Normal", "Obese")))
    }
  }
  NULL
}

extract_sex_group <- function(meta_df) {
  sex_cols <- c("Sex", "sex", "Gender", "gender")
  for (cn in sex_cols) {
    col_found <- find_col_by_variants(cn, meta_df)
    if (!is.null(col_found)) {
      v <- as.character(meta_df[[col_found]])
      lv <- unique(v[!is.na(v)])
      if (all(c("Male", "Female") %in% lv) || all(c("M", "F") %in% lv)) {
        v[v == "M"] <- "Male"; v[v == "F"] <- "Female"
        return(factor(v, levels = c("Female", "Male")))
      }
    }
  }
  NULL
}

extract_age_group <- function(meta_df) {
  age_cols <- c("age_cluster", "Age_cluster", "ageCluster", "AgeCluster", "age.group", "Age group")
  for (cn in age_cols) {
    col_found <- find_col_by_variants(cn, meta_df)
    if (!is.null(col_found)) {
      v <- as.character(meta_df[[col_found]])
      if (all(c("Young", "Middle", "Old") %in% unique(v[!is.na(v)]))) {
        return(factor(v, levels = c("Young", "Middle", "Old")))
      }
    }
  }
  NULL
}

# =============================================================================
# Covariate checking and preparation
# =============================================================================
can_model_covariate <- function(group, cov_vec, min_per_level = 2) {
  if (is.null(cov_vec)) return(FALSE)
  
  if (is.character(cov_vec) || is.factor(cov_vec)) {
    cov_vec <- droplevels(factor(cov_vec))
    if (nlevels(cov_vec) < 2) return(FALSE)
    tab <- table(cov_vec)
    if (any(tab < min_per_level)) return(FALSE)
    for (g in levels(group)) {
      cov_in_g <- droplevels(factor(cov_vec[group == g]))
      if (nlevels(cov_in_g) < 2) return(FALSE)
    }
    return(TRUE)
  }
  
  if (is.numeric(cov_vec)) {
    vv <- var(cov_vec, na.rm = TRUE)
    return(is.finite(vv) && vv > 0)
  }
  
  FALSE
}

check_all_batch_covariates <- function(group, meta_df, batch_keys, exclude_keys, min_per_level = 2) {
  cov_keys_to_check <- batch_keys[!batch_keys %in% exclude_keys]
  
  results <- list(can_model = TRUE, failed_covariates = character(0), details = character(0))
  
  for (cov_key in cov_keys_to_check) {
    col_found <- find_col_by_variants(cov_key, meta_df)
    
    if (is.null(col_found)) {
      results$can_model <- FALSE
      results$failed_covariates <- c(results$failed_covariates, cov_key)
      results$details <- c(results$details, sprintf("%s: NOT FOUND in metadata", cov_key))
      next
    }
    
    cov_vec <- meta_df[[col_found]]
    
    if (!can_model_covariate(group, cov_vec, min_per_level = min_per_level)) {
      results$can_model <- FALSE
      results$failed_covariates <- c(results$failed_covariates, cov_key)
      
      if (is.character(cov_vec) || is.factor(cov_vec)) {
        cov_vec <- droplevels(factor(cov_vec))
        tab <- table(cov_vec)
        tab_str <- paste(names(tab), as.integer(tab), sep="=", collapse=", ")
        per_group_info <- sapply(levels(group), function(g) {
          cov_in_g <- droplevels(factor(cov_vec[group == g]))
          paste0(g, ":[", paste(levels(cov_in_g), table(cov_in_g)[levels(cov_in_g)], sep="=", collapse=","), "]")
        })
        results$details <- c(results$details,
                             sprintf("%s: FAILED - levels=%d, counts=(%s), per_group=(%s)",
                                     cov_key, nlevels(cov_vec), tab_str, paste(per_group_info, collapse="; ")))
      } else {
        results$details <- c(results$details,
                             sprintf("%s: FAILED - insufficient variance or not valid", cov_key))
      }
    } else {
      if (is.character(cov_vec) || is.factor(cov_vec)) {
        cov_vec <- droplevels(factor(cov_vec))
        tab <- table(cov_vec)
        tab_str <- paste(names(tab), as.integer(tab), sep="=", collapse=", ")
        results$details <- c(results$details, sprintf("%s: OK - levels=%d, counts=(%s)", cov_key, nlevels(cov_vec), tab_str))
      } else {
        results$details <- c(results$details, sprintf("%s: OK - numeric covariate", cov_key))
      }
    }
  }
  
  results
}

prepare_covariates <- function(meta_df, cov_keys, exclude_keys = NULL) {
  if (is.null(cov_keys) || length(cov_keys) == 0) return(NULL)
  if (!is.null(exclude_keys)) cov_keys <- cov_keys[!cov_keys %in% exclude_keys]
  
  use_keys <- character(0)
  for (k in cov_keys) {
    col_found <- find_col_by_variants(k, meta_df)
    if (!is.null(col_found)) use_keys <- c(use_keys, col_found)
  }
  
  if (length(use_keys) == 0) return(NULL)
  
  cov_df <- meta_df[, use_keys, drop = FALSE]
  for (k in use_keys) {
    v <- cov_df[[k]]
    if (is.character(v) || is.factor(v)) cov_df[[k]] <- factor(v)
    else cov_df[[k]] <- suppressWarnings(as.numeric(as.character(v)))
  }
  
  keep <- character(0)
  for (k in use_keys) {
    v <- cov_df[[k]]
    if (is.factor(v) && nlevels(droplevels(v)) > 1) keep <- c(keep, k)
    else if (is.numeric(v)) {
      vv <- var(v, na.rm=TRUE)
      if (is.finite(vv) && vv > 0) keep <- c(keep, k)
    }
  }
  
  if (length(keep) == 0) return(NULL)
  cov_df <- cov_df[, keep, drop = FALSE]
  attr(cov_df, "complete") <- complete.cases(cov_df)
  cov_df
}

apply_covariate_filter <- function(expr_mat, group_vec, cov_df) {
  if (is.null(cov_df)) return(list(expr = expr_mat, group = group_vec, cov = NULL))
  complete <- attr(cov_df, "complete")
  if (is.null(complete)) complete <- complete.cases(cov_df)
  keep <- which(complete)
  if (length(keep) < 3) return(NULL)
  list(expr = expr_mat[keep,,drop=FALSE], group = droplevels(group_vec[keep]),
       cov = cov_df[keep,,drop=FALSE])
}

build_safe_design <- function(group, cov_df, min_resid_df = 2) {
  notes <- c()
  cov_used <- cov_df
  
  make_design <- function(covv) {
    if (!is.null(covv) && ncol(covv) > 0) {
      model.matrix(~ 0 + group + ., data = data.frame(group = group, covv))
    } else {
      model.matrix(~ 0 + group)
    }
  }
  
  resid_df <- function(des) nrow(des) - qr(des)$rank
  
  if (is.null(cov_used) || ncol(cov_used) == 0) {
    des <- make_design(NULL)
    colnames(des) <- make.names(colnames(des))
    return(list(design = des, cov_used = NULL, notes = c(notes, "Covariates: OFF")))
  }
  
  des <- make_design(cov_used)
  rdf <- resid_df(des)
  
  while (rdf < min_resid_df && !is.null(cov_used) && ncol(cov_used) > 0) {
    is_fac <- sapply(cov_used, function(x) is.factor(x) || is.character(x))
    fac_names <- names(is_fac)[is_fac]
    num_names <- setdiff(colnames(cov_used), fac_names)
    
    drop_k <- NULL
    if (length(fac_names) > 0) {
      nlv <- sapply(fac_names, function(k) nlevels(droplevels(factor(cov_used[[k]]))))
      drop_k <- fac_names[which.max(nlv)]
      notes <- c(notes, sprintf("Dropped covariate '%s' (factor) to increase residual df", drop_k))
    } else if (length(num_names) > 0) {
      drop_k <- num_names[1]
      notes <- c(notes, sprintf("Dropped covariate '%s' (numeric) to increase residual df", drop_k))
    } else break
    
    cov_used[[drop_k]] <- NULL
    if (ncol(cov_used) == 0) cov_used <- NULL
    des <- make_design(cov_used)
    rdf <- resid_df(des)
  }
  
  colnames(des) <- make.names(colnames(des))
  
  if (rdf < min_resid_df) {
    return(list(design = NULL, cov_used = NULL,
                notes = c(notes, sprintf("FAILED: residual df (%d) < min_resid_df (%d)", rdf, min_resid_df))))
  }
  
  list(design = des, cov_used = cov_used, notes = c(notes, sprintf("OK: residual df = %d", rdf)))
}

# -- Limma core ---------------------------------------------------------------
run_limma <- function(expr_mat_sxg, design, contrast_mat, coef_name) {
  mat <- t(expr_mat_sxg)
  fit <- lmFit(mat, design)
  fit <- contrasts.fit(fit, contrast_mat)
  fit <- eBayes(fit, trend = TRUE)
  tt <- topTable(fit, coef = coef_name, number = Inf, sort.by = "P")
  tt$gene <- rownames(tt)
  tt %>% transmute(gene, logFC, AveExpr, t, PValue = P.Value, FDR = adj.P.Val, B)
}

# =============================================================================
# TWO-TIER DE STATUS: FDR+logFC and FDR-only
# =============================================================================
add_de_status_generic <- function(df, up_label = "Up", down_label = "Down") {
  df %>% mutate(DE_status = factor(case_when(
    FDR < 0.05 & logFC > 1 ~ up_label,
    FDR < 0.05 & logFC < -1 ~ down_label,
    FDR < 0.05 ~ "FDR-only",
    TRUE ~ "NS"),
    levels = c(up_label, down_label, "FDR-only", "NS")))
}

find_design_col <- function(design, var_prefix, level_value) {
  dcn <- colnames(design)
  cand1 <- paste0(var_prefix, level_value)
  if (cand1 %in% dcn) return(cand1)
  cand2 <- paste0(var_prefix, make.names(level_value))
  if (cand2 %in% dcn) return(cand2)
  pattern <- paste0("^", gsub("\\.", "\\\\.", var_prefix))
  matches <- grep(pattern, dcn, value = TRUE)
  if (length(matches) > 0) {
    for (m in matches) {
      suffix <- sub(paste0("^", gsub("\\.", "\\\\.", var_prefix)), "", m)
      if (suffix == level_value || suffix == make.names(level_value)) return(m)
    }
    for (m in matches) {
      if (grepl(gsub("[^A-Za-z0-9]", ".", level_value), m)) return(m)
    }
  }
  NULL
}

# =============================================================================
# Gene filtering helpers
# =============================================================================
filter_ensg_genes <- function(gene_vec) {
  gene_vec[!grepl("^ENSG", gene_vec)]
}

filter_ensg_from_df <- function(df) {
  df %>% filter(!grepl("^ENSG", gene))
}

# =============================================================================
# Gene selection helpers (unified, matching Step 1 pattern)
# =============================================================================
get_sig_genes <- function(res, fdr_thresh = 0.05) {
  res %>% filter(!grepl("^ENSG", gene)) %>%
    filter(!is.na(FDR), FDR < fdr_thresh) %>%
    arrange(FDR, desc(abs(logFC))) %>% pull(gene)
}

get_sig_genes_strict <- function(res, fdr_thresh = 0.05, logfc_thresh = 1) {
  res %>% filter(!grepl("^ENSG", gene)) %>%
    filter(!is.na(FDR), FDR < fdr_thresh, abs(logFC) > logfc_thresh) %>%
    arrange(FDR, desc(abs(logFC))) %>% pull(gene)
}

get_sig_genes_fdr_only <- function(res, fdr_thresh = 0.05, logfc_thresh = 1) {
  res %>% filter(!grepl("^ENSG", gene)) %>%
    filter(!is.na(FDR), FDR < fdr_thresh, abs(logFC) <= logfc_thresh) %>%
    arrange(FDR, desc(abs(logFC))) %>% pull(gene)
}

# =============================================================================
# Get top N genes by lowest FDR (regardless of significance)
# =============================================================================
get_top_fdr_genes <- function(res, n_top = 10) {
  res %>% filter(!grepl("^ENSG", gene)) %>%
    filter(!is.na(FDR)) %>%
    arrange(FDR, desc(abs(logFC))) %>%
    head(n_top) %>% pull(gene)
}

# =============================================================================
# HVG identification
# =============================================================================
identify_hvg <- function(expr_mat, n_top = 2000, min_mean = 0.1, span = 0.3) {
  gene_means <- colMeans(expr_mat, na.rm = TRUE)
  gene_vars <- apply(expr_mat, 2, var, na.rm = TRUE)
  
  keep <- gene_means >= min_mean & gene_vars > 0 & is.finite(gene_means) & is.finite(gene_vars)
  
  if (sum(keep) < 50) {
    gene_vars_clean <- gene_vars[is.finite(gene_vars) & gene_vars > 0]
    if (length(gene_vars_clean) == 0) return(colnames(expr_mat))
    top_var <- names(sort(gene_vars_clean, decreasing = TRUE))[1:min(n_top, length(gene_vars_clean))]
    return(top_var)
  }
  
  gene_means_filt <- gene_means[keep]
  gene_vars_filt <- gene_vars[keep]
  
  df <- data.frame(gene = names(gene_means_filt), mean_expr = gene_means_filt, var_expr = gene_vars_filt, stringsAsFactors = FALSE)
  df$log_mean <- log10(df$mean_expr + 1)
  df$log_var <- log10(df$var_expr + 1)
  df <- df[is.finite(df$log_mean) & is.finite(df$log_var), ]
  
  if (nrow(df) < 50) {
    return(names(sort(gene_vars_filt, decreasing = TRUE))[1:min(n_top, length(gene_vars_filt))])
  }
  
  fit_success <- FALSE
  for (sp in c(span, 0.5, 0.75, 1.0)) {
    fit_result <- tryCatch({
      fit <- loess(log_var ~ log_mean, data = df, span = sp)
      df$expected_var <- predict(fit, df$log_mean)
      if (all(is.na(df$expected_var))) stop("All predictions are NA")
      TRUE
    }, error = function(e) FALSE, warning = function(w) FALSE)
    
    if (fit_result && sum(!is.na(df$expected_var)) > 50) {
      fit_success <- TRUE
      break
    }
  }
  
  if (!fit_success) {
    df$cv <- sqrt(df$var_expr) / (df$mean_expr + 0.01)
    df <- df[is.finite(df$cv), ]
    hvg <- df %>% arrange(desc(cv)) %>% head(n_top) %>% pull(gene)
    return(hvg)
  }
  
  df$residual <- df$log_var - df$expected_var
  df <- df[is.finite(df$residual), ]
  
  if (nrow(df) < 10) {
    return(names(sort(gene_vars_filt, decreasing = TRUE))[1:min(n_top, length(gene_vars_filt))])
  }
  
  df %>% arrange(desc(residual)) %>% head(n_top) %>% pull(gene)
}

# =========================================================================
# PLOT FUNCTIONS
# =========================================================================
save_plot <- function(p, filepath, w = 7, h = 5) {
  dir.create(dirname(filepath), recursive = TRUE, showWarnings = FALSE)
  tryCatch({
    ggsave(filepath, p, width = w, height = h, dpi = 300)
  }, error = function(e) {
    message("Warning: Could not save plot to ", filepath, ": ", e$message)
  })
}

make_volcano <- function(res, title_str, figpath, color_map, top_n = 15) {
  vol_df <- res %>% mutate(neglog10P = -log10(PValue + 1e-300))
  sl <- levels(vol_df$DE_status)
  uc <- color_map[names(color_map) %in% sl]
  
  # Label top FDR+logFC genes
  top_strict_labels <- vol_df %>%
    filter(!grepl("^ENSG", gene), FDR < 0.05, abs(logFC) > 1) %>%
    arrange(FDR) %>% head(top_n)
  # Label top FDR-only genes
  top_fdr_only_labels <- vol_df %>%
    filter(!grepl("^ENSG", gene), DE_status == "FDR-only") %>%
    arrange(FDR) %>% head(5)
  
  p <- ggplot(vol_df, aes(x = logFC, y = neglog10P, color = DE_status)) +
    geom_point(aes(size = DE_status, alpha = DE_status)) +
    scale_color_manual(values = uc) +
    scale_size_manual(values = setNames(c(rep(2, length(sl)-1), 1), sl)) +
    scale_alpha_manual(values = setNames(c(rep(0.8, length(sl)-1), 0.3), sl)) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey30") +
    geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "grey30") +
    theme_bw(base_size = 12) + theme(panel.grid.minor = element_blank(), legend.position = "right") +
    labs(title = title_str, x = "log2 fold-change", y = "-log10(P-value)",
         color = "DE Status", size = "DE Status", alpha = "DE Status")
  
  if (nrow(top_strict_labels) > 0) {
    p <- p + ggrepel::geom_text_repel(
      data = top_strict_labels, aes(label = gene),
      size = 3, max.overlaps = 20, box.padding = 0.5,
      segment.color = "grey50", segment.size = 0.3, color = "black", fontface = "bold")
  }
  if (nrow(top_fdr_only_labels) > 0) {
    p <- p + ggrepel::geom_text_repel(
      data = top_fdr_only_labels, aes(label = gene),
      size = 2.8, max.overlaps = 15, box.padding = 0.4,
      segment.color = "grey70", segment.size = 0.2, segment.linetype = "dashed",
      color = "darkorange3", fontface = "italic")
  }
  
  save_plot(p, figpath, w = 9, h = 6)
}

make_ma_plot <- function(res, title_str, figpath, color_map) {
  ma_df <- res %>% mutate(A = AveExpr, M = logFC)
  sl <- levels(ma_df$DE_status)
  uc <- color_map[names(color_map) %in% sl]
  p <- ggplot(ma_df, aes(x = A, y = M, color = DE_status)) +
    geom_point(aes(size = DE_status, alpha = DE_status)) +
    scale_color_manual(values = uc) +
    scale_size_manual(values = setNames(c(rep(1.8, length(sl)-1), 0.8), sl)) +
    scale_alpha_manual(values = setNames(c(rep(0.7, length(sl)-1), 0.2), sl)) +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
    geom_hline(yintercept = c(-1, 1), linetype = "dashed", color = "grey30") +
    theme_bw(base_size = 12) + theme(panel.grid.minor = element_blank(), legend.position = "right") +
    labs(title = title_str, x = "Average expression", y = "log2 fold-change",
         color = "DE Status", size = "DE Status", alpha = "DE Status")
  save_plot(p, figpath, w = 9, h = 6)
}

make_pca_plot <- function(expr_mat, color_vec, color_name, color_values, title_str, figpath,
                          use_hvg = TRUE, n_hvg = 2000, label_samples = TRUE) {
  
  gene_vars <- apply(expr_mat, 2, function(x) {
    x_clean <- x[is.finite(x)]
    if (length(x_clean) < 2) return(0)
    var(x_clean, na.rm = TRUE)
  })
  
  valid_genes <- names(gene_vars)[gene_vars > 0 & is.finite(gene_vars)]
  if (length(valid_genes) < 10) return(invisible(NULL))
  
  expr_clean <- expr_mat[, valid_genes, drop = FALSE]
  for (j in 1:ncol(expr_clean)) {
    col_vals <- expr_clean[, j]
    bad_idx <- !is.finite(col_vals)
    if (any(bad_idx)) {
      col_mean <- mean(col_vals[is.finite(col_vals)], na.rm = TRUE)
      if (!is.finite(col_mean)) col_mean <- 0
      expr_clean[bad_idx, j] <- col_mean
    }
  }
  
  if (use_hvg && ncol(expr_clean) > n_hvg) {
    hvg <- tryCatch({
      identify_hvg(expr_clean, n_top = n_hvg)
    }, error = function(e) {
      gene_vars_clean <- apply(expr_clean, 2, var, na.rm = TRUE)
      names(sort(gene_vars_clean, decreasing = TRUE))[1:min(n_hvg, length(gene_vars_clean))]
    })
    
    hvg_in_mat <- intersect(hvg, colnames(expr_clean))
    if (length(hvg_in_mat) >= 50) {
      expr_for_pca <- expr_clean[, hvg_in_mat, drop = FALSE]
      hvg_note <- sprintf("Using %d HVG", length(hvg_in_mat))
    } else {
      expr_for_pca <- expr_clean
      hvg_note <- "All genes (HVG insufficient)"
    }
  } else {
    expr_for_pca <- expr_clean
    hvg_note <- sprintf("All %d genes", ncol(expr_clean))
  }
  
  final_vars <- apply(expr_for_pca, 2, var, na.rm = TRUE)
  expr_for_pca <- expr_for_pca[, final_vars > 1e-10 & is.finite(final_vars), drop = FALSE]
  if (ncol(expr_for_pca) < 10) return(invisible(NULL))
  
  pca <- tryCatch({
    prcomp(expr_for_pca, scale. = TRUE, center = TRUE)
  }, error = function(e) {
    tryCatch({ prcomp(expr_for_pca, scale. = FALSE, center = TRUE) }, error = function(e2) NULL)
  })
  
  if (is.null(pca)) return(invisible(NULL))
  
  ve <- summary(pca)$importance[2, 1:2] * 100
  pca_df <- data.frame(sample = rownames(expr_mat), PC1 = pca$x[,1], PC2 = pca$x[,2], cv = color_vec)
  
  p <- ggplot(pca_df, aes(PC1, PC2, color = cv)) + 
    geom_point(size = 4, alpha = 0.85) +
    scale_color_manual(values = color_values) + 
    theme_bw(base_size = 12) +
    theme(panel.grid.minor = element_blank()) +
    labs(title = title_str, subtitle = hvg_note,
         x = sprintf("PC1 (%.1f%%)", ve[1]), y = sprintf("PC2 (%.1f%%)", ve[2]), color = color_name)
  
  if (label_samples) {
    n_samples <- nrow(pca_df)
    max_overlaps <- ifelse(n_samples > 30, 15, 25)
    text_size <- ifelse(n_samples > 50, 2, ifelse(n_samples > 30, 2.5, 3))
    p <- p + ggrepel::geom_text_repel(aes(label = sample), size = text_size, max.overlaps = max_overlaps,
                                      box.padding = 0.3, segment.color = "grey60", segment.size = 0.2,
                                      show.legend = FALSE, min.segment.length = 0.1)
  }
  
  save_plot(p, figpath, w = 9, h = 7)
}

make_pval_hist <- function(res, title_str, figpath) {
  p <- ggplot(res, aes(x = PValue)) +
    geom_histogram(bins = 50, fill = "steelblue", color = "black", alpha = 0.7) +
    theme_bw(base_size = 12) + theme(panel.grid.minor = element_blank()) +
    labs(title = title_str, x = "P-value", y = "Frequency")
  save_plot(p, figpath, w = 7, h = 5)
}

make_de_summary_bar <- function(res, title_str, figpath, color_map) {
  res_filtered <- filter_ensg_from_df(res)
  ds <- res_filtered %>% filter(DE_status != "NS") %>% count(DE_status)
  if (nrow(ds) == 0) return(invisible(NULL))
  uc <- color_map[names(color_map) %in% ds$DE_status]
  p <- ggplot(ds, aes(x = DE_status, y = n, fill = DE_status)) +
    geom_bar(stat = "identity", alpha = 0.8) + geom_text(aes(label = n), vjust = -0.5, size = 5) +
    scale_fill_manual(values = uc) + theme_bw(base_size = 12) +
    theme(legend.position = "none", panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) +
    labs(title = title_str, subtitle = "FDR < 0.05 (FDR+logFC and FDR-only)", x = "DE Status", y = "Number of Genes") +
    ylim(0, max(ds$n) * 1.15)
  save_plot(p, figpath, w = 7, h = 5)
}

# =============================================================================
# Heatmap functions
# =============================================================================
prep_heatmap_inputs <- function(plot_mat, genes, group_vec, anno_name, anno_colors_list) {
  top_genes <- intersect(genes, rownames(plot_mat))
  if (length(top_genes) < 2) return(NULL)
  
  mat_top <- plot_mat[top_genes, , drop = FALSE]
  mat_top <- t(scale(t(mat_top)))
  mat_top[!is.finite(mat_top)] <- 0
  
  anno_df <- data.frame(V1 = group_vec, row.names = colnames(mat_top))
  colnames(anno_df) <- anno_name
  so <- order(group_vec)
  mat_top <- mat_top[, so, drop = FALSE]
  anno_df <- anno_df[so, , drop = FALSE]
  
  list(mat = mat_top, anno_df = anno_df, anno_colors = anno_colors_list)
}

render_heatmap_png <- function(hm_inputs, figpath, title_str) {
  if (is.null(hm_inputs)) return(invisible(NULL))
  
  n_genes <- nrow(hm_inputs$mat)
  dir.create(dirname(figpath), recursive = TRUE, showWarnings = FALSE)
  
  tryCatch({
    png(figpath, width = 1200, height = 800 + n_genes * 8, res = 150)
    pheatmap(hm_inputs$mat,
             annotation_col = hm_inputs$anno_df,
             annotation_colors = hm_inputs$anno_colors,
             cluster_rows = TRUE, cluster_cols = FALSE,
             show_colnames = FALSE, show_rownames = (n_genes <= 50),
             color = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
             breaks = seq(-2, 2, length.out = 101),
             fontsize_row = max(6, 10 - n_genes/10),
             main = title_str, border_color = NA)
    dev.off()
  }, error = function(e) {
    try(dev.off(), silent = TRUE)
    message("Warning: Could not render heatmap to ", figpath, ": ", e$message)
  })
}

# =============================================================================
# Unified heatmap generation (matching Step 1 two-tier pattern)
# =============================================================================
make_heatmaps <- function(plot_mat, group_vec, figdir, title_prefix,
                          sig_genes_strict, sig_genes_fdr_only, sig_genes_all,
                          anno_name = "Group", anno_colors_list = NULL,
                          gene_counts = c(10, 20, 30, 40, 50)) {
  
  n_all_sig <- length(sig_genes_all)
  
  # All significant gene heatmaps (FDR < 0.05, any logFC)
  if (n_all_sig > 0) {
    for (n_genes in gene_counts) {
      if (n_genes > n_all_sig) next
      top_genes <- head(sig_genes_all, n_genes)
      hm_inputs <- prep_heatmap_inputs(plot_mat, top_genes, group_vec, anno_name, anno_colors_list)
      if (!is.null(hm_inputs)) {
        figpath <- file.path(figdir, sprintf("heatmap_sig_%d_genes.png", n_genes))
        title_str <- sprintf("%s\nTop %d Significant DE Genes (FDR<0.05)", title_prefix, n_genes)
        render_heatmap_png(hm_inputs, figpath, title_str)
      }
    }
  }
  
  # Strict (FDR+logFC) heatmaps
  n_strict <- length(sig_genes_strict)
  if (n_strict > 0) {
    for (n_genes in gene_counts) {
      if (n_genes > n_strict) next
      top_genes <- head(sig_genes_strict, n_genes)
      hm_inputs <- prep_heatmap_inputs(plot_mat, top_genes, group_vec, anno_name, anno_colors_list)
      if (!is.null(hm_inputs)) {
        figpath <- file.path(figdir, sprintf("heatmap_strict_%d_genes.png", n_genes))
        title_str <- sprintf("%s\nTop %d Strict DE Genes (FDR<0.05 & |logFC|>1)", title_prefix, n_genes)
        render_heatmap_png(hm_inputs, figpath, title_str)
      }
    }
  }
  
  # FDR-only heatmaps
  n_fdr_only <- length(sig_genes_fdr_only)
  if (n_fdr_only > 0) {
    for (n_genes in gene_counts) {
      if (n_genes > n_fdr_only) next
      top_genes <- head(sig_genes_fdr_only, n_genes)
      hm_inputs <- prep_heatmap_inputs(plot_mat, top_genes, group_vec, anno_name, anno_colors_list)
      if (!is.null(hm_inputs)) {
        figpath <- file.path(figdir, sprintf("heatmap_fdr_only_%d_genes.png", n_genes))
        title_str <- sprintf("%s\nTop %d FDR-only DE Genes (FDR<0.05, |logFC|<=1)", title_prefix, n_genes)
        render_heatmap_png(hm_inputs, figpath, title_str)
      }
    }
  }
  
  invisible(NULL)
}

# =============================================================================
# Gene boxplots
# =============================================================================
make_gene_boxplots_batch <- function(genes, expr_mat, group_vec, res_df, outdir,
                                     group_colors_use, max_n = 100,
                                     label_type = "sig") {
  if (length(genes) == 0) return(invisible(NULL))
  
  genes <- filter_ensg_genes(genes)
  if (length(genes) == 0) return(invisible(NULL))
  
  genes <- head(genes, max_n)
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  
  future_lapply(genes, function(gene) {
    if (!gene %in% colnames(expr_mat)) return(NULL)
    df <- data.frame(sample = rownames(expr_mat), group = group_vec, value = as.numeric(expr_mat[, gene]))
    
    gene_row <- res_df %>% filter(gene == !!gene)
    pval <- if (nrow(gene_row) > 0) gene_row$PValue[1] else NA
    fdr <- if (nrow(gene_row) > 0) gene_row$FDR[1] else NA
    logfc <- if (nrow(gene_row) > 0) gene_row$logFC[1] else NA
    
    ptxt <- if (is.na(pval)) "NA" else if (pval < 1e-4) format(pval, scientific=TRUE, digits=2) else as.character(signif(pval, 2))
    
    if (label_type == "fdr_only") {
      title_main <- paste0("Expression: ", gene, "\n(FDR-only significant, |logFC|<=1)")
      subtitle_txt <- sprintf("logFC=%.2f, FDR=%.3e", ifelse(is.na(logfc), 0, logfc), ifelse(is.na(fdr), 1, fdr))
    } else if (label_type == "top_fdr") {
      title_main <- paste0("Expression: ", gene, "\n(Top FDR, not significant)")
      subtitle_txt <- sprintf("logFC=%.2f, FDR=%.3e", ifelse(is.na(logfc), 0, logfc), ifelse(is.na(fdr), 1, fdr))
    } else {
      title_main <- paste0("Expression: ", gene)
      subtitle_txt <- NULL
    }
    
    ymax <- max(df$value, na.rm=TRUE); ymin <- min(df$value, na.rm=TRUE); yrng <- ymax - ymin
    if (!is.finite(yrng) || yrng == 0) yrng <- max(abs(ymax), 1) * 0.1
    yb <- ymax + 0.12*yrng; ht <- 0.04*yrng; yt <- yb + 0.03*yrng
    
    p <- ggplot(df, aes(x = group, y = value, fill = group)) +
      geom_boxplot(outlier.shape = NA, alpha = 0.7) +
      geom_jitter(width = 0.15, alpha = 0.6, size = 2.2) +
      scale_fill_manual(values = group_colors_use) +
      theme_bw(base_size = 12) +
      theme(legend.position = "none", panel.grid.minor = element_blank()) +
      labs(title = title_main, subtitle = subtitle_txt, x = "Group", y = "Expression")
    
    if (length(unique(df$group)) == 2) {
      p <- p + annotate("segment", x=1, xend=1, y=yb-ht, yend=yb) +
        annotate("segment", x=1, xend=2, y=yb, yend=yb) +
        annotate("segment", x=2, xend=2, y=yb-ht, yend=yb) +
        annotate("text", x=1.5, y=yt, label=ptxt, size=5) +
        expand_limits(y = yt + 0.05*yrng)
    }
    
    if (label_type == "fdr_only") {
      p <- p + theme(plot.title = element_text(color = "darkorange", face = "bold"),
                     plot.subtitle = element_text(color = "grey40", face = "italic"))
    } else if (label_type == "top_fdr") {
      p <- p + theme(plot.title = element_text(color = "steelblue", face = "bold"),
                     plot.subtitle = element_text(color = "grey40", face = "italic"))
    }
    
    tryCatch({
      ggsave(file.path(outdir, paste0(gsub("[^A-Za-z0-9_.-]", "_", gene), ".png")),
             p, width=6.5, height=4.5, dpi=300)
    }, error = function(e) NULL)
    NULL
  }, future.seed = TRUE)
  
  invisible(NULL)
}

# =============================================================================
# Main plot suite (matching Step 1 two-tier pattern)
# Now: if no sig genes, plot top 10 lowest FDR boxplots
# =============================================================================
run_plot_suite <- function(res, expr_mat, group_vec, figdir, title_prefix,
                           sig05_all, sig05_strict, sig05_fdr_only,
                           sig10,
                           color_map, group_color_values,
                           anno_name = "Group", anno_colors_list = NULL,
                           do_boxplots = TRUE, max_boxplots = 100,
                           heatmap_gene_counts = c(10, 20, 30, 40, 50),
                           use_hvg_for_pca = TRUE, n_hvg = 2000) {
  
  dir.create(figdir, recursive = TRUE, showWarnings = FALSE)
  plot_mat <- t(expr_mat)
  has_any_sig <- length(sig05_all) > 0
  
  # Core plots
  make_volcano(res, paste0(title_prefix, " - Volcano"), file.path(figdir, "volcano.png"),
               color_map, top_n = 15)
  make_ma_plot(res, paste0(title_prefix, " - MA"), file.path(figdir, "ma_plot.png"), color_map)
  make_pca_plot(expr_mat, group_vec, anno_name, group_color_values,
                paste0(title_prefix, " - PCA"), file.path(figdir, "pca.png"),
                use_hvg = use_hvg_for_pca, n_hvg = n_hvg, label_samples = TRUE)
  make_pval_hist(res, paste0(title_prefix, " - P-value Dist"), file.path(figdir, "pvalue_distribution.png"))
  
  # Heatmaps (two-tier)
  make_heatmaps(plot_mat, group_vec, figdir, title_prefix,
                sig_genes_strict = sig05_strict, sig_genes_fdr_only = sig05_fdr_only,
                sig_genes_all = sig05_all,
                anno_name = anno_name, anno_colors_list = anno_colors_list,
                gene_counts = heatmap_gene_counts)
  
  if (has_any_sig) {
    make_de_summary_bar(res, paste0(title_prefix, " - DE Summary"),
                        file.path(figdir, "de_summary_barplot.png"), color_map)
    
    if (do_boxplots) {
      # Boxplots for strict significant genes
      if (length(sig05_strict) > 0) {
        make_gene_boxplots_batch(sig05_strict, expr_mat, group_vec, res,
                                file.path(figdir, "boxplots_FDR_logFC_0.05"),
                                group_colors_use = group_color_values,
                                max_n = max_boxplots, label_type = "sig")
      }
      # Boxplots for FDR-only significant genes
      if (length(sig05_fdr_only) > 0) {
        make_gene_boxplots_batch(sig05_fdr_only, expr_mat, group_vec, res,
                                file.path(figdir, "boxplots_FDR_only_0.05"),
                                group_colors_use = group_color_values,
                                max_n = max_boxplots, label_type = "fdr_only")
      }
      # FDR 0.10 all
      if (length(sig10) > 0) {
        make_gene_boxplots_batch(sig10, expr_mat, group_vec, res,
                                file.path(figdir, "boxplots_FDR_0.10"),
                                group_colors_use = group_color_values,
                                max_n = max_boxplots, label_type = "sig")
      }
    }
  } else {
    message("    -> No significant genes (FDR<0.05). Plotting top 10 lowest FDR genes.")
    if (do_boxplots) {
      top_fdr_genes <- get_top_fdr_genes(res, n_top = 10)
      if (length(top_fdr_genes) > 0) {
        make_gene_boxplots_batch(top_fdr_genes, expr_mat, group_vec, res,
                                file.path(figdir, "boxplots_top10_lowest_FDR"),
                                group_colors_use = group_color_values,
                                max_n = 10, label_type = "top_fdr")
      }
    }
  }
}

# =============================================================================
# Save DE results (matching Step 1 two-tier pattern)
# DE_results.csv now sorted by lowest FDR
# =============================================================================
save_de_results <- function(res, outdir, method_label, n_samples, n_group_table,
                            cov_label, sig05_all, sig05_strict, sig05_fdr_only,
                            sig10, extra_lines = NULL) {
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  
  # Sort by lowest FDR first, then by descending abs(logFC)
  res_sorted <- res %>% arrange(FDR, desc(abs(logFC)))
  fwrite(res_sorted, file.path(outdir, "DE_results.csv"))
  
  writeLines(sig05_all, file.path(outdir, "significant_genes_FDR_0.05_all.txt"))
  writeLines(sig05_strict, file.path(outdir, "significant_genes_FDR_0.05_and_logFC.txt"))
  writeLines(sig05_fdr_only, file.path(outdir, "significant_genes_FDR_0.05_only.txt"))
  writeLines(sig10, file.path(outdir, "significant_genes_FDR_0.10.txt"))
  
  sl <- c(paste0("Method: limma"),
          paste0("Comparison: ", method_label),
          paste0("Samples total: ", n_samples),
          paste0("Genes tested: ", nrow(res)),
          paste0("Group counts: ", paste(names(n_group_table), as.integer(n_group_table), sep="=", collapse=", ")),
          paste0("Covariate adjustment: ", cov_label),
          "",
          "--- Two-tier significance (FDR < 0.05) ---",
          sprintf("Total significant (FDR<0.05): %d", length(sig05_all)),
          sprintf("  FDR + logFC significant (FDR<0.05 & |logFC|>1): %d", length(sig05_strict)),
          sprintf("  FDR-only significant (FDR<0.05 & |logFC|<=1): %d", length(sig05_fdr_only)),
          "",
          sprintf("Significant (FDR<0.10): %d", length(sig10)))
  
  if (!is.null(extra_lines)) sl <- c(sl, extra_lines)
  writeLines(sl, file.path(outdir, "summary.txt"))
}

# =============================================================================
# STRATIFICATION CONFIGURATION
# =============================================================================
get_stratification_config <- function(strat_type) {
  configs <- list(
    "LC_Recovered" = list(
      name = "LC_Recovered",
      display_name = "LC/Recovered Status",
      extract_fn = extract_group,
      levels = c("LC", "Recovered"),
      colors = GROUP_COLORS,
      exclude_from_cov = c("LC/Recovered", "LC.Recovered"),
      subdir = "by_LC_status"
    ),
    "BMI" = list(
      name = "BMI",
      display_name = "BMI Category",
      extract_fn = extract_bmi_group,
      levels = c("Normal", "Elevated"),
      colors = BMI_COLORS,
      exclude_from_cov = c("BMI category", "BMI_category", "BMI.category"),
      subdir = "by_BMI"
    ),
    "Sex" = list(
      name = "Sex",
      display_name = "Sex",
      extract_fn = extract_sex_group,
      levels = c("Female", "Male"),
      colors = SEX_COLORS,
      exclude_from_cov = c("Sex", "sex", "Gender", "gender"),
      subdir = "by_Sex"
    ),
    "Age" = list(
      name = "Age",
      display_name = "Age Group",
      extract_fn = extract_age_group,
      levels = c("Young", "Middle", "Old"),
      colors = AGE_COLORS,
      exclude_from_cov = c("age_cluster", "Age_cluster", "ageCluster", "AgeCluster", "age.group", "Age group"),
      subdir = "by_Age"
    )
  )
  
  if (!strat_type %in% names(configs)) {
    stop("Unknown stratification type: ", strat_type,
         "\nAvailable: ", paste(names(configs), collapse = ", "))
  }
  
  configs[[strat_type]]
}

# =============================================================================
# PROPORTION ANALYSIS: Run for one timepoint pair within a stratum
# (Matching Step 1's run_proportion_analysis_one_tp pattern)
# =============================================================================
run_proportion_analysis_one_pair <- function(parent_dir, tp1, tp2, grp_name, config,
                                             batch_keys, output_dir, strict_batch_control,
                                             min_samples_per_tp, min_cov_per_level, min_resid_df) {
  
  tp1s <- tp_short(tp1)
  tp2s <- tp_short(tp2)
  comparison_name <- paste0(grp_name, "_", tp2s, "_vs_", tp1s)
  
  # Make skip info helper
  make_skip <- function(reason, failed_cov = character(0), n_samp = "NA", grp_counts = "NA") {
    list(success = FALSE, de_result = NULL,
         skipped_info = list(
           timepoint_early = tp1, timepoint_late = tp2,
           celltype = "ALL (proportion)", stratum = grp_name,
           reason = reason, failed_covariates = failed_cov,
           n_samples = n_samp, group_counts = grp_counts))
  }
  
  # Load proportion data from each timepoint
  prop_file1 <- file.path(parent_dir, tp1, "celltype_proportions.csv")
  prop_file2 <- file.path(parent_dir, tp2, "celltype_proportions.csv")
  
  if (!file.exists(prop_file1) || !file.exists(prop_file2)) {
    return(make_skip("celltype_proportions.csv not found at one or both timepoints"))
  }
  
  prop_data1 <- load_proportion_data(prop_file1)
  prop_data2 <- load_proportion_data(prop_file2)
  
  if (is.null(prop_data1) || is.null(prop_data2)) {
    return(make_skip("Failed to load proportion data or insufficient samples"))
  }
  
  # Load metadata for each timepoint (matching Step 1 pattern)
  meta1 <- load_proportion_metadata(parent_dir, tp1, rownames(prop_data1$prop_mat))
  meta2 <- load_proportion_metadata(parent_dir, tp2, rownames(prop_data2$prop_mat))
  
  if (is.null(meta1) || is.null(meta2)) {
    return(make_skip("Could not find matching metadata for proportion samples"))
  }
  
  # Align samples with metadata
  common1 <- intersect(rownames(prop_data1$prop_mat), rownames(meta1))
  common2 <- intersect(rownames(prop_data2$prop_mat), rownames(meta2))
  
  if (length(common1) < min_samples_per_tp || length(common2) < min_samples_per_tp) {
    return(make_skip(sprintf("Insufficient overlapping samples (tp1=%d, tp2=%d)", length(common1), length(common2))))
  }
  
  prop_mat1 <- prop_data1$prop_mat[common1, , drop = FALSE]
  prop_mat2 <- prop_data2$prop_mat[common2, , drop = FALSE]
  meta1 <- meta1[common1, , drop = FALSE]
  meta2 <- meta2[common2, , drop = FALSE]
  
  # Extract stratification group and filter to stratum
  group1 <- config$extract_fn(meta1)
  group2 <- config$extract_fn(meta2)
  
  if (is.null(group1) || is.null(group2)) {
    return(make_skip(paste0("Could not extract ", config$name, " from metadata")))
  }
  
  idx1 <- which(group1 == grp_name)
  idx2 <- which(group2 == grp_name)
  
  if (length(idx1) < min_samples_per_tp || length(idx2) < min_samples_per_tp) {
    return(make_skip(sprintf("Insufficient samples in stratum %s (tp1=%d, tp2=%d)",
                             grp_name, length(idx1), length(idx2))))
  }
  
  prop_mat1 <- prop_mat1[idx1, , drop = FALSE]
  prop_mat2 <- prop_mat2[idx2, , drop = FALSE]
  meta1 <- meta1[idx1, , drop = FALSE]
  meta2 <- meta2[idx2, , drop = FALSE]
  
  # Find common cell types
  common_cts <- intersect(colnames(prop_mat1), colnames(prop_mat2))
  if (length(common_cts) < 2) {
    return(make_skip(sprintf("Insufficient common cell types (%d)", length(common_cts))))
  }
  
  prop_mat1 <- prop_mat1[, common_cts, drop = FALSE]
  prop_mat2 <- prop_mat2[, common_cts, drop = FALSE]
  
  # Unique sample names and combine
  rownames(prop_mat1) <- paste0(tp1s, "_", rownames(prop_mat1))
  rownames(prop_mat2) <- paste0(tp2s, "_", rownames(prop_mat2))
  rownames(meta1) <- paste0(tp1s, "_", rownames(meta1))
  rownames(meta2) <- paste0(tp2s, "_", rownames(meta2))
  
  combined_prop <- rbind(prop_mat1, prop_mat2)
  combined_meta <- rbind(meta1, meta2)
  tf <- factor(c(rep(tp1, nrow(prop_mat1)), rep(tp2, nrow(prop_mat2))), levels = c(tp1, tp2))
  
  tp_counts <- table(tf)
  
  # Strict batch covariate check (matching Step 1 pattern)
  cov_keys_use <- batch_keys[!batch_keys %in% config$exclude_from_cov]
  
  if (isTRUE(strict_batch_control)) {
    batch_check <- check_all_batch_covariates(
      group = tf, meta_df = combined_meta, batch_keys = cov_keys_use,
      exclude_keys = config$exclude_from_cov, min_per_level = min_cov_per_level)
    
    if (!batch_check$can_model) {
      return(make_skip(
        "Cannot control for all batch covariates (proportions)",
        failed_cov = batch_check$failed_covariates,
        n_samp = nrow(combined_prop),
        grp_counts = paste(names(tp_counts), as.integer(tp_counts), sep="=", collapse=", ")))
    }
  }
  
  # Prepare covariates
  cov_df <- prepare_covariates(combined_meta, cov_keys_use, exclude_keys = config$exclude_from_cov)
  
  filt <- apply_covariate_filter(combined_prop, tf, cov_df)
  if (is.null(filt)) {
    return(make_skip("Covariate filtering removed too many samples (proportions)",
                     n_samp = nrow(combined_prop),
                     grp_counts = paste(names(tp_counts), as.integer(tp_counts), sep="=", collapse=", ")))
  }
  
  combined_prop <- filt$expr
  tf <- filt$group
  cov_df <- filt$cov
  
  tp_counts <- table(tf)
  if (any(tp_counts < min_samples_per_tp)) {
    return(make_skip("Insufficient samples per timepoint after covariate filtering (proportions)",
                     n_samp = nrow(combined_prop),
                     grp_counts = paste(names(tp_counts), as.integer(tp_counts), sep="=", collapse=", ")))
  }
  
  # Build design
  ds <- build_safe_design(tf, cov_df, min_resid_df = min_resid_df)
  design <- ds$design
  cov_used <- ds$cov_used
  
  if (is.null(design)) {
    return(make_skip("Design matrix build failed (proportions)",
                     n_samp = nrow(combined_prop),
                     grp_counts = paste(names(tp_counts), as.integer(tp_counts), sep="=", collapse=", ")))
  }
  
  # Find design columns for contrast
  col1 <- find_design_col(design, "group", tp1)
  col2 <- find_design_col(design, "group", tp2)
  
  if (is.null(col1) || is.null(col2)) {
    return(make_skip(sprintf("Cannot find design columns for %s and %s (proportions)", tp1, tp2)))
  }
  
  # Output directory
  prop_outdir <- file.path(output_dir, "step2_proportion_analysis", config$subdir, comparison_name)
  dir.create(prop_outdir, recursive = TRUE, showWarnings = FALSE)
  
  # Log covariate decisions
  writeLines(c(
    paste0("Timepoint comparison: ", tp2, " vs ", tp1),
    paste0("Stratum: ", grp_name),
    paste0("Analysis: Cell type proportions"),
    paste0("Stratification: ", config$name),
    paste0("N samples: ", nrow(combined_prop)),
    paste0("N cell types: ", ncol(combined_prop)),
    paste0("Cell types: ", paste(colnames(combined_prop), collapse = ", ")),
    paste0("Sample counts: ", paste(names(tp_counts), as.integer(tp_counts), sep="=", collapse=", ")),
    "", "Covariate notes:", ds$notes, "",
    paste0("Covariates used: ", if (!is.null(cov_used) && ncol(cov_used) > 0) paste(colnames(cov_used), collapse = ", ") else "OFF")
  ), file.path(prop_outdir, "COVARIATE_DECISIONS.txt"))
  
  # Run limma (using the same run_limma as gene DE, with prop_mat as "expression")
  contrast_formula <- paste0(col2, " - ", col1)
  contrast <- makeContrasts(contrasts = contrast_formula, levels = design)
  colnames(contrast) <- "tp_diff"
  
  res <- tryCatch(run_limma(combined_prop, design, contrast, "tp_diff"), error = function(e) NULL)
  if (is.null(res)) {
    return(make_skip("Limma failed for proportion analysis"))
  }
  
  up_lab <- paste0("Up at ", tp2s)
  dn_lab <- paste0("Down at ", tp2s)
  res <- add_de_status_generic(res, up_label = up_lab, down_label = dn_lab)
  
  # Two-tier significance (uses gene column which stores cell type names)
  sig05_all <- get_sig_genes(res, fdr_thresh = 0.05)
  sig05_strict <- get_sig_genes_strict(res, fdr_thresh = 0.05, logfc_thresh = 1)
  sig05_fdr_only <- get_sig_genes_fdr_only(res, fdr_thresh = 0.05, logfc_thresh = 1)
  sig10 <- get_sig_genes(res, fdr_thresh = 0.10)
  
  cov_label <- if (!is.null(cov_used) && ncol(cov_used) > 0) paste(colnames(cov_used), collapse = ", ") else "OFF"
  
  save_de_results(
    res, prop_outdir,
    paste0("Proportions | ", config$display_name, ": ", grp_name, " | ", tp2, " vs ", tp1),
    nrow(combined_prop), table(tf), cov_label,
    sig05_all, sig05_strict, sig05_fdr_only, sig10)
  
  # Figures
  figdir <- file.path(prop_outdir, "figures")
  dir.create(figdir, recursive = TRUE, showWarnings = FALSE)
  
  color_map <- setNames(c("#E64B35", "#4DBBD5", "#FFA500", "grey70"),
                        c(up_lab, dn_lab, "FDR-only", "NS"))
  tp_colors <- setNames(c("#4DBBD5", "#E64B35"), c(tp1, tp2))
  anno_colors_list <- list(Timepoint = tp_colors)
  
  # Volcano
  make_volcano(res, paste0("Proportions (", grp_name, ") ", tp2s, " vs ", tp1s, " - Volcano"),
               file.path(figdir, "volcano.png"), color_map)
  # P-value histogram
  make_pval_hist(res, paste0("Proportions (", grp_name, ") - P-value Dist"),
                 file.path(figdir, "pvalue_distribution.png"))
  # DE summary bar
  if (length(sig05_all) > 0) {
    make_de_summary_bar(res, paste0("Proportions (", grp_name, ") - DE Summary"),
                        file.path(figdir, "de_summary_barplot.png"), color_map)
  }
  
  # Boxplots for all cell types
  make_gene_boxplots_batch(
    colnames(combined_prop), combined_prop, tf, res,
    file.path(figdir, "boxplots_all_celltypes"),
    group_colors_use = tp_colors,
    max_n = ncol(combined_prop), label_type = "sig")
  
  # Heatmap of all proportions
  plot_mat <- t(combined_prop)
  if (ncol(combined_prop) >= 2) {
    hm_inputs <- prep_heatmap_inputs(plot_mat, colnames(combined_prop), tf, "Timepoint", anno_colors_list)
    if (!is.null(hm_inputs)) {
      render_heatmap_png(hm_inputs, file.path(figdir, "heatmap_all_proportions.png"),
                         paste0("Cell Type Proportions (", grp_name, ") ", tp2s, " vs ", tp1s))
    }
  }
  
  res_out <- res %>%
    mutate(timepoint_early = tp1, timepoint_late = tp2,
           stratification = config$name, stratum = grp_name,
           analysis_type = "proportion") %>%
    select(analysis_type, stratification, stratum, timepoint_early, timepoint_late, everything())
  
  list(success = TRUE, de_result = res_out, skipped_info = NULL)
}

# =============================================================================
# Copy boxplots to a common folder, organized by stratification category
# =============================================================================
collect_boxplots_to_common_folder <- function(output_dir, config) {
  common_boxplot_dir <- file.path(output_dir, "summary", "all_boxplots", config$subdir)
  dir.create(common_boxplot_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Search for all boxplot PNGs under step2_across_timepoint/<subdir>
  step2_dir <- file.path(output_dir, "step2_across_timepoint", config$subdir)
  if (!dir.exists(step2_dir)) return(invisible(0))
  
  boxplot_files <- list.files(step2_dir, pattern = "\\.png$", recursive = TRUE, full.names = TRUE)
  boxplot_files <- boxplot_files[grepl("boxplots_", boxplot_files)]
  # Exclude top10 lowest FDR boxplots (those are NOT significant, only for local reference)
  boxplot_files <- boxplot_files[!grepl("boxplots_top10_lowest_FDR", boxplot_files)]
  
  if (length(boxplot_files) == 0) return(invisible(0))
  
  n_copied <- 0
  for (f in boxplot_files) {
    # Extract context from path: celltype / stratum_comparison / figures / boxplots_* / gene.png
    rel_path <- sub(paste0("^", normalizePath(step2_dir, mustWork = FALSE), "/?"), "", normalizePath(f, mustWork = FALSE))
    path_parts <- strsplit(rel_path, .Platform$file.sep)[[1]]
    
    # Build a descriptive prefix from path components (celltype, stratum_comparison, boxplot_type)
    # Typical: CellType/Stratum_M3_vs_M0/figures/boxplots_FDR_logFC_0.05/GENE.png
    ct_name <- if (length(path_parts) >= 1) path_parts[1] else "unknown"
    comparison <- if (length(path_parts) >= 2) path_parts[2] else "unknown"
    boxplot_type_dir <- if (length(path_parts) >= 4) path_parts[length(path_parts) - 1] else "boxplots"
    gene_file <- basename(f)
    
    # Create descriptive filename: CellType__Stratum_Comparison__gene.png
    new_name <- paste0(ct_name, "__", comparison, "__", gene_file)
    new_name <- gsub("[^A-Za-z0-9_.-]", "_", new_name)
    
    dest <- file.path(common_boxplot_dir, new_name)
    tryCatch({
      file.copy(f, dest, overwrite = TRUE)
      n_copied <- n_copied + 1
    }, error = function(e) NULL)
  }
  
  # Also collect proportion boxplots
  prop_dir <- file.path(output_dir, "step2_proportion_analysis", config$subdir)
  if (dir.exists(prop_dir)) {
    prop_boxplots <- list.files(prop_dir, pattern = "\\.png$", recursive = TRUE, full.names = TRUE)
    prop_boxplots <- prop_boxplots[grepl("boxplots_", prop_boxplots)]
    
    for (f in prop_boxplots) {
      rel_path <- sub(paste0("^", normalizePath(prop_dir, mustWork = FALSE), "/?"), "", normalizePath(f, mustWork = FALSE))
      path_parts <- strsplit(rel_path, .Platform$file.sep)[[1]]
      comparison <- if (length(path_parts) >= 1) path_parts[1] else "unknown"
      gene_file <- basename(f)
      
      new_name <- paste0("PROPORTION__", comparison, "__", gene_file)
      new_name <- gsub("[^A-Za-z0-9_.-]", "_", new_name)
      
      dest <- file.path(common_boxplot_dir, new_name)
      tryCatch({
        file.copy(f, dest, overwrite = TRUE)
        n_copied <- n_copied + 1
      }, error = function(e) NULL)
    }
  }
  
  message(sprintf("  -> Copied %d boxplots to common folder: %s", n_copied, common_boxplot_dir))
  invisible(n_copied)
}

# =============================================================================
# Comprehensive summary generator for Step 2 (with proportion)
# Now includes: all sig gene names listed, boxplot collection
# =============================================================================
generate_group_summary <- function(all_results, skipped_list, config, struc, summary_dir,
                                   proportion_results = NULL, proportion_skipped = NULL,
                                   output_dir = NULL) {
  group_summary_dir <- file.path(summary_dir, config$subdir)
  dir.create(group_summary_dir, recursive = TRUE, showWarnings = FALSE)
  figures_dir <- file.path(group_summary_dir, "figures")
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Combined DE results (sorted by FDR)
  if (length(all_results) > 0) {
    combined_df <- do.call(rbind, all_results)
    combined_df_sorted <- combined_df %>%
      arrange(FDR, desc(abs(logFC)))
    fwrite(combined_df_sorted, file.path(group_summary_dir, "all_DE_results_combined.csv"))
  } else {
    combined_df <- data.frame()
  }
  
  # Combined proportion results (sorted by FDR)
  if (!is.null(proportion_results) && length(proportion_results) > 0) {
    combined_prop_df <- do.call(rbind, proportion_results)
    combined_prop_sorted <- combined_prop_df %>%
      arrange(FDR, desc(abs(logFC)))
    fwrite(combined_prop_sorted, file.path(group_summary_dir, "all_proportion_results_combined.csv"))
  } else {
    combined_prop_df <- data.frame()
  }
  
  # Skipped analyses summary
  if (length(skipped_list) > 0) {
    skipped_df <- do.call(rbind, lapply(skipped_list, function(x) {
      data.frame(
        celltype = x$celltype,
        stratum = x$stratum,
        timepoint_early = x$timepoint_early,
        timepoint_late = x$timepoint_late,
        reason = x$reason,
        failed_covariates = paste(x$failed_covariates, collapse = ", "),
        n_samples = as.character(x$n_samples),
        stringsAsFactors = FALSE
      )
    }))
  } else {
    skipped_df <- data.frame(
      celltype = character(0), stratum = character(0),
      timepoint_early = character(0), timepoint_late = character(0),
      reason = character(0), failed_covariates = character(0),
      n_samples = character(0), stringsAsFactors = FALSE
    )
  }
  
  # Proportion skipped
  if (!is.null(proportion_skipped) && length(proportion_skipped) > 0) {
    prop_skipped_df <- do.call(rbind, lapply(proportion_skipped, function(x) {
      data.frame(
        celltype = x$celltype,
        stratum = x$stratum,
        timepoint_early = x$timepoint_early,
        timepoint_late = x$timepoint_late,
        reason = x$reason,
        failed_covariates = paste(x$failed_covariates, collapse = ", "),
        n_samples = as.character(x$n_samples),
        stringsAsFactors = FALSE
      )
    }))
    skipped_df <- rbind(skipped_df, prop_skipped_df)
  }
  
  if (nrow(skipped_df) > 0) {
    fwrite(skipped_df, file.path(group_summary_dir, "skipped_analyses.csv"))
    
    skip_txt <- c(
      paste(rep("=", 70), collapse = ""),
      paste0("SKIPPED ANALYSES SUMMARY: ", config$display_name),
      paste(rep("=", 70), collapse = ""),
      "",
      sprintf("Total skipped: %d comparisons", nrow(skipped_df)),
      "",
      paste(rep("-", 70), collapse = ""),
      "DETAILS:",
      paste(rep("-", 70), collapse = ""),
      ""
    )
    
    for (i in 1:min(50, nrow(skipped_df))) {
      skip_txt <- c(skip_txt,
                    sprintf("[%d] %s | %s | %s vs %s",
                            i, skipped_df$celltype[i], skipped_df$stratum[i],
                            skipped_df$timepoint_late[i], skipped_df$timepoint_early[i]),
                    sprintf("    Reason: %s", skipped_df$reason[i]),
                    sprintf("    Failed covariates: %s", skipped_df$failed_covariates[i]),
                    ""
      )
    }
    if (nrow(skipped_df) > 50) {
      skip_txt <- c(skip_txt, sprintf("... and %d more", nrow(skipped_df) - 50))
    }
    
    writeLines(skip_txt, file.path(group_summary_dir, "skipped_analyses_summary.txt"))
  } else {
    writeLines(c("No analyses were skipped.", "All comparisons passed covariate checks."),
               file.path(group_summary_dir, "skipped_analyses_summary.txt"))
  }
  
  if (nrow(combined_df) == 0 && nrow(combined_prop_df) == 0) {
    writeLines(c(paste0("No results available for ", config$display_name),
                 "All analyses may have been skipped."),
               file.path(group_summary_dir, "analysis_summary.txt"))
    return(invisible(NULL))
  }
  
  # Summary statistics
  if (nrow(combined_df) > 0) {
    sig_summary <- combined_df %>%
      filter(!grepl("^ENSG", gene)) %>%
      mutate(comparison = paste0(tp_short(timepoint_late), " vs ", tp_short(timepoint_early))) %>%
      group_by(stratum, comparison, celltype) %>%
      summarise(
        n_genes_tested = n(),
        n_sig_05_all = sum(FDR < 0.05, na.rm = TRUE),
        n_sig_05_strict = sum(FDR < 0.05 & abs(logFC) > 1, na.rm = TRUE),
        n_sig_05_fdr_only = sum(FDR < 0.05 & abs(logFC) <= 1, na.rm = TRUE),
        n_up = sum(FDR < 0.05 & logFC > 1, na.rm = TRUE),
        n_down = sum(FDR < 0.05 & logFC < -1, na.rm = TRUE),
        mean_abs_logFC = mean(abs(logFC), na.rm = TRUE),
        .groups = "drop"
      )
    
    fwrite(sig_summary, file.path(group_summary_dir, "DE_summary_by_stratum_comparison_celltype.csv"))
    
    total_sig_all <- sum(combined_df$FDR < 0.05 & !grepl("^ENSG", combined_df$gene), na.rm = TRUE)
    total_sig_strict <- sum(combined_df$FDR < 0.05 & abs(combined_df$logFC) > 1 & !grepl("^ENSG", combined_df$gene), na.rm = TRUE)
    total_sig_fdr_only <- sum(combined_df$FDR < 0.05 & abs(combined_df$logFC) <= 1 & !grepl("^ENSG", combined_df$gene), na.rm = TRUE)
    total_up <- sum(combined_df$FDR < 0.05 & combined_df$logFC > 1 & !grepl("^ENSG", combined_df$gene), na.rm = TRUE)
    total_down <- sum(combined_df$FDR < 0.05 & combined_df$logFC < -1 & !grepl("^ENSG", combined_df$gene), na.rm = TRUE)
    unique_sig_genes <- combined_df %>% filter(FDR < 0.05, !grepl("^ENSG", gene)) %>% pull(gene) %>% unique()
  } else {
    sig_summary <- data.frame()
    total_sig_all <- 0; total_sig_strict <- 0; total_sig_fdr_only <- 0
    total_up <- 0; total_down <- 0; unique_sig_genes <- character(0)
  }
  
  # ==========================================================================
  # PLOTS
  # ==========================================================================
  
  if (nrow(combined_df) > 0 && nrow(sig_summary) > 0) {
    # 1. Heatmap of significant gene counts per stratum
    for (grp in config$levels) {
      grp_data <- sig_summary %>% filter(stratum == grp)
      if (nrow(grp_data) == 0) next
      
      heatmap_df <- grp_data %>%
        select(comparison, celltype, n_sig_05_all) %>%
        pivot_wider(names_from = comparison, values_from = n_sig_05_all, values_fill = 0)
      
      if (nrow(heatmap_df) > 1 && ncol(heatmap_df) > 2) {
        heatmap_mat <- as.matrix(heatmap_df[, -1])
        rownames(heatmap_mat) <- heatmap_df$celltype
        
        if (any(heatmap_mat > 0)) {
          tryCatch({
            png(file.path(figures_dir, paste0("heatmap_sig_genes_", grp, ".png")),
                width = max(800, 100 + ncol(heatmap_mat) * 80),
                height = max(600, 100 + nrow(heatmap_mat) * 20), res = 150)
            
            max_val <- max(heatmap_mat, na.rm = TRUE)
            if (max_val == 0) max_val <- 1
            color_breaks <- seq(0, max_val, length.out = 101)
            
            grp_color <- config$colors[grp]
            if (is.na(grp_color)) grp_color <- "#3C5488"
            
            pheatmap(heatmap_mat,
                     cluster_rows = (nrow(heatmap_mat) > 1 && any(apply(heatmap_mat, 1, var) > 0)),
                     cluster_cols = FALSE,
                     color = colorRampPalette(c("white", "lightyellow", grp_color))(100),
                     breaks = color_breaks,
                     main = paste0(config$display_name, ": ", grp, "\nSignificant Genes (FDR < 0.05, all)"),
                     fontsize_row = 8, fontsize_col = 10,
                     display_numbers = TRUE, number_format = "%.0f", number_color = "black",
                     border_color = "grey80")
            dev.off()
          }, error = function(e) {
            try(dev.off(), silent = TRUE)
          })
        }
      }
    }
    
    # 2. Stacked bar: FDR+logFC vs FDR-only by stratum
    stacked_data <- sig_summary %>%
      group_by(stratum) %>%
      summarise(total_sig_strict = sum(n_sig_05_strict), total_sig_fdr_only = sum(n_sig_05_fdr_only), .groups = "drop") %>%
      pivot_longer(cols = c(total_sig_strict, total_sig_fdr_only),
                   names_to = "tier", values_to = "count") %>%
      mutate(tier = ifelse(tier == "total_sig_strict", "FDR + logFC", "FDR-only"))
    
    if (sum(stacked_data$count) > 0) {
      p_stacked <- ggplot(stacked_data, aes(x = stratum, y = count, fill = tier)) +
        geom_bar(stat = "identity", alpha = 0.85) +
        geom_text(aes(label = count), position = position_stack(vjust = 0.5), size = 4) +
        scale_fill_manual(values = c("FDR + logFC" = "#E64B35", "FDR-only" = "#FFA500")) +
        theme_bw(base_size = 12) +
        theme(panel.grid.major.x = element_blank()) +
        labs(title = paste0("Significant Genes by Tier: ", config$display_name),
             subtitle = "FDR < 0.05",
             x = config$display_name, y = "Number of Genes", fill = "Significance Tier")
      save_plot(p_stacked, file.path(figures_dir, "barplot_sig_tiers_by_stratum.png"), w = 8, h = 6)
    }
    
    # 3. Total significant genes by stratum
    total_by_stratum <- sig_summary %>%
      group_by(stratum) %>%
      summarise(total_sig = sum(n_sig_05_all), total_up = sum(n_up), total_down = sum(n_down), .groups = "drop")
    
    if (sum(total_by_stratum$total_sig) > 0) {
      p_total <- ggplot(total_by_stratum, aes(x = stratum, y = total_sig, fill = stratum)) +
        geom_bar(stat = "identity", alpha = 0.8, width = 0.6) +
        geom_text(aes(label = total_sig), vjust = -0.5, size = 5, fontface = "bold") +
        scale_fill_manual(values = config$colors) +
        theme_bw(base_size = 12) +
        theme(legend.position = "none", panel.grid.major.x = element_blank()) +
        labs(title = paste0("Total Significant DE Genes by ", config$display_name),
             subtitle = "FDR < 0.05 (all tiers), across all comparisons and cell types",
             x = config$display_name, y = "Total Significant Genes") +
        ylim(0, max(total_by_stratum$total_sig, 1) * 1.15)
      save_plot(p_total, file.path(figures_dir, "barplot_total_sig_by_stratum.png"), w = 7, h = 5)
    }
  }
  
  # ==========================================================================
  # Proportion summary plots
  # ==========================================================================
  if (nrow(combined_prop_df) > 0) {
    prop_fig_dir <- file.path(figures_dir, "proportions")
    dir.create(prop_fig_dir, recursive = TRUE, showWarnings = FALSE)
    
    # Volcano-like plot for proportions (all comparisons)
    prop_vol <- combined_prop_df %>%
      mutate(neglog10P = -log10(PValue + 1e-300),
             comparison = paste0(tp_short(timepoint_late), " vs ", tp_short(timepoint_early)),
             sig_tier = case_when(
               FDR < 0.05 & abs(logFC) > 1 ~ "FDR + logFC",
               FDR < 0.05 ~ "FDR-only",
               TRUE ~ "NS"))
    
    p_prop_vol <- ggplot(prop_vol, aes(x = logFC, y = neglog10P, color = sig_tier)) +
      geom_point(size = 3, alpha = 0.7) +
      geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey30") +
      geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "grey30") +
      ggrepel::geom_text_repel(
        data = prop_vol %>% filter(FDR < 0.05),
        aes(label = paste0(gene, " (", stratum, ")")),
        size = 2.5, max.overlaps = 20, color = "black") +
      scale_color_manual(values = c("FDR + logFC" = "#E64B35", "FDR-only" = "#FFA500", "NS" = "grey70")) +
      facet_wrap(~ comparison) +
      theme_bw(base_size = 12) +
      labs(title = paste0(config$display_name, " - Differential Cell Type Proportions"),
           x = "log2 Fold Change", y = "-log10(P-value)", color = "Significance")
    save_plot(p_prop_vol, file.path(prop_fig_dir, "volcano_proportions.png"), w = 14, h = 8)
    
    # Bar plot of significant proportions
    prop_sig <- combined_prop_df %>% filter(FDR < 0.05)
    if (nrow(prop_sig) > 0) {
      prop_sig <- prop_sig %>%
        mutate(comparison = paste0(tp_short(timepoint_late), " vs ", tp_short(timepoint_early)))
      
      p_prop_bar <- ggplot(prop_sig, aes(x = reorder(gene, logFC), y = logFC, fill = logFC > 0)) +
        geom_bar(stat = "identity", alpha = 0.8) +
        coord_flip() +
        facet_wrap(~ paste0(stratum, " | ", comparison), scales = "free_y") +
        theme_bw(base_size = 12) +
        theme(legend.position = "none") +
        labs(title = paste0(config$display_name, " - Significant Differential Proportions"),
             subtitle = "FDR < 0.05",
             x = "Cell Type", y = "log2 Fold Change") +
        scale_fill_manual(values = c("TRUE" = "#E64B35", "FALSE" = "#4DBBD5"))
      save_plot(p_prop_bar, file.path(prop_fig_dir, "barplot_sig_proportions.png"), w = 14, h = 8)
    }
  }
  
  # ==========================================================================
  # Collect boxplots to common folder
  # ==========================================================================
  if (!is.null(output_dir)) {
    collect_boxplots_to_common_folder(output_dir, config)
  }
  
  # ==========================================================================
  # Summary text file (now includes all sig gene names)
  # ==========================================================================
  summary_txt <- c(
    paste(rep("=", 70), collapse = ""),
    paste0("STEP 2 SUMMARY: Within-group Across-timepoint Analysis"),
    paste0("Stratification: ", config$display_name),
    paste(rep("=", 70), collapse = ""),
    "",
    sprintf("Generated: %s", Sys.time()),
    "",
    "OVERVIEW:",
    sprintf("  - Strata: %s", paste(config$levels, collapse = ", ")),
    sprintf("  - Cell types analyzed: %d", if (nrow(sig_summary) > 0) length(unique(sig_summary$celltype)) else 0),
    sprintf("  - Total gene analyses completed: %d", length(all_results)),
    sprintf("  - Total gene analyses skipped: %d", sum(sapply(skipped_list, function(x) x$celltype != "ALL (proportion)"))),
    "",
    paste(rep("-", 70), collapse = ""),
    "DIFFERENTIAL EXPRESSION STATISTICS (TWO-TIER):",
    paste(rep("-", 70), collapse = ""),
    sprintf("  - Total gene tests: %d", nrow(combined_df)),
    "",
    sprintf("  - Total significant (FDR < 0.05): %d", total_sig_all),
    sprintf("      FDR + logFC significant (FDR<0.05 & |logFC|>1): %d", total_sig_strict),
    sprintf("      FDR-only significant (FDR<0.05 & |logFC|<=1): %d", total_sig_fdr_only),
    "",
    sprintf("  - Up-regulated (FDR < 0.05 & logFC > 1): %d", total_up),
    sprintf("  - Down-regulated (FDR < 0.05 & logFC < -1): %d", total_down),
    sprintf("  - Unique significant genes: %d", length(unique_sig_genes)),
    ""
  )
  
  # ========================================================================
  # LIST ALL SIGNIFICANT GENE NAMES (per stratum, celltype, comparison)
  # ========================================================================
  if (nrow(combined_df) > 0) {
    sig_genes_df <- combined_df %>%
      filter(FDR < 0.05, !grepl("^ENSG", gene)) %>%
      arrange(stratum, celltype, FDR)
    
    if (nrow(sig_genes_df) > 0) {
      summary_txt <- c(summary_txt,
                       paste(rep("=", 70), collapse = ""),
                       "ALL SIGNIFICANT GENE NAMES (FDR < 0.05):",
                       paste(rep("=", 70), collapse = ""),
                       "")
      
      # Unique gene list across all
      summary_txt <- c(summary_txt,
                       sprintf("Unique significant genes (n=%d):", length(unique_sig_genes)),
                       paste0("  ", paste(sort(unique_sig_genes), collapse = ", ")),
                       "")
      
      # Per stratum x celltype x comparison
      for (grp in config$levels) {
        grp_sig <- sig_genes_df %>% filter(stratum == grp)
        if (nrow(grp_sig) == 0) next
        
        summary_txt <- c(summary_txt,
                         paste(rep("-", 50), collapse = ""),
                         paste0("Stratum: ", grp),
                         paste(rep("-", 50), collapse = ""))
        
        comparisons_in_grp <- grp_sig %>%
          mutate(comparison = paste0(tp_short(timepoint_late), " vs ", tp_short(timepoint_early))) %>%
          distinct(comparison, celltype) %>%
          arrange(celltype, comparison)
        
        for (i in 1:nrow(comparisons_in_grp)) {
          ct_i <- comparisons_in_grp$celltype[i]
          comp_i <- comparisons_in_grp$comparison[i]
          
          genes_here <- grp_sig %>%
            mutate(comparison = paste0(tp_short(timepoint_late), " vs ", tp_short(timepoint_early))) %>%
            filter(celltype == ct_i, comparison == comp_i) %>%
            arrange(FDR)
          
          strict_genes <- genes_here %>% filter(abs(logFC) > 1) %>% pull(gene)
          fdr_only_genes <- genes_here %>% filter(abs(logFC) <= 1) %>% pull(gene)
          
          summary_txt <- c(summary_txt,
                           sprintf("  %s | %s (n=%d):", ct_i, comp_i, nrow(genes_here)))
          
          if (length(strict_genes) > 0) {
            summary_txt <- c(summary_txt,
                             sprintf("    FDR+logFC (n=%d): %s", length(strict_genes),
                                     paste(strict_genes, collapse = ", ")))
          }
          if (length(fdr_only_genes) > 0) {
            summary_txt <- c(summary_txt,
                             sprintf("    FDR-only (n=%d): %s", length(fdr_only_genes),
                                     paste(fdr_only_genes, collapse = ", ")))
          }
        }
        summary_txt <- c(summary_txt, "")
      }
    } else {
      summary_txt <- c(summary_txt,
                       paste(rep("=", 70), collapse = ""),
                       "NO SIGNIFICANT GENES FOUND (FDR < 0.05)",
                       paste(rep("=", 70), collapse = ""),
                       "")
    }
  }
  
  # Per-stratum summary
  for (grp in config$levels) {
    if (nrow(sig_summary) > 0) {
      grp_stats <- sig_summary %>% filter(stratum == grp)
      total_sig_grp <- sum(grp_stats$n_sig_05_all)
      total_strict_grp <- sum(grp_stats$n_sig_05_strict)
      total_fdr_only_grp <- sum(grp_stats$n_sig_05_fdr_only)
      total_up_grp <- sum(grp_stats$n_up)
      total_down_grp <- sum(grp_stats$n_down)
    } else {
      total_sig_grp <- 0; total_strict_grp <- 0; total_fdr_only_grp <- 0
      total_up_grp <- 0; total_down_grp <- 0
    }
    
    summary_txt <- c(summary_txt,
                     paste(rep("-", 50), collapse = ""),
                     paste0("STRATUM: ", grp),
                     paste(rep("-", 50), collapse = ""),
                     sprintf("  Total significant: %d", total_sig_grp),
                     sprintf("    - FDR + logFC: %d (up: %d, down: %d)", total_strict_grp, total_up_grp, total_down_grp),
                     sprintf("    - FDR-only: %d", total_fdr_only_grp),
                     ""
    )
  }
  
  # Proportion analysis summary section
  if (nrow(combined_prop_df) > 0) {
    prop_sig_all <- sum(combined_prop_df$FDR < 0.05, na.rm = TRUE)
    prop_sig_strict <- sum(combined_prop_df$FDR < 0.05 & abs(combined_prop_df$logFC) > 1, na.rm = TRUE)
    prop_sig_fdr_only <- sum(combined_prop_df$FDR < 0.05 & abs(combined_prop_df$logFC) <= 1, na.rm = TRUE)
    
    summary_txt <- c(summary_txt,
                     "",
                     paste(rep("=", 70), collapse = ""),
                     "DIFFERENTIAL CELL TYPE PROPORTION ANALYSIS:",
                     paste(rep("=", 70), collapse = ""),
                     sprintf("  - Total cell type tests: %d", nrow(combined_prop_df)),
                     sprintf("  - Total significant (FDR < 0.05): %d", prop_sig_all),
                     sprintf("      FDR + logFC (FDR<0.05 & |logFC|>1): %d", prop_sig_strict),
                     sprintf("      FDR-only (FDR<0.05 & |logFC|<=1): %d", prop_sig_fdr_only))
    
    prop_by_comp <- combined_prop_df %>%
      mutate(comparison = paste0(tp_short(timepoint_late), " vs ", tp_short(timepoint_early))) %>%
      group_by(stratum, comparison) %>%
      summarise(
        n_tested = n(),
        n_sig = sum(FDR < 0.05, na.rm = TRUE),
        .groups = "drop"
      )
    
    summary_txt <- c(summary_txt, "", "  By stratum and comparison:")
    for (i in 1:nrow(prop_by_comp)) {
      summary_txt <- c(summary_txt,
                       sprintf("    %s | %s: %d tested, %d significant",
                               prop_by_comp$stratum[i], prop_by_comp$comparison[i],
                               prop_by_comp$n_tested[i], prop_by_comp$n_sig[i]))
    }
    
    # List significant cell types
    sig_props <- combined_prop_df %>% filter(FDR < 0.05) %>% arrange(FDR)
    if (nrow(sig_props) > 0) {
      summary_txt <- c(summary_txt, "", "  Significant cell type proportions:")
      for (i in 1:nrow(sig_props)) {
        tier <- if (abs(sig_props$logFC[i]) > 1) "FDR+logFC" else "FDR-only"
        summary_txt <- c(summary_txt,
                         sprintf("    %s @ %s %s vs %s: logFC=%.3f, FDR=%.2e (%s)",
                                 sig_props$gene[i], sig_props$stratum[i],
                                 tp_short(sig_props$timepoint_late[i]),
                                 tp_short(sig_props$timepoint_early[i]),
                                 sig_props$logFC[i], sig_props$FDR[i], tier))
      }
    }
  }
  
  writeLines(summary_txt, file.path(group_summary_dir, "analysis_summary.txt"))
  
  message("  -> Summary generated for ", config$display_name, " in: ", group_summary_dir)
  invisible(NULL)
}

# =============================================================================
# STEP 2: Within-group, across-timepoint pairwise comparisons - PARALLELIZED
# Now includes proportion analysis (matching Step 1 pattern)
# =============================================================================
run_step2_stratified <- function(parent_dir, output_dir, strat_type,
                                 env_lib = "~/R_envs/differential_gene",
                                 batch_keys = c("Sex", "BMI category", "LC/Recovered", "age_cluster"),
                                 heatmap_gene_counts = c(10, 20, 30, 40, 50),
                                 make_boxplots = TRUE, max_boxplots = 100,
                                 min_samples_per_tp = 2,
                                 use_hvg_for_pca = TRUE, n_hvg = 2000,
                                 min_cov_per_level = 2, min_resid_df = 2,
                                 strict_batch_control = TRUE) {
  
  setup_pipeline_env(env_lib)
  config <- get_stratification_config(strat_type)
  struc <- discover_structure(parent_dir)
  
  message("\n", strrep("=", 70))
  message("STRATIFICATION: ", config$display_name)
  message("Levels: ", paste(config$levels, collapse = ", "))
  message("Found timepoints: ", paste(struc$tp_sorted, collapse = ", "))
  message("Found ", length(struc$all_celltypes), " unique cell types")
  message(strrep("=", 70))
  
  step2_dir <- file.path(output_dir, "step2_across_timepoint", config$subdir)
  cov_keys_use <- batch_keys[!batch_keys %in% config$exclude_from_cov]
  tp_pairs <- combn(struc$tp_sorted, 2, simplify = FALSE)
  
  # =========================================================================
  # GENE EXPRESSION ANALYSIS (parallelized)
  # =========================================================================
  tasks <- list()
  for (ct in struc$all_celltypes) {
    for (pair in tp_pairs) {
      for (grp_name in config$levels) {
        tasks[[length(tasks) + 1]] <- list(ct = ct, tp1 = pair[1], tp2 = pair[2], grp_name = grp_name)
      }
    }
  }
  
  message(sprintf("Processing %d gene expression comparisons in parallel...", length(tasks)))
  
  all_outputs <- future_lapply(tasks, function(task) {
    ct <- task$ct
    tp1 <- task$tp1
    tp2 <- task$tp2
    grp_name <- task$grp_name
    tp1s <- tp_short(tp1)
    tp2s <- tp_short(tp2)
    
    result <- list(success = FALSE, de_result = NULL, skipped_info = NULL)
    
    tryCatch({
      # Check cell type exists at both timepoints
      if (!ct %in% struc$celltype_map[[tp1]] || !ct %in% struc$celltype_map[[tp2]]) {
        result$skipped_info <- list(
          celltype = ct, stratum = grp_name,
          timepoint_early = tp1, timepoint_late = tp2,
          reason = "Cell type not found at one or both timepoints",
          failed_covariates = character(0), n_samples = "NA"
        )
        return(result)
      }
      
      dat1 <- load_celltype_data(file.path(parent_dir, tp1, ct))
      dat2 <- load_celltype_data(file.path(parent_dir, tp2, ct))
      if (is.null(dat1) || is.null(dat2)) {
        result$skipped_info <- list(
          celltype = ct, stratum = grp_name,
          timepoint_early = tp1, timepoint_late = tp2,
          reason = "Data loading failed",
          failed_covariates = character(0), n_samples = "NA"
        )
        return(result)
      }
      
      group1 <- config$extract_fn(dat1$meta)
      group2 <- config$extract_fn(dat2$meta)
      if (is.null(group1) || is.null(group2)) {
        result$skipped_info <- list(
          celltype = ct, stratum = grp_name,
          timepoint_early = tp1, timepoint_late = tp2,
          reason = paste0("Could not extract ", config$name, " from metadata"),
          failed_covariates = character(0), n_samples = "NA"
        )
        return(result)
      }
      
      idx1 <- which(group1 == grp_name)
      idx2 <- which(group2 == grp_name)
      
      if (length(idx1) < min_samples_per_tp || length(idx2) < min_samples_per_tp) {
        result$skipped_info <- list(
          celltype = ct, stratum = grp_name,
          timepoint_early = tp1, timepoint_late = tp2,
          reason = sprintf("Insufficient samples (need >= %d per tp, got %d and %d)",
                           min_samples_per_tp, length(idx1), length(idx2)),
          failed_covariates = character(0),
          n_samples = paste0(tp1, "=", length(idx1), ", ", tp2, "=", length(idx2))
        )
        return(result)
      }
      
      expr1 <- dat1$expr[idx1, , drop=FALSE]
      expr2 <- dat2$expr[idx2, , drop=FALSE]
      meta1 <- dat1$meta[idx1, , drop=FALSE]
      meta2 <- dat2$meta[idx2, , drop=FALSE]
      
      cg <- intersect(colnames(expr1), colnames(expr2))
      if (length(cg) < 10) {
        result$skipped_info <- list(
          celltype = ct, stratum = grp_name,
          timepoint_early = tp1, timepoint_late = tp2,
          reason = sprintf("Insufficient common genes (%d)", length(cg)),
          failed_covariates = character(0), n_samples = "NA"
        )
        return(result)
      }
      expr1 <- expr1[, cg, drop=FALSE]
      expr2 <- expr2[, cg, drop=FALSE]
      
      # Unique sample names
      rownames(expr1) <- paste0(tp1s, "_", rownames(expr1))
      rownames(expr2) <- paste0(tp2s, "_", rownames(expr2))
      rownames(meta1) <- paste0(tp1s, "_", rownames(meta1))
      rownames(meta2) <- paste0(tp2s, "_", rownames(meta2))
      
      ce <- rbind(expr1, expr2)
      cm <- rbind(meta1, meta2)
      tf <- factor(c(rep(tp1, nrow(expr1)), rep(tp2, nrow(expr2))), levels = c(tp1, tp2))
      
      outdir_this <- file.path(step2_dir, ct, paste0(grp_name, "_", tp1s, "_vs_", tp2s))
      dir.create(outdir_this, recursive = TRUE, showWarnings = FALSE)
      
      # Batch covariate check (strict mode)
      if (isTRUE(strict_batch_control)) {
        batch_check <- check_all_batch_covariates(
          group = tf, meta_df = cm, batch_keys = cov_keys_use,
          exclude_keys = config$exclude_from_cov, min_per_level = min_cov_per_level
        )
        
        if (!batch_check$can_model) {
          writeLines(c(
            "SKIPPED: Cannot control for all batch covariates",
            paste0("Stratification: ", config$name),
            paste0("Stratum: ", grp_name),
            paste0("Cell type: ", ct),
            paste0("Comparison: ", tp2, " vs ", tp1),
            paste0("N samples: ", nrow(ce)),
            "", "Batch covariate check details:", batch_check$details
          ), file.path(outdir_this, "SKIP_REASON.txt"))
          
          result$skipped_info <- list(
            celltype = ct, stratum = grp_name,
            timepoint_early = tp1, timepoint_late = tp2,
            reason = "Cannot control for all batch covariates",
            failed_covariates = batch_check$failed_covariates,
            n_samples = nrow(ce)
          )
          return(result)
        }
      }
      
      # Prepare covariates
      cov_df <- prepare_covariates(cm, cov_keys_use, exclude_keys = config$exclude_from_cov)
      filt <- apply_covariate_filter(ce, tf, cov_df)
      
      if (is.null(filt)) {
        result$skipped_info <- list(
          celltype = ct, stratum = grp_name,
          timepoint_early = tp1, timepoint_late = tp2,
          reason = "Covariate filtering removed too many samples",
          failed_covariates = character(0), n_samples = nrow(ce)
        )
        return(result)
      }
      
      ce <- filt$expr
      tf <- filt$group
      cov_df <- filt$cov
      
      tp_counts <- table(tf)
      if (any(tp_counts < min_samples_per_tp)) {
        result$skipped_info <- list(
          celltype = ct, stratum = grp_name,
          timepoint_early = tp1, timepoint_late = tp2,
          reason = "Insufficient samples after covariate filtering",
          failed_covariates = character(0),
          n_samples = paste(names(tp_counts), as.integer(tp_counts), sep="=", collapse=", ")
        )
        return(result)
      }
      
      # Build design matrix
      ds <- build_safe_design(tf, cov_df, min_resid_df = min_resid_df)
      design <- ds$design
      cov_used <- ds$cov_used
      cov_notes <- ds$notes
      
      if (is.null(design)) {
        writeLines(c("SKIPPED: design build failed", "", "Notes:", cov_notes),
                   file.path(outdir_this, "SKIP_REASON.txt"))
        result$skipped_info <- list(
          celltype = ct, stratum = grp_name,
          timepoint_early = tp1, timepoint_late = tp2,
          reason = "Design matrix build failed",
          failed_covariates = character(0), n_samples = nrow(ce)
        )
        return(result)
      }
      
      # Log covariate decisions
      writeLines(c(
        paste0("Stratification: ", config$name),
        paste0("Stratum: ", grp_name),
        paste0("Cell type: ", ct),
        paste0("Comparison: ", tp2, " vs ", tp1),
        paste0("N samples: ", nrow(ce)),
        paste0("Sample counts: ", paste(names(tp_counts), as.integer(tp_counts), sep="=", collapse=", ")),
        "", "Covariate notes:", cov_notes, "",
        paste0("Covariates used: ", if (!is.null(cov_used) && ncol(cov_used) > 0) paste(colnames(cov_used), collapse=", ") else "OFF"),
        paste0("Design cols: ", paste(colnames(design), collapse=", "))
      ), file.path(outdir_this, "COVARIATE_DECISIONS.txt"))
      
      # Find design columns
      col1 <- find_design_col(design, "group", tp1)
      col2 <- find_design_col(design, "group", tp2)
      
      if (is.null(col1) || is.null(col2)) {
        result$skipped_info <- list(
          celltype = ct, stratum = grp_name,
          timepoint_early = tp1, timepoint_late = tp2,
          reason = sprintf("Cannot find design columns for %s and %s", tp1, tp2),
          failed_covariates = character(0), n_samples = nrow(ce)
        )
        return(result)
      }
      
      # Contrast: later - earlier
      contrast <- makeContrasts(contrasts = paste0(col2, " - ", col1), levels = design)
      colnames(contrast) <- "tp_diff"
      
      res <- tryCatch(run_limma(ce, design, contrast, "tp_diff"), error = function(e) NULL)
      if (is.null(res)) {
        result$skipped_info <- list(
          celltype = ct, stratum = grp_name,
          timepoint_early = tp1, timepoint_late = tp2,
          reason = "limma analysis failed",
          failed_covariates = character(0), n_samples = nrow(ce)
        )
        return(result)
      }
      
      up_lab <- paste0("Up at ", tp2s)
      dn_lab <- paste0("Down at ", tp2s)
      res <- add_de_status_generic(res, up_lab, dn_lab)
      
      # Two-tier significance
      sig05_all <- get_sig_genes(res, fdr_thresh = 0.05)
      sig05_strict <- get_sig_genes_strict(res, fdr_thresh = 0.05, logfc_thresh = 1)
      sig05_fdr_only <- get_sig_genes_fdr_only(res, fdr_thresh = 0.05, logfc_thresh = 1)
      sig10 <- get_sig_genes(res, fdr_thresh = 0.10)
      
      color_map <- setNames(c("#E64B35", "#4DBBD5", "#FFA500", "grey70"),
                            c(up_lab, dn_lab, "FDR-only", "NS"))
      tp_colors <- setNames(c("#4DBBD5", "#E64B35"), c(tp1, tp2))
      
      cov_label <- if (!is.null(cov_used) && ncol(cov_used) > 0) paste(colnames(cov_used), collapse=", ") else "OFF"
      
      save_de_results(res, outdir_this,
                      paste0(config$display_name, ": ", grp_name, " | ", tp2, " vs ", tp1, " | ", ct),
                      nrow(ce), table(tf), cov_label,
                      sig05_all, sig05_strict, sig05_fdr_only, sig10)
      
      fwrite(as.data.frame(design) %>% mutate(sample = rownames(design)) %>% select(sample, everything()),
             file.path(outdir_this, "design_matrix.csv"))
      
      run_plot_suite(res, ce, tf, file.path(outdir_this, "figures"),
                     paste0(ct, " ", grp_name, " ", tp2s, " vs ", tp1s),
                     sig05_all = sig05_all, sig05_strict = sig05_strict,
                     sig05_fdr_only = sig05_fdr_only, sig10 = sig10,
                     color_map = color_map, group_color_values = tp_colors,
                     anno_name = "Timepoint", anno_colors_list = list(Timepoint = tp_colors),
                     do_boxplots = make_boxplots, max_boxplots = max_boxplots,
                     heatmap_gene_counts = heatmap_gene_counts,
                     use_hvg_for_pca = use_hvg_for_pca, n_hvg = n_hvg)
      
      result$success <- TRUE
      result$de_result <- res %>%
        mutate(celltype = ct, stratification = config$name, stratum = grp_name,
               timepoint_early = tp1, timepoint_late = tp2) %>%
        select(stratification, stratum, celltype, timepoint_early, timepoint_late, everything())
      
      return(result)
      
    }, error = function(e) {
      result$skipped_info <- list(
        celltype = ct, stratum = grp_name,
        timepoint_early = tp1, timepoint_late = tp2,
        reason = paste("Error:", e$message),
        failed_covariates = character(0), n_samples = "NA"
      )
      return(result)
    })
    
  }, future.seed = TRUE)
  
  # Separate successful results from skipped
  successful_results <- lapply(all_outputs, function(x) if (x$success) x$de_result else NULL)
  successful_results <- successful_results[!sapply(successful_results, is.null)]
  
  skipped_list <- lapply(all_outputs, function(x) x$skipped_info)
  skipped_list <- skipped_list[!sapply(skipped_list, is.null)]
  
  message(sprintf("Gene DE: Completed %d/%d analyses successfully", length(successful_results), length(tasks)))
  message(sprintf("Gene DE: Skipped %d analyses", length(skipped_list)))
  
  # =========================================================================
  # PROPORTION ANALYSIS (per timepoint pair x stratum, parallelized)
  # =========================================================================
  message("\n--- Running proportion analysis ---")
  
  prop_tasks <- list()
  for (pair in tp_pairs) {
    for (grp_name in config$levels) {
      prop_tasks[[length(prop_tasks) + 1]] <- list(tp1 = pair[1], tp2 = pair[2], grp_name = grp_name)
    }
  }
  
  prop_outputs <- future_lapply(prop_tasks, function(task) {
    tryCatch({
      run_proportion_analysis_one_pair(
        parent_dir = parent_dir, tp1 = task$tp1, tp2 = task$tp2,
        grp_name = task$grp_name, config = config,
        batch_keys = batch_keys, output_dir = output_dir,
        strict_batch_control = strict_batch_control,
        min_samples_per_tp = min_samples_per_tp,
        min_cov_per_level = min_cov_per_level,
        min_resid_df = min_resid_df)
    }, error = function(e) {
      list(success = FALSE, de_result = NULL,
           skipped_info = list(
             timepoint_early = task$tp1, timepoint_late = task$tp2,
             celltype = "ALL (proportion)", stratum = task$grp_name,
             reason = paste("Error:", e$message),
             failed_covariates = character(0),
             n_samples = "NA", group_counts = "NA"))
    })
  }, future.seed = TRUE)
  
  prop_results <- lapply(prop_outputs, function(x) if (x$success) x$de_result else NULL)
  prop_results <- prop_results[!sapply(prop_results, is.null)]
  
  prop_skipped <- lapply(prop_outputs, function(x) x$skipped_info)
  prop_skipped <- prop_skipped[!sapply(prop_skipped, is.null)]
  
  message(sprintf("Proportion: Completed %d/%d analyses successfully", length(prop_results), length(prop_tasks)))
  message(sprintf("Proportion: Skipped %d analyses", length(prop_skipped)))
  
  list(results = successful_results, skipped = skipped_list,
       proportion_results = prop_results, proportion_skipped = prop_skipped,
       config = config, struc = struc)
}

# =============================================================================
# MAIN STEP 2 RUNNER: Run all stratifications - PARALLELIZED
# =============================================================================
run_step2 <- function(parent_dir, output_dir,
                      env_lib = "~/R_envs/differential_gene",
                      batch_keys = c("Sex", "BMI category", "LC/Recovered", "age_cluster"),
                      stratifications = c("LC_Recovered", "BMI", "Sex", "Age"),
                      heatmap_gene_counts = c(10, 20, 30, 40, 50),
                      make_boxplots = TRUE, max_boxplots = 100,
                      min_samples_per_tp = 2,
                      use_hvg_for_pca = TRUE, n_hvg = 2000,
                      n_workers = NULL, memory_per_worker = "4GB",
                      strict_batch_control = TRUE) {
  
  setup_pipeline_env(env_lib)
  setup_parallel(n_workers = n_workers, memory_per_worker = memory_per_worker)
  struc <- discover_structure(parent_dir)
  
  message("\n", strrep("=", 70))
  message("STEP 2: Within-group, across-timepoint pairwise comparisons (PARALLEL)")
  message("Includes: Differential gene expression + Cell type proportions")
  message("Stratifications to run: ", paste(stratifications, collapse = ", "))
  message("Strict batch control: ", strict_batch_control)
  message(strrep("=", 70))
  
  all_outputs_combined <- list()
  
  for (strat in stratifications) {
    message("\n\n", strrep("#", 70))
    message("Starting stratification: ", strat)
    message(strrep("#", 70))
    
    output <- run_step2_stratified(
      parent_dir = parent_dir,
      output_dir = output_dir,
      strat_type = strat,
      env_lib = env_lib,
      batch_keys = batch_keys,
      heatmap_gene_counts = heatmap_gene_counts,
      make_boxplots = make_boxplots,
      max_boxplots = max_boxplots,
      min_samples_per_tp = min_samples_per_tp,
      use_hvg_for_pca = use_hvg_for_pca,
      n_hvg = n_hvg,
      strict_batch_control = strict_batch_control
    )
    
    all_outputs_combined[[strat]] <- output
  }
  
  # ==========================================================================
  # Generate comprehensive summaries
  # ==========================================================================
  summary_dir <- file.path(output_dir, "summary")
  dir.create(summary_dir, recursive = TRUE, showWarnings = FALSE)
  
  message("\n\n", strrep("=", 70))
  message("GENERATING COMPREHENSIVE SUMMARIES FOR EACH STRATIFICATION")
  message(strrep("=", 70))
  
  for (strat in names(all_outputs_combined)) {
    output <- all_outputs_combined[[strat]]
    config <- get_stratification_config(strat)
    message("\n--- Summary for: ", config$display_name, " ---")
    tryCatch({
      generate_group_summary(
        all_results = output$results,
        skipped_list = output$skipped,
        config = config,
        struc = output$struc,
        summary_dir = summary_dir,
        proportion_results = output$proportion_results,
        proportion_skipped = output$proportion_skipped,
        output_dir = output_dir
      )
    }, error = function(e) {
      message("  Warning: Error generating summary for ", config$display_name, ": ", e$message)
    })
  }
  
  # ==========================================================================
  # Overall cross-stratification summary
  # ==========================================================================
  message("\n--- Generating overall summary ---")
  
  overall_summary <- c(
    paste(rep("=", 70), collapse = ""),
    "OVERALL STEP 2 SUMMARY: Within-group Across-timepoint Analysis",
    paste(rep("=", 70), collapse = ""),
    "",
    sprintf("Analysis date: %s", Sys.time()),
    sprintf("Input directory: %s", parent_dir),
    sprintf("Output directory: %s", output_dir),
    "",
    paste(rep("-", 70), collapse = ""),
    "STRATIFICATIONS ANALYZED:",
    paste(rep("-", 70), collapse = "")
  )
  
  for (strat in names(all_outputs_combined)) {
    output <- all_outputs_combined[[strat]]
    config <- get_stratification_config(strat)
    n_completed <- length(output$results)
    n_skipped <- length(output$skipped)
    
    if (n_completed > 0) {
      combined <- do.call(rbind, output$results)
      n_sig_all <- sum(combined$FDR < 0.05 & !grepl("^ENSG", combined$gene), na.rm = TRUE)
      n_sig_strict <- sum(combined$FDR < 0.05 & abs(combined$logFC) > 1 & !grepl("^ENSG", combined$gene), na.rm = TRUE)
      n_sig_fdr_only <- sum(combined$FDR < 0.05 & abs(combined$logFC) <= 1 & !grepl("^ENSG", combined$gene), na.rm = TRUE)
      
      # Sort combined results by FDR
      combined <- combined %>% arrange(FDR, desc(abs(logFC)))
      fwrite(combined, file.path(summary_dir, paste0("all_step2_", strat, ".csv")))
      
      # Collect all significant gene names for overall summary
      sig_genes_this_strat <- combined %>%
        filter(FDR < 0.05, !grepl("^ENSG", gene)) %>%
        pull(gene) %>% unique() %>% sort()
    } else {
      n_sig_all <- 0; n_sig_strict <- 0; n_sig_fdr_only <- 0
      sig_genes_this_strat <- character(0)
    }
    
    # Proportion stats
    n_prop_completed <- length(output$proportion_results)
    n_prop_skipped <- length(output$proportion_skipped)
    if (n_prop_completed > 0) {
      combined_prop <- do.call(rbind, output$proportion_results)
      n_prop_sig <- sum(combined_prop$FDR < 0.05, na.rm = TRUE)
    } else {
      n_prop_sig <- 0
    }
    
    overall_summary <- c(overall_summary,
                         "",
                         sprintf("  %s:", config$display_name),
                         sprintf("    - Levels: %s", paste(config$levels, collapse = ", ")),
                         sprintf("    Gene DE:"),
                         sprintf("      - Completed analyses: %d", n_completed),
                         sprintf("      - Skipped analyses: %d", n_skipped),
                         sprintf("      - Total sig (FDR<0.05): %d", n_sig_all),
                         sprintf("        FDR+logFC: %d | FDR-only: %d", n_sig_strict, n_sig_fdr_only),
                         sprintf("      - Unique sig genes (n=%d): %s",
                                 length(sig_genes_this_strat),
                                 if (length(sig_genes_this_strat) > 0) paste(sig_genes_this_strat, collapse = ", ") else "none"),
                         sprintf("    Proportion DE:"),
                         sprintf("      - Completed analyses: %d", n_prop_completed),
                         sprintf("      - Skipped analyses: %d", n_prop_skipped),
                         sprintf("      - Significant proportions (FDR<0.05): %d", n_prop_sig)
    )
  }
  
  overall_summary <- c(overall_summary,
                       "",
                       paste(rep("-", 70), collapse = ""),
                       "OUTPUT STRUCTURE:",
                       paste(rep("-", 70), collapse = ""),
                       "",
                       "step2_across_timepoint/",
                       "  by_LC_status/[CellType]/[Stratum]_[Comparison]/",
                       "  by_BMI/[CellType]/[Stratum]_[Comparison]/",
                       "  by_Sex/[CellType]/[Stratum]_[Comparison]/",
                       "  by_Age/[CellType]/[Stratum]_[Comparison]/",
                       "",
                       "step2_proportion_analysis/",
                       "  by_*/[Stratum]_[Comparison]/",
                       "",
                       "summary/",
                       "  by_*/                                  - Stratification-specific summaries",
                       "  all_boxplots/by_*/                    - All boxplots collected by category",
                       "  all_step2_*.csv                       - Combined results per stratification (sorted by FDR)",
                       "  overall_summary.txt                   - This file",
                       ""
  )
  
  writeLines(overall_summary, file.path(summary_dir, "overall_summary.txt"))
  
  plan(sequential)
  message("\n\nStep 2 complete for all stratifications (gene expression + proportions).")
  message("Output directory: ", output_dir)
  message("Summary directory: ", summary_dir)
  message("Boxplots collected in: ", file.path(summary_dir, "all_boxplots"))
  
  invisible(all_outputs_combined)
}


# if (sys.nframe() == 0) {
#   run_step2(
#     parent_dir = "/dcs07/hongkai/data/harry/result/long_covid/sample_pseudobulk_differential_analysis/different_time_point",
#     output_dir = "/dcs07/hongkai/data/harry/result/long_covid/sample_pseudobulk_differential_analysis/step2",
#     env_lib    = "~/R_envs/differential_gene",
#     batch_keys = c("Sex", "BMI category", "LC/Recovered", "age_cluster"),
#     stratifications = c("LC_Recovered", "BMI", "Sex", "Age"),
#     heatmap_gene_counts = c(10, 20, 30),
#     make_boxplots = TRUE,
#     max_boxplots = 100,
#     min_samples_per_tp = 2,
#     use_hvg_for_pca = TRUE,
#     n_hvg = 2000,
#     n_workers = NULL,
#     memory_per_worker = "30GB",
#     strict_batch_control = TRUE
#   )
# }

if (sys.nframe() == 0) {
  run_step2(
    parent_dir = "/dcs07/hongkai/data/harry/result/long_covid/subset/sample_pseudobulk_differential_gene/B_cell",
    output_dir = "/dcs07/hongkai/data/harry/result/long_covid/subset/sample_pseudobulk_differential_gene/B_analysis/step2",
    env_lib    = "~/R_envs/differential_gene",
    batch_keys = c("Sex", "BMI category", "LC/Recovered", "age_cluster"),
    stratifications = c("LC_Recovered", "BMI", "Sex", "Age"),
    heatmap_gene_counts = c(10, 20, 30),
    make_boxplots = TRUE,
    max_boxplots = 100,
    min_samples_per_tp = 2,
    use_hvg_for_pca = TRUE,
    n_hvg = 2000,
    n_workers = NULL,
    memory_per_worker = "30GB",
    strict_batch_control = TRUE
  )
}