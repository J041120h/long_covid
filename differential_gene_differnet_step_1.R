#!/usr/bin/env Rscript
# =============================================================================
# Step 1: Cross-group Differential Expression Analysis (PARALLELIZED)
# Supports:
#   - LC vs Recovered
#   - BMI (Elevated vs Normal)
#   - Sex (Male vs Female)
#   - Age groups (Young / Middle / Old) via pairwise contrasts
#
# Includes:
#   - Differential GENE expression (pseudobulk)
#   - Differential CELL PROPORTION analysis (limma on proportions)
#   - Two-tier significance: FDR+logFC and FDR-only
#
# STRICT MODE: If ANY batch covariate cannot be controlled, SKIP that
# (timepoint x celltype) entirely.
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

DE_COLORS    <- c("Up in LC" = "#E64B35", "Down in LC" = "#4DBBD5",
                  "FDR-only" = "#FFA500", "NS" = "grey70")
DE_COLORS_BMI <- c("Up in Elevated" = "#F39B7F", "Down in Elevated" = "#00A087",
                   "FDR-only" = "#FFA500", "NS" = "grey70")
DE_COLORS_SEX <- c("Up in Male" = "#1B9E77", "Down in Male" = "#7570B3",
                   "FDR-only" = "#FFA500", "NS" = "grey70")

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
# Load cell type proportions data
# Format: rows = cell types, columns = samples, values = proportions
# Returns: list(prop_mat = samples x celltypes matrix, sample_ids = character)
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
# Load metadata for proportion analysis
# We need to find a metadata file that matches the sample IDs in proportions
# =============================================================================
load_proportion_metadata <- function(parent_dir, tp, sample_ids) {
  # Try to load metadata from any celltype directory (metadata is shared across celltypes)
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
# Group extraction functions (unified pattern)
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
      lv <- unique(v)
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
      lv <- unique(v)
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
      if (all(c("Young", "Middle", "Old") %in% unique(v))) {
        return(factor(v, levels = c("Young", "Middle", "Old")))
      }
    }
  }
  NULL
}

# -- Covariates ---------------------------------------------------------------
prepare_covariates <- function(meta_df, cov_keys, exclude_key = NULL) {
  if (is.null(cov_keys) || length(cov_keys) == 0) return(NULL)
  if (!is.null(exclude_key)) cov_keys <- cov_keys[!cov_keys %in% exclude_key]
  
  use_keys <- cov_keys[cov_keys %in% colnames(meta_df)]
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

# =============================================================================
# Check if a single covariate can be modeled given the main group
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

# =============================================================================
# Check if ALL required batch covariates can be modeled
# =============================================================================
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

# =============================================================================
# Build safe design matrix with minimum residual df
# =============================================================================
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
# TWO-TIER DE STATUS: FDR+logFC ("Up"/"Down") and FDR-only ("FDR-only")
# =============================================================================
add_de_status_generic <- function(df, up_label = "Up", down_label = "Down") {
  df %>% mutate(DE_status = factor(case_when(
    FDR < 0.05 & logFC > 1 ~ up_label,
    FDR < 0.05 & logFC < -1 ~ down_label,
    FDR < 0.05 ~ "FDR-only",
    TRUE ~ "NS"),
    levels = c(up_label, down_label, "FDR-only", "NS")))
}

# =============================================================================
# Gene selection helpers (unified)
# =============================================================================
get_top_logfc_genes <- function(res, top_n = 10) {
  top_up <- res %>% filter(logFC > 0) %>% arrange(desc(logFC)) %>% head(top_n) %>% pull(gene)
  top_down <- res %>% filter(logFC < 0) %>% arrange(logFC) %>% head(top_n) %>% pull(gene)
  list(top_up = top_up, top_down = top_down, all_top = c(top_up, top_down))
}

get_sig_genes <- function(res, fdr_thresh = 0.05) {
  res %>% filter(!is.na(FDR), FDR < fdr_thresh) %>%
    arrange(FDR, desc(abs(logFC))) %>% pull(gene)
}

# FDR+logFC significant genes
get_sig_genes_strict <- function(res, fdr_thresh = 0.05, logfc_thresh = 1) {
  res %>% filter(!is.na(FDR), FDR < fdr_thresh, abs(logFC) > logfc_thresh) %>%
    arrange(FDR, desc(abs(logFC))) %>% pull(gene)
}

# FDR-only significant genes (pass FDR but NOT logFC threshold)
get_sig_genes_fdr_only <- function(res, fdr_thresh = 0.05, logfc_thresh = 1) {
  res %>% filter(!is.na(FDR), FDR < fdr_thresh, abs(logFC) <= logfc_thresh) %>%
    arrange(FDR, desc(abs(logFC))) %>% pull(gene)
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

make_volcano <- function(res, title_str, figpath, color_map = DE_COLORS, top_n = 15, top_n_logfc = 10) {
  vol_df <- res %>% mutate(neglog10P = -log10(PValue + 1e-300))
  sl <- levels(vol_df$DE_status)
  uc <- color_map[names(color_map) %in% sl]
  
  # Label top FDR+logFC genes
  top_strict_labels <- vol_df %>% filter(FDR < 0.05, abs(logFC) > 1) %>% arrange(FDR) %>% head(top_n)
  # Label top FDR-only genes
  top_fdr_only_labels <- vol_df %>% filter(DE_status == "FDR-only") %>% arrange(FDR) %>% head(5)
  
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

make_ma_plot <- function(res, title_str, figpath, color_map = DE_COLORS) {
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

make_pval_hist <- function(res, title_str, figpath) {
  p <- ggplot(res, aes(x = PValue)) +
    geom_histogram(bins = 50, fill = "steelblue", color = "black", alpha = 0.7) +
    theme_bw(base_size = 12) + theme(panel.grid.minor = element_blank()) +
    labs(title = title_str, x = "P-value", y = "Frequency")
  save_plot(p, figpath, w = 7, h = 5)
}

make_de_summary_bar <- function(res, title_str, figpath, color_map = DE_COLORS) {
  ds <- res %>% filter(DE_status != "NS") %>% count(DE_status)
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
# Unified heatmap helper: prepare inputs and render
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
  
  if (is.null(anno_colors_list)) {
    anno_colors_list <- list()
    anno_colors_list[[anno_name]] <- GROUP_COLORS
  }
  
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
# Unified heatmap generation function (updated for two-tier significance)
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

make_gene_boxplots_batch <- function(genes, expr_mat, group_vec, res_df, outdir,
                                     group_colors_use = GROUP_COLORS, max_n = 100,
                                     label_type = "sig") {
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
    } else if (label_type == "top_fdr_nosig") {
      title_main <- paste0("Expression: ", gene, "\n(Top lowest FDR - no sig genes in this analysis)")
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
    } else if (label_type == "top_fdr_nosig") {
      p <- p + theme(plot.title = element_text(color = "purple4", face = "bold"),
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
# Main plot suite (updated for two-tier significance)
# CHANGE: If no sig genes (FDR<0.05), plot top 10 lowest FDR boxplots
# =============================================================================
run_plot_suite <- function(res, expr_mat, group_vec, figdir, title_prefix,
                           sig05_all, sig05_strict, sig05_fdr_only,
                           sig10,
                           color_map = DE_COLORS, group_color_values = GROUP_COLORS,
                           anno_name = "Group", anno_colors_list = NULL,
                           do_boxplots = TRUE, max_boxplots = 100,
                           heatmap_gene_counts = c(10, 20, 30, 40, 50)) {
  
  dir.create(figdir, recursive = TRUE, showWarnings = FALSE)
  plot_mat <- t(expr_mat)
  has_any_sig <- length(sig05_all) > 0
  
  # Core plots
  make_volcano(res, paste0(title_prefix, " - Volcano"), file.path(figdir, "volcano.png"),
               color_map, top_n = 15, top_n_logfc = 10)
  make_ma_plot(res, paste0(title_prefix, " - MA"), file.path(figdir, "ma_plot.png"), color_map)
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
    # =========================================================================
    # NEW: Plot boxplots for top 10 lowest FDR genes when no sig genes found
    # =========================================================================
    if (do_boxplots) {
      top10_fdr_genes <- res %>%
        filter(!is.na(FDR)) %>%
        arrange(FDR) %>%
        head(10) %>%
        pull(gene)
      
      if (length(top10_fdr_genes) > 0) {
        make_gene_boxplots_batch(top10_fdr_genes, expr_mat, group_vec, res,
                                file.path(figdir, "boxplots_top10_lowest_FDR"),
                                group_colors_use = group_color_values,
                                max_n = 10, label_type = "top_fdr_nosig")
      }
    }
  }
}

# =============================================================================
# Save DE results (updated for two-tier significance)
# CHANGE: Sort DE_results.csv by lowest FDR
# =============================================================================
save_de_results <- function(res, outdir, method_label, n_samples, n_group_table,
                            cov_label, sig05_all, sig05_strict, sig05_fdr_only,
                            sig10, extra_lines = NULL) {
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  
  # Sort by lowest FDR first, then by highest absolute logFC
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
# COMPARISON CONFIGURATION
# =============================================================================
get_comparison_config <- function(comparison_type) {
  configs <- list(
    "LC_vs_Recovered" = list(
      name = "LC_vs_Recovered",
      display_name = "LC vs Recovered",
      extract_fn = extract_group,
      group_colors = GROUP_COLORS,
      de_colors = DE_COLORS,
      up_label = "Up in LC",
      down_label = "Down in LC",
      coef_name = "LC_vs_Recovered",
      level1 = "Recovered",
      level2 = "LC",
      exclude_from_cov = c("LC/Recovered", "LC.Recovered"),
      subdir = "LC_vs_Recovered",
      is_multigroup = FALSE,
      category_label = "LC"
    ),
    "BMI_Elevated_vs_Normal" = list(
      name = "BMI_Elevated_vs_Normal",
      display_name = "Elevated BMI vs Normal",
      extract_fn = extract_bmi_group,
      group_colors = BMI_COLORS,
      de_colors = DE_COLORS_BMI,
      up_label = "Up in Elevated",
      down_label = "Down in Elevated",
      coef_name = "Elevated_vs_Normal",
      level1 = "Normal",
      level2 = "Elevated",
      exclude_from_cov = c("BMI category", "BMI_category", "BMI.category"),
      subdir = "BMI_Elevated_vs_Normal",
      is_multigroup = FALSE,
      category_label = "BMI"
    ),
    "Sex_Male_vs_Female" = list(
      name = "Sex_Male_vs_Female",
      display_name = "Male vs Female",
      extract_fn = extract_sex_group,
      group_colors = SEX_COLORS,
      de_colors = DE_COLORS_SEX,
      up_label = "Up in Male",
      down_label = "Down in Male",
      coef_name = "Male_vs_Female",
      level1 = "Female",
      level2 = "Male",
      exclude_from_cov = c("Sex", "sex", "Gender", "gender"),
      subdir = "Sex_Male_vs_Female",
      is_multigroup = FALSE,
      category_label = "Sex"
    ),
    "Age_Young_Middle_Old" = list(
      name = "Age_Young_Middle_Old",
      display_name = "Age groups (Young/Middle/Old)",
      extract_fn = extract_age_group,
      group_colors = AGE_COLORS,
      de_colors = c("Up" = "#E64B35", "Down" = "#4DBBD5", "FDR-only" = "#FFA500", "NS" = "grey70"),
      exclude_from_cov = c("age_cluster", "Age_cluster", "ageCluster", "AgeCluster", "age.group", "Age group"),
      subdir = "Age_Young_Middle_Old",
      is_multigroup = TRUE,
      contrasts = list(
        list(coef_name = "Old_vs_Young",    level1 = "Young",  level2 = "Old",    up_label = "Up in Old",    down_label = "Down in Old"),
        list(coef_name = "Middle_vs_Young", level1 = "Young",  level2 = "Middle", up_label = "Up in Middle", down_label = "Down in Middle"),
        list(coef_name = "Old_vs_Middle",   level1 = "Middle", level2 = "Old",    up_label = "Up in Old",    down_label = "Down in Old")
      ),
      category_label = "Age"
    )
  )
  
  if (!comparison_type %in% names(configs)) {
    stop("Unknown comparison type: ", comparison_type,
         "\nAvailable: ", paste(names(configs), collapse = ", "))
  }
  configs[[comparison_type]]
}

# =============================================================================
# COMPREHENSIVE SUMMARY GENERATOR (updated for two-tier + proportion)
# CHANGES:
#   1. Collect all sig gene names in summary txt
#   2. Copy all boxplots to a common folder with category label
# =============================================================================
generate_group_summary <- function(all_results, skipped_list, config, struc, summary_dir,
                                   proportion_results = NULL, proportion_skipped = NULL,
                                   step1_dir_root = NULL) {
  group_summary_dir <- file.path(summary_dir, config$subdir)
  dir.create(group_summary_dir, recursive = TRUE, showWarnings = FALSE)
  figures_dir <- file.path(group_summary_dir, "figures")
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
  
  # =========================================================================
  # NEW: Common boxplot folder at summary level
  # =========================================================================
  boxplot_common_dir <- file.path(summary_dir, "all_boxplots")
  dir.create(boxplot_common_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Determine category label (LC, BMI, Sex, Age)
  category_label <- if (!is.null(config$category_label)) config$category_label else config$subdir
  
  # -----------------------------------------------------------------------------
  # 1. Combined DE results CSV (sorted by lowest FDR)
  # -----------------------------------------------------------------------------
  if (length(all_results) > 0) {
    combined_df <- do.call(rbind, all_results)
    combined_df_sorted <- combined_df %>%
      arrange(FDR, desc(abs(logFC)))
    fwrite(combined_df_sorted, file.path(group_summary_dir, "all_DE_results_combined.csv"))
  } else {
    combined_df <- data.frame()
  }
  
  # -----------------------------------------------------------------------------
  # 1b. Combined proportion results CSV (sorted by lowest FDR)
  # -----------------------------------------------------------------------------
  if (!is.null(proportion_results) && length(proportion_results) > 0) {
    combined_prop_df <- do.call(rbind, proportion_results)
    combined_prop_sorted <- combined_prop_df %>%
      arrange(FDR, desc(abs(logFC)))
    fwrite(combined_prop_sorted, file.path(group_summary_dir, "all_proportion_results_combined.csv"))
  } else {
    combined_prop_df <- data.frame()
  }
  
  # -----------------------------------------------------------------------------
  # 2. Skipped analyses summary (TXT + CSV)
  # -----------------------------------------------------------------------------
  if (length(skipped_list) > 0) {
    skipped_df <- do.call(rbind, lapply(skipped_list, function(x) {
      data.frame(
        timepoint = x$timepoint,
        celltype = x$celltype,
        reason = x$reason,
        failed_covariates = paste(x$failed_covariates, collapse = ", "),
        n_samples = as.character(x$n_samples),
        group_counts = x$group_counts,
        stringsAsFactors = FALSE
      )
    }))
  } else {
    skipped_df <- data.frame(
      timepoint = character(0), celltype = character(0), reason = character(0),
      failed_covariates = character(0), n_samples = character(0), group_counts = character(0),
      stringsAsFactors = FALSE
    )
  }
  
  # Proportion skipped
  if (!is.null(proportion_skipped) && length(proportion_skipped) > 0) {
    prop_skipped_df <- do.call(rbind, lapply(proportion_skipped, function(x) {
      data.frame(
        timepoint = x$timepoint,
        celltype = "ALL (proportion)",
        reason = x$reason,
        failed_covariates = paste(x$failed_covariates, collapse = ", "),
        n_samples = as.character(x$n_samples),
        group_counts = x$group_counts,
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
      sprintf("Total skipped: %d analyses", nrow(skipped_df)),
      "",
      paste(rep("-", 70), collapse = ""),
      "DETAILS:",
      paste(rep("-", 70), collapse = ""),
      ""
    )
    
    for (i in 1:nrow(skipped_df)) {
      skip_txt <- c(skip_txt,
                    sprintf("[%d] Timepoint: %s | Cell type: %s", i, skipped_df$timepoint[i], skipped_df$celltype[i]),
                    sprintf("    Reason: %s", skipped_df$reason[i]),
                    sprintf("    Failed covariates: %s", skipped_df$failed_covariates[i]),
                    sprintf("    N samples: %s", skipped_df$n_samples[i]),
                    sprintf("    Group counts: %s", skipped_df$group_counts[i]),
                    "")
    }
    
    skip_by_tp <- skipped_df %>% group_by(timepoint) %>% summarise(n = n(), .groups = "drop")
    skip_txt <- c(skip_txt, paste(rep("-", 70), collapse = ""), "SUMMARY BY TIMEPOINT:", paste(rep("-", 70), collapse = ""))
    for (i in 1:nrow(skip_by_tp)) {
      skip_txt <- c(skip_txt, sprintf("  %s: %d skipped", skip_by_tp$timepoint[i], skip_by_tp$n[i]))
    }
    
    all_failed <- unlist(strsplit(skipped_df$failed_covariates, ", "))
    all_failed <- all_failed[all_failed != ""]
    if (length(all_failed) > 0) {
      failed_counts <- table(all_failed)
      skip_txt <- c(skip_txt, "", paste(rep("-", 70), collapse = ""), "SUMMARY BY FAILED COVARIATE:", paste(rep("-", 70), collapse = ""))
      for (nm in names(failed_counts)) {
        skip_txt <- c(skip_txt, sprintf("  %s: %d times", nm, failed_counts[nm]))
      }
    }
    
    writeLines(skip_txt, file.path(group_summary_dir, "skipped_analyses_summary.txt"))
  } else {
    writeLines(c("No analyses were skipped.", 
                 "All cell type x timepoint combinations passed covariate checks."),
               file.path(group_summary_dir, "skipped_analyses_summary.txt"))
  }
  
  # -----------------------------------------------------------------------------
  # 3. Main summary statistics (TXT) and plots
  # -----------------------------------------------------------------------------
  if (nrow(combined_df) > 0) {
    sig_summary <- combined_df %>%
      group_by(timepoint, celltype) %>%
      summarise(
        n_genes_tested = n(),
        n_sig_05_all = sum(FDR < 0.05, na.rm = TRUE),
        n_sig_05_strict = sum(FDR < 0.05 & abs(logFC) > 1, na.rm = TRUE),
        n_sig_05_fdr_only = sum(FDR < 0.05 & abs(logFC) <= 1, na.rm = TRUE),
        n_sig_10 = sum(FDR < 0.10, na.rm = TRUE),
        n_up = sum(FDR < 0.05 & logFC > 1, na.rm = TRUE),
        n_down = sum(FDR < 0.05 & logFC < -1, na.rm = TRUE),
        mean_abs_logFC = mean(abs(logFC), na.rm = TRUE),
        max_abs_logFC = ifelse(all(is.na(logFC)), 0, max(abs(logFC), na.rm = TRUE)),
        .groups = "drop"
      ) %>%
      arrange(timepoint, celltype)
    
    fwrite(sig_summary, file.path(group_summary_dir, "DE_summary_by_celltype_timepoint.csv"))
    
    total_sig_05_all <- sum(combined_df$FDR < 0.05, na.rm = TRUE)
    total_sig_05_strict <- sum(combined_df$FDR < 0.05 & abs(combined_df$logFC) > 1, na.rm = TRUE)
    total_sig_05_fdr_only <- sum(combined_df$FDR < 0.05 & abs(combined_df$logFC) <= 1, na.rm = TRUE)
    total_sig_10 <- sum(combined_df$FDR < 0.10, na.rm = TRUE)
    total_up <- sum(combined_df$FDR < 0.05 & combined_df$logFC > 1, na.rm = TRUE)
    total_down <- sum(combined_df$FDR < 0.05 & combined_df$logFC < -1, na.rm = TRUE)
    
    unique_sig_genes <- combined_df %>% filter(FDR < 0.05) %>% pull(gene) %>% unique()
    
    sig_genes_df <- combined_df %>% filter(FDR < 0.05)
    if (nrow(sig_genes_df) > 0) {
      top_genes_overall <- sig_genes_df %>%
        group_by(gene) %>%
        summarise(
          n_comparisons = n(),
          mean_logFC = mean(logFC, na.rm = TRUE),
          min_FDR = min(FDR, na.rm = TRUE),
          .groups = "drop"
        ) %>%
        arrange(min_FDR) %>%
        head(50)
      fwrite(top_genes_overall, file.path(group_summary_dir, "top_50_genes_across_analyses.csv"))
    } else {
      top_genes_overall <- data.frame(gene = character(0), n_comparisons = integer(0),
                                      mean_logFC = numeric(0), min_FDR = numeric(0))
    }
    
    summary_txt <- c(
      paste(rep("=", 70), collapse = ""),
      paste0("DIFFERENTIAL EXPRESSION ANALYSIS SUMMARY: ", config$display_name),
      paste(rep("=", 70), collapse = ""),
      "",
      "OVERVIEW:",
      sprintf("  - Comparison: %s", config$display_name),
      sprintf("  - Timepoints analyzed: %d", length(unique(combined_df$timepoint))),
      sprintf("  - Cell types analyzed: %d", length(unique(combined_df$celltype))),
      sprintf("  - Total analyses completed: %d", nrow(sig_summary)),
      sprintf("  - Total analyses skipped: %d", nrow(skipped_df)),
      "",
      paste(rep("-", 70), collapse = ""),
      "DIFFERENTIAL EXPRESSION STATISTICS (TWO-TIER):",
      paste(rep("-", 70), collapse = ""),
      sprintf("  - Total gene tests: %d", nrow(combined_df)),
      "",
      sprintf("  - Total significant (FDR < 0.05): %d", total_sig_05_all),
      sprintf("      FDR + logFC significant (FDR<0.05 & |logFC|>1): %d", total_sig_05_strict),
      sprintf("      FDR-only significant (FDR<0.05 & |logFC|<=1): %d", total_sig_05_fdr_only),
      "",
      sprintf("  - Up-regulated (FDR < 0.05 & logFC > 1): %d", total_up),
      sprintf("  - Down-regulated (FDR < 0.05 & logFC < -1): %d", total_down),
      sprintf("  - Unique significant genes (FDR < 0.05): %d", length(unique_sig_genes)),
      sprintf("  - Significant genes (FDR < 0.10): %d", total_sig_10),
      "",
      paste(rep("-", 70), collapse = ""),
      "SIGNIFICANT GENES BY TIMEPOINT:",
      paste(rep("-", 70), collapse = "")
    )
    
    tp_summary <- sig_summary %>%
      group_by(timepoint) %>%
      summarise(
        n_celltypes = n(),
        total_sig_all = sum(n_sig_05_all),
        total_sig_strict = sum(n_sig_05_strict),
        total_sig_fdr_only = sum(n_sig_05_fdr_only),
        total_up = sum(n_up),
        total_down = sum(n_down),
        .groups = "drop"
      )
    
    for (i in 1:nrow(tp_summary)) {
      summary_txt <- c(summary_txt,
                       sprintf("  %s: %d cell types, %d sig total (%d FDR+logFC, %d FDR-only), up %d, down %d",
                               tp_summary$timepoint[i], tp_summary$n_celltypes[i],
                               tp_summary$total_sig_all[i], tp_summary$total_sig_strict[i],
                               tp_summary$total_sig_fdr_only[i],
                               tp_summary$total_up[i], tp_summary$total_down[i]))
    }
    
    summary_txt <- c(summary_txt, "",
                     paste(rep("-", 70), collapse = ""),
                     "SIGNIFICANT GENES BY CELL TYPE:",
                     paste(rep("-", 70), collapse = ""))
    
    ct_summary <- sig_summary %>%
      group_by(celltype) %>%
      summarise(
        n_timepoints = n(),
        total_sig_all = sum(n_sig_05_all),
        total_sig_strict = sum(n_sig_05_strict),
        total_sig_fdr_only = sum(n_sig_05_fdr_only),
        total_up = sum(n_up),
        total_down = sum(n_down),
        .groups = "drop"
      ) %>%
      arrange(desc(total_sig_all))
    
    for (i in 1:min(20, nrow(ct_summary))) {
      summary_txt <- c(summary_txt,
                       sprintf("  %s: %d tp, %d sig total (%d FDR+logFC, %d FDR-only), up %d, down %d",
                               ct_summary$celltype[i], ct_summary$n_timepoints[i],
                               ct_summary$total_sig_all[i], ct_summary$total_sig_strict[i],
                               ct_summary$total_sig_fdr_only[i],
                               ct_summary$total_up[i], ct_summary$total_down[i]))
    }
    if (nrow(ct_summary) > 20) {
      summary_txt <- c(summary_txt, sprintf("  ... and %d more cell types", nrow(ct_summary) - 20))
    }
    
    if (nrow(top_genes_overall) > 0) {
      summary_txt <- c(summary_txt, "",
                       paste(rep("-", 70), collapse = ""),
                       "TOP 20 MOST FREQUENTLY SIGNIFICANT GENES:",
                       paste(rep("-", 70), collapse = ""))
      for (i in 1:min(20, nrow(top_genes_overall))) {
        summary_txt <- c(summary_txt,
                         sprintf("  %d. %s: %d analyses, mean logFC=%.2f, min FDR=%.2e",
                                 i, top_genes_overall$gene[i], top_genes_overall$n_comparisons[i],
                                 top_genes_overall$mean_logFC[i], top_genes_overall$min_FDR[i]))
      }
    } else {
      summary_txt <- c(summary_txt, "",
                       paste(rep("-", 70), collapse = ""),
                       "No significant genes found (FDR < 0.05) across all analyses.",
                       paste(rep("-", 70), collapse = ""))
    }
    
    # =========================================================================
    # NEW: Collect ALL significant gene names into the summary txt
    # =========================================================================
    summary_txt <- c(summary_txt, "",
                     paste(rep("=", 70), collapse = ""),
                     "ALL SIGNIFICANT GENE NAMES (FDR < 0.05):",
                     paste(rep("=", 70), collapse = ""))
    
    if (length(unique_sig_genes) > 0) {
      # List unique sig genes with details per celltype/timepoint
      sig_gene_details <- combined_df %>%
        filter(FDR < 0.05) %>%
        arrange(gene, FDR) %>%
        group_by(gene) %>%
        summarise(
          n_hits = n(),
          best_FDR = min(FDR, na.rm = TRUE),
          best_logFC = logFC[which.min(FDR)],
          contexts = paste(unique(paste0(timepoint, "/", celltype)), collapse = "; "),
          .groups = "drop"
        ) %>%
        arrange(best_FDR)
      
      summary_txt <- c(summary_txt,
                       sprintf("Total unique significant genes: %d", nrow(sig_gene_details)),
                       "")
      
      for (i in 1:nrow(sig_gene_details)) {
        tier <- if (abs(sig_gene_details$best_logFC[i]) > 1) "FDR+logFC" else "FDR-only"
        summary_txt <- c(summary_txt,
                         sprintf("  %s | hits=%d | best_FDR=%.2e | best_logFC=%.3f | tier=%s | in: %s",
                                 sig_gene_details$gene[i],
                                 sig_gene_details$n_hits[i],
                                 sig_gene_details$best_FDR[i],
                                 sig_gene_details$best_logFC[i],
                                 tier,
                                 sig_gene_details$contexts[i]))
      }
    } else {
      summary_txt <- c(summary_txt, "  (none)")
    }
    
    # --- Proportion analysis summary section ---
    if (nrow(combined_prop_df) > 0) {
      summary_txt <- c(summary_txt, "",
                       paste(rep("=", 70), collapse = ""),
                       "DIFFERENTIAL CELL TYPE PROPORTION ANALYSIS:",
                       paste(rep("=", 70), collapse = ""))
      
      prop_sig_all <- sum(combined_prop_df$FDR < 0.05, na.rm = TRUE)
      prop_sig_strict <- sum(combined_prop_df$FDR < 0.05 & abs(combined_prop_df$logFC) > 1, na.rm = TRUE)
      prop_sig_fdr_only <- sum(combined_prop_df$FDR < 0.05 & abs(combined_prop_df$logFC) <= 1, na.rm = TRUE)
      
      summary_txt <- c(summary_txt,
                       sprintf("  - Total cell type tests: %d", nrow(combined_prop_df)),
                       sprintf("  - Total significant (FDR < 0.05): %d", prop_sig_all),
                       sprintf("      FDR + logFC (FDR<0.05 & |logFC|>1): %d", prop_sig_strict),
                       sprintf("      FDR-only (FDR<0.05 & |logFC|<=1): %d", prop_sig_fdr_only))
      
      prop_by_tp <- combined_prop_df %>%
        group_by(timepoint) %>%
        summarise(
          n_tested = n(),
          n_sig = sum(FDR < 0.05, na.rm = TRUE),
          .groups = "drop"
        )
      
      summary_txt <- c(summary_txt, "",
                       "  By timepoint:")
      for (i in 1:nrow(prop_by_tp)) {
        summary_txt <- c(summary_txt,
                         sprintf("    %s: %d tested, %d significant",
                                 prop_by_tp$timepoint[i], prop_by_tp$n_tested[i], prop_by_tp$n_sig[i]))
      }
      
      # List significant cell types
      sig_props <- combined_prop_df %>% filter(FDR < 0.05) %>% arrange(FDR)
      if (nrow(sig_props) > 0) {
        summary_txt <- c(summary_txt, "",
                         "  Significant cell type proportions:")
        for (i in 1:nrow(sig_props)) {
          tier <- if (abs(sig_props$logFC[i]) > 1) "FDR+logFC" else "FDR-only"
          summary_txt <- c(summary_txt,
                           sprintf("    %s @ %s: logFC=%.3f, FDR=%.2e (%s)",
                                   sig_props$gene[i], sig_props$timepoint[i],
                                   sig_props$logFC[i], sig_props$FDR[i], tier))
        }
      }
    }
    
    writeLines(summary_txt, file.path(group_summary_dir, "analysis_summary.txt"))
    
    # =========================================================================
    # NEW: Copy all boxplots to common folder with category label prefix
    # =========================================================================
    if (!is.null(step1_dir_root)) {
      boxplot_source_dir <- file.path(step1_dir_root, "step1_cross_group", config$subdir)
      if (dir.exists(boxplot_source_dir)) {
        all_boxplot_files <- list.files(boxplot_source_dir, pattern = "\\.png$",
                                        recursive = TRUE, full.names = TRUE)
        # Filter to only boxplot directories, EXCLUDE top10_lowest_FDR (not significant)
        boxplot_files <- all_boxplot_files[grepl("boxplots_", all_boxplot_files)]
        boxplot_files <- boxplot_files[!grepl("boxplots_top10_lowest_FDR", boxplot_files)]
        
        if (length(boxplot_files) > 0) {
          message(sprintf("  -> Copying %d boxplot files to common folder (category: %s)",
                          length(boxplot_files), category_label))
          for (bp_file in boxplot_files) {
            # Create new filename: category__original_filename.png
            gene_name <- basename(bp_file)
            # Get the boxplot subfolder name (e.g. boxplots_FDR_logFC_0.05)
            bp_subfolder <- basename(dirname(bp_file))
            new_name <- paste0(category_label, "__", bp_subfolder, "__", gene_name)
            dest <- file.path(boxplot_common_dir, new_name)
            tryCatch(file.copy(bp_file, dest, overwrite = TRUE),
                     error = function(e) message("  Warning: Could not copy ", bp_file))
          }
        }
      }
    }
    
    # -----------------------------------------------------------------------------
    # 4. Summary plots (same as before with two-tier adjustments)
    # -----------------------------------------------------------------------------
    
    # 4a. Heatmap of significant gene counts by celltype x timepoint
    heatmap_df <- sig_summary %>%
      select(timepoint, celltype, n_sig_05_all) %>%
      pivot_wider(names_from = timepoint, values_from = n_sig_05_all, values_fill = 0)
    
    if (nrow(heatmap_df) > 1 && ncol(heatmap_df) > 2) {
      heatmap_mat <- as.matrix(heatmap_df[, -1])
      rownames(heatmap_mat) <- heatmap_df$celltype
      
      if (any(heatmap_mat > 0)) {
        tryCatch({
          png(file.path(figures_dir, "heatmap_sig_genes_by_celltype_timepoint.png"),
              width = 1200, height = max(600, 50 + nrow(heatmap_mat) * 20), res = 150)
          max_val <- max(heatmap_mat, na.rm = TRUE)
          if (max_val == 0) max_val <- 1
          color_breaks <- seq(0, max_val, length.out = 101)
          pheatmap(heatmap_mat,
                   cluster_rows = (nrow(heatmap_mat) > 1 && any(apply(heatmap_mat, 1, var) > 0)),
                   cluster_cols = FALSE,
                   color = colorRampPalette(c("white", "yellow", "orange", "red"))(100),
                   breaks = color_breaks,
                   main = paste0(config$display_name, "\nSignificant Genes (FDR < 0.05, all) by Cell Type and Timepoint"),
                   fontsize_row = 8, fontsize_col = 10,
                   display_numbers = TRUE, number_format = "%.0f", number_color = "black",
                   border_color = "grey80")
          dev.off()
        }, error = function(e) {
          try(dev.off(), silent = TRUE)
          message("Warning: Could not create heatmap: ", e$message)
        })
      }
    }
    
    # 4b. Bar plot: total significant genes by timepoint
    if (nrow(tp_summary) > 0 && sum(tp_summary$total_sig_all) > 0) {
      tp_summary <- tp_summary %>%
        mutate(tp_num = as.numeric(gsub("[^0-9]", "", timepoint)))
      tp_summary$tp_num[is.na(tp_summary$tp_num)] <- Inf
      tp_summary <- tp_summary %>% arrange(tp_num)
      
      # Stacked bar: FDR+logFC vs FDR-only
      tp_stacked <- tp_summary %>%
        select(timepoint, total_sig_strict, total_sig_fdr_only) %>%
        pivot_longer(cols = c(total_sig_strict, total_sig_fdr_only),
                     names_to = "tier", values_to = "count") %>%
        mutate(tier = ifelse(tier == "total_sig_strict", "FDR + logFC", "FDR-only"))
      
      p_tp <- ggplot(tp_stacked, aes(x = factor(timepoint, levels = tp_summary$timepoint),
                                     y = count, fill = tier)) +
        geom_bar(stat = "identity", alpha = 0.8) +
        theme_bw(base_size = 12) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(title = paste0(config$display_name, " - Significant Genes by Timepoint (Two-Tier)"),
             subtitle = "FDR < 0.05",
             x = "Timepoint", y = "Number of Significant Genes", fill = "Significance Tier") +
        scale_fill_manual(values = c("FDR + logFC" = "#E64B35", "FDR-only" = "#FFA500"))
      save_plot(p_tp, file.path(figures_dir, "barplot_sig_genes_by_timepoint.png"), w = 8, h = 6)
    }
    
    # 4c. Bar plot: total significant genes by cell type (top 20)
    ct_top20 <- ct_summary %>% filter(total_sig_all > 0) %>% head(20)
    if (nrow(ct_top20) > 0) {
      p_ct <- ggplot(ct_top20, aes(x = reorder(celltype, total_sig_all), y = total_sig_all, fill = celltype)) +
        geom_bar(stat = "identity", alpha = 0.8) +
        geom_text(aes(label = total_sig_all), hjust = -0.2, size = 3) +
        coord_flip() +
        theme_bw(base_size = 12) +
        theme(legend.position = "none") +
        labs(title = paste0(config$display_name, " - Top 20 Cell Types by Significant Genes"),
             subtitle = "FDR < 0.05 (all tiers)",
             x = "Cell Type", y = "Number of Significant Genes") +
        scale_fill_viridis_d(option = "turbo")
      save_plot(p_ct, file.path(figures_dir, "barplot_sig_genes_by_celltype_top20.png"), w = 10, h = 8)
    }
    
    # 4d. Stacked bar: up vs down regulated by timepoint
    tp_updown <- sig_summary %>%
      group_by(timepoint) %>%
      summarise(Up = sum(n_up), Down = sum(n_down), .groups = "drop") %>%
      pivot_longer(cols = c(Up, Down), names_to = "Direction", values_to = "Count")
    
    if (sum(tp_updown$Count) > 0) {
      tp_order <- tp_updown %>% 
        select(timepoint) %>% distinct() %>%
        mutate(tp_num = as.numeric(gsub("[^0-9]", "", timepoint)))
      tp_order$tp_num[is.na(tp_order$tp_num)] <- Inf
      tp_order <- tp_order %>% arrange(tp_num) %>% pull(timepoint)
      
      p_updown <- ggplot(tp_updown, aes(x = factor(timepoint, levels = tp_order),
                                        y = Count, fill = Direction)) +
        geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
        theme_bw(base_size = 12) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(title = paste0(config$display_name, " - Up/Down Regulated Genes by Timepoint"),
             subtitle = "FDR < 0.05, |logFC| > 1",
             x = "Timepoint", y = "Number of Genes") +
        scale_fill_manual(values = c("Up" = "#E64B35", "Down" = "#4DBBD5"))
      save_plot(p_updown, file.path(figures_dir, "barplot_up_down_by_timepoint.png"), w = 8, h = 6)
    }
    
    # 4e. Scatter plot: mean |logFC| vs n significant genes per celltype/timepoint
    sig_summary_filtered <- sig_summary %>% filter(n_sig_05_all > 0 | mean_abs_logFC > 0)
    if (nrow(sig_summary_filtered) > 0) {
      p_scatter <- ggplot(sig_summary_filtered, aes(x = n_sig_05_all, y = mean_abs_logFC, color = timepoint)) +
        geom_point(size = 3, alpha = 0.7)
      
      if (nrow(sig_summary_filtered) > 5) {
        label_data <- sig_summary_filtered %>% 
          filter(n_sig_05_all > quantile(n_sig_05_all, 0.9, na.rm = TRUE) | 
                   mean_abs_logFC > quantile(mean_abs_logFC, 0.9, na.rm = TRUE))
        if (nrow(label_data) > 0) {
          p_scatter <- p_scatter + ggrepel::geom_text_repel(
            data = label_data, aes(label = celltype), size = 2.5, max.overlaps = 15)
        }
      }
      
      p_scatter <- p_scatter +
        theme_bw(base_size = 12) +
        labs(title = paste0(config$display_name, " - Effect Size vs Significance"),
             x = "Number of Significant Genes (FDR < 0.05, all)",
             y = "Mean |logFC|", color = "Timepoint") +
        scale_color_brewer(palette = "Set1")
      save_plot(p_scatter, file.path(figures_dir, "scatter_effect_vs_significance.png"), w = 10, h = 8)
    }
    
    # 4f. Tile plot showing analysis coverage
    all_combos <- expand.grid(
      timepoint = struc$tp_sorted, celltype = struc$all_celltypes, stringsAsFactors = FALSE)
    completed_combos <- sig_summary %>% select(timepoint, celltype) %>% mutate(status = "Completed")
    gene_skipped_df <- skipped_df %>% filter(celltype != "ALL (proportion)")
    if (nrow(gene_skipped_df) > 0) {
      skipped_combos <- gene_skipped_df %>% select(timepoint, celltype) %>% mutate(status = "Skipped")
    } else {
      skipped_combos <- data.frame(timepoint = character(0), celltype = character(0), status = character(0))
    }
    
    coverage_df <- all_combos %>%
      left_join(bind_rows(completed_combos, skipped_combos), by = c("timepoint", "celltype")) %>%
      mutate(status = ifelse(is.na(status), "No Data", status))
    
    p_coverage <- ggplot(coverage_df, aes(x = timepoint, y = celltype, fill = status)) +
      geom_tile(color = "white", linewidth = 0.5) +
      theme_bw(base_size = 10) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            axis.text.y = element_text(size = 6)) +
      labs(title = paste0(config$display_name, " - Analysis Coverage"),
           x = "Timepoint", y = "Cell Type", fill = "Status") +
      scale_fill_manual(values = c("Completed" = "#4DBBD5", "Skipped" = "#E64B35", "No Data" = "grey90"))
    save_plot(p_coverage, file.path(figures_dir, "tile_analysis_coverage.png"), 
              w = max(8, 4 + length(unique(coverage_df$timepoint)) * 0.5),
              h = max(8, 4 + length(unique(coverage_df$celltype)) * 0.15))
    
    # 4g. Distribution of FDR values
    p_fdr <- ggplot(combined_df, aes(x = FDR)) +
      geom_histogram(bins = 50, fill = "steelblue", color = "black", alpha = 0.7) +
      geom_vline(xintercept = 0.05, color = "red", linetype = "dashed", linewidth = 1) +
      geom_vline(xintercept = 0.10, color = "orange", linetype = "dashed", linewidth = 1) +
      theme_bw(base_size = 12) +
      labs(title = paste0(config$display_name, " - FDR Distribution (All Tests)"),
           x = "FDR", y = "Frequency") +
      annotate("text", x = 0.08, y = Inf, label = "FDR=0.05", vjust = 2, color = "red", size = 3) +
      annotate("text", x = 0.13, y = Inf, label = "FDR=0.10", vjust = 2, color = "orange", size = 3)
    save_plot(p_fdr, file.path(figures_dir, "histogram_fdr_distribution.png"), w = 8, h = 6)
    
    # 4h. Distribution of logFC values
    p_logfc <- ggplot(combined_df, aes(x = logFC)) +
      geom_histogram(bins = 100, fill = "grey50", color = "black", alpha = 0.7) +
      geom_vline(xintercept = c(-1, 1), color = "red", linetype = "dashed", linewidth = 0.8) +
      theme_bw(base_size = 12) +
      labs(title = paste0(config$display_name, " - logFC Distribution (All Tests)"),
           x = "log2 Fold Change", y = "Frequency")
    save_plot(p_logfc, file.path(figures_dir, "histogram_logfc_distribution.png"), w = 8, h = 6)
    
    # 4i. Proportion analysis plots (if available)
    if (nrow(combined_prop_df) > 0) {
      prop_fig_dir <- file.path(figures_dir, "proportions")
      dir.create(prop_fig_dir, recursive = TRUE, showWarnings = FALSE)
      
      # Volcano-like plot for proportions (all timepoints combined)
      prop_vol <- combined_prop_df %>%
        mutate(neglog10P = -log10(PValue + 1e-300),
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
          aes(label = paste0(gene, " (", timepoint, ")")),
          size = 2.5, max.overlaps = 20, color = "black") +
        scale_color_manual(values = c("FDR + logFC" = "#E64B35", "FDR-only" = "#FFA500", "NS" = "grey70")) +
        facet_wrap(~ timepoint) +
        theme_bw(base_size = 12) +
        labs(title = paste0(config$display_name, " - Differential Cell Type Proportions"),
             x = "log2 Fold Change", y = "-log10(P-value)", color = "Significance")
      save_plot(p_prop_vol, file.path(prop_fig_dir, "volcano_proportions.png"), w = 14, h = 6)
      
      # Bar plot of significant proportions
      prop_sig <- combined_prop_df %>% filter(FDR < 0.05)
      if (nrow(prop_sig) > 0) {
        p_prop_bar <- ggplot(prop_sig, aes(x = reorder(gene, logFC), y = logFC, fill = logFC > 0)) +
          geom_bar(stat = "identity", alpha = 0.8) +
          coord_flip() +
          facet_wrap(~ timepoint, scales = "free_y") +
          theme_bw(base_size = 12) +
          theme(legend.position = "none") +
          labs(title = paste0(config$display_name, " - Significant Differential Proportions"),
               subtitle = "FDR < 0.05",
               x = "Cell Type", y = "log2 Fold Change") +
          scale_fill_manual(values = c("TRUE" = "#E64B35", "FALSE" = "#4DBBD5"))
        save_plot(p_prop_bar, file.path(prop_fig_dir, "barplot_sig_proportions.png"), w = 12, h = 6)
      }
    }
    
  } else {
    writeLines(c(
      paste0("No DE results available for ", config$display_name),
      "All analyses may have been skipped due to covariate issues.",
      "See skipped_analyses_summary.txt for details."
    ), file.path(group_summary_dir, "analysis_summary.txt"))
  }
  
  message("  -> Summary generated for ", config$display_name, " in: ", group_summary_dir)
  invisible(NULL)
}

# =============================================================================
# PROPORTION ANALYSIS: Run differential proportion for one timepoint
# =============================================================================
run_proportion_analysis_one_tp <- function(parent_dir, tp, config, batch_keys,
                                           output_dir, strict_batch_control,
                                           min_group_per_level, min_cov_per_level, min_resid_df) {
  
  prop_file <- file.path(parent_dir, tp, "celltype_proportions.csv")
  if (!file.exists(prop_file)) {
    return(list(success = FALSE, de_result = NULL,
                skipped_info = list(
                  timepoint = tp, celltype = "ALL (proportion)",
                  reason = "celltype_proportions.csv not found",
                  failed_covariates = character(0),
                  n_samples = "NA", group_counts = "NA")))
  }
  
  prop_data <- load_proportion_data(prop_file)
  if (is.null(prop_data)) {
    return(list(success = FALSE, de_result = NULL,
                skipped_info = list(
                  timepoint = tp, celltype = "ALL (proportion)",
                  reason = "Failed to load proportion data or insufficient samples",
                  failed_covariates = character(0),
                  n_samples = "NA", group_counts = "NA")))
  }
  
  # Load metadata matching these sample IDs
  meta_df <- load_proportion_metadata(parent_dir, tp, rownames(prop_data$prop_mat))
  if (is.null(meta_df)) {
    return(list(success = FALSE, de_result = NULL,
                skipped_info = list(
                  timepoint = tp, celltype = "ALL (proportion)",
                  reason = "Could not find matching metadata for proportion samples",
                  failed_covariates = character(0),
                  n_samples = nrow(prop_data$prop_mat), group_counts = "NA")))
  }
  
  # Align samples
  common <- intersect(rownames(prop_data$prop_mat), rownames(meta_df))
  if (length(common) < 3) {
    return(list(success = FALSE, de_result = NULL,
                skipped_info = list(
                  timepoint = tp, celltype = "ALL (proportion)",
                  reason = "Insufficient overlapping samples between proportions and metadata",
                  failed_covariates = character(0),
                  n_samples = length(common), group_counts = "NA")))
  }
  
  prop_mat <- prop_data$prop_mat[common, , drop = FALSE]
  meta_df <- meta_df[common, , drop = FALSE]
  
  # Extract group
  group <- config$extract_fn(meta_df)
  if (is.null(group) || length(unique(group)) < 2) {
    return(list(success = FALSE, de_result = NULL,
                skipped_info = list(
                  timepoint = tp, celltype = "ALL (proportion)",
                  reason = "Could not extract group or insufficient group levels for proportions",
                  failed_covariates = character(0),
                  n_samples = nrow(prop_mat), group_counts = "NA")))
  }
  
  group_counts <- table(group)
  if (any(group_counts < min_group_per_level)) {
    return(list(success = FALSE, de_result = NULL,
                skipped_info = list(
                  timepoint = tp, celltype = "ALL (proportion)",
                  reason = sprintf("Insufficient samples per group for proportions (min required: %d)", min_group_per_level),
                  failed_covariates = character(0),
                  n_samples = nrow(prop_mat),
                  group_counts = paste(names(group_counts), as.integer(group_counts), sep="=", collapse=", "))))
  }
  
  # Check batch covariates (strict mode)
  if (isTRUE(strict_batch_control) && !isTRUE(config$is_multigroup)) {
    batch_check <- check_all_batch_covariates(
      group = group, meta_df = meta_df, batch_keys = batch_keys,
      exclude_keys = config$exclude_from_cov, min_per_level = min_cov_per_level)
    
    if (!batch_check$can_model) {
      return(list(success = FALSE, de_result = NULL,
                  skipped_info = list(
                    timepoint = tp, celltype = "ALL (proportion)",
                    reason = "Cannot control for all batch covariates (proportions)",
                    failed_covariates = batch_check$failed_covariates,
                    n_samples = nrow(prop_mat),
                    group_counts = paste(names(group_counts), as.integer(group_counts), sep="=", collapse=", "))))
    }
  }
  
  # Prepare covariates
  cov_keys_use <- batch_keys[!batch_keys %in% config$exclude_from_cov]
  cov_df <- prepare_covariates(meta_df, cov_keys_use, exclude_key = NULL)
  
  filt <- apply_covariate_filter(prop_mat, group, cov_df)
  if (is.null(filt)) {
    return(list(success = FALSE, de_result = NULL,
                skipped_info = list(
                  timepoint = tp, celltype = "ALL (proportion)",
                  reason = "Covariate filtering removed too many samples (proportions)",
                  failed_covariates = character(0),
                  n_samples = nrow(prop_mat),
                  group_counts = paste(names(group_counts), as.integer(group_counts), sep="=", collapse=", "))))
  }
  
  prop_mat <- filt$expr
  group <- filt$group
  cov_df <- filt$cov
  
  group_counts <- table(group)
  if (any(group_counts < min_group_per_level)) {
    return(list(success = FALSE, de_result = NULL,
                skipped_info = list(
                  timepoint = tp, celltype = "ALL (proportion)",
                  reason = "Insufficient samples per group after covariate filtering (proportions)",
                  failed_covariates = character(0),
                  n_samples = nrow(prop_mat),
                  group_counts = paste(names(group_counts), as.integer(group_counts), sep="=", collapse=", "))))
  }
  
  # Build design
  ds <- build_safe_design(group, cov_df, min_resid_df = min_resid_df)
  design <- ds$design
  cov_used <- ds$cov_used
  
  if (is.null(design)) {
    return(list(success = FALSE, de_result = NULL,
                skipped_info = list(
                  timepoint = tp, celltype = "ALL (proportion)",
                  reason = "Design matrix build failed (proportions)",
                  failed_covariates = character(0),
                  n_samples = nrow(prop_mat),
                  group_counts = paste(names(group_counts), as.integer(group_counts), sep="=", collapse=", "))))
  }
  
  # Output directory for proportion results
  prop_outdir <- file.path(output_dir, "step1_cross_group", config$subdir, tp, "_proportions")
  dir.create(prop_outdir, recursive = TRUE, showWarnings = FALSE)
  
  # Log covariate decisions
  writeLines(c(
    paste0("Timepoint: ", tp),
    paste0("Analysis: Cell type proportions"),
    paste0("Comparison: ", config$name),
    paste0("N samples: ", nrow(prop_mat)),
    paste0("N cell types: ", ncol(prop_mat)),
    paste0("Cell types: ", paste(colnames(prop_mat), collapse = ", ")),
    paste0("Group counts: ", paste(names(group_counts), as.integer(group_counts), sep="=", collapse=", ")),
    "", "Covariate notes:", ds$notes, "",
    paste0("Covariates used: ", if (!is.null(cov_used) && ncol(cov_used) > 0) paste(colnames(cov_used), collapse = ", ") else "OFF")
  ), file.path(prop_outdir, "COVARIATE_DECISIONS.txt"))
  
  # Helper: run one contrast on proportions
  .run_prop_contrast <- function(level1, level2, coef_name, up_label, down_label) {
    level1_col <- paste0("group", level1)
    level2_col <- paste0("group", level2)
    if (!all(c(level1_col, level2_col) %in% colnames(design))) return(NULL)
    
    contrast_formula <- paste0(level2_col, " - ", level1_col)
    contrast <- makeContrasts(contrasts = contrast_formula, levels = design)
    colnames(contrast) <- coef_name
    
    res <- tryCatch(run_limma(prop_mat, design, contrast, coef_name), error = function(e) NULL)
    if (is.null(res)) return(NULL)
    
    res <- add_de_status_generic(res, up_label = up_label, down_label = down_label)
    
    # Two-tier significance
    sig05_all <- get_sig_genes(res, fdr_thresh = 0.05)
    sig05_strict <- get_sig_genes_strict(res, fdr_thresh = 0.05, logfc_thresh = 1)
    sig05_fdr_only <- get_sig_genes_fdr_only(res, fdr_thresh = 0.05, logfc_thresh = 1)
    sig10 <- get_sig_genes(res, fdr_thresh = 0.10)
    
    contrast_outdir <- file.path(prop_outdir, coef_name)
    dir.create(contrast_outdir, recursive = TRUE, showWarnings = FALSE)
    
    cov_label <- if (!is.null(cov_used) && ncol(cov_used) > 0) paste(colnames(cov_used), collapse = ", ") else "OFF"
    
    save_de_results(
      res, contrast_outdir,
      paste0("Proportions | ", config$display_name, " | ", coef_name, " | ", tp),
      nrow(prop_mat), table(group), cov_label,
      sig05_all, sig05_strict, sig05_fdr_only, sig10)
    
    # Figures for proportion analysis
    figdir <- file.path(contrast_outdir, "figures")
    dir.create(figdir, recursive = TRUE, showWarnings = FALSE)
    
    plot_color_map <- c("#E64B35", "#4DBBD5", "#FFA500", "grey70")
    names(plot_color_map) <- c(up_label, down_label, "FDR-only", "NS")
    
    anno_colors_list <- list()
    anno_colors_list[["Group"]] <- config$group_colors
    
    # Volcano
    make_volcano(res, paste0("Proportions (", tp, ") ", config$display_name, " | ", coef_name, " - Volcano"),
                 file.path(figdir, "volcano.png"), plot_color_map)
    # P-value histogram
    make_pval_hist(res, paste0("Proportions (", tp, ") - P-value Dist"),
                   file.path(figdir, "pvalue_distribution.png"))
    # DE summary bar
    if (length(sig05_all) > 0) {
      make_de_summary_bar(res, paste0("Proportions (", tp, ") - DE Summary"),
                          file.path(figdir, "de_summary_barplot.png"), plot_color_map)
    }
    
    # Boxplots for each cell type tested
    make_gene_boxplots_batch(
      colnames(prop_mat), prop_mat, group, res,
      file.path(figdir, "boxplots_all_celltypes"),
      group_colors_use = config$group_colors,
      max_n = ncol(prop_mat), label_type = "sig")
    
    # Heatmap of all proportions
    plot_mat <- t(prop_mat)
    if (ncol(prop_mat) >= 2) {
      hm_inputs <- prep_heatmap_inputs(plot_mat, colnames(prop_mat), group, "Group", anno_colors_list)
      if (!is.null(hm_inputs)) {
        render_heatmap_png(hm_inputs, file.path(figdir, "heatmap_all_proportions.png"),
                           paste0("Cell Type Proportions (", tp, ") ", config$display_name))
      }
    }
    
    res %>%
      mutate(timepoint = tp, comparison = config$name, contrast = coef_name, analysis_type = "proportion") %>%
      select(analysis_type, comparison, contrast, timepoint, everything())
  }
  
  # Run contrasts
  if (!isTRUE(config$is_multigroup)) {
    prop_res <- .run_prop_contrast(config$level1, config$level2, config$coef_name, config$up_label, config$down_label)
    if (!is.null(prop_res)) {
      return(list(success = TRUE, de_result = prop_res, skipped_info = NULL))
    }
  } else {
    res_list <- lapply(config$contrasts, function(cc) {
      .run_prop_contrast(cc$level1, cc$level2, cc$coef_name, cc$up_label, cc$down_label)
    })
    res_list <- res_list[!sapply(res_list, is.null)]
    if (length(res_list) > 0) {
      return(list(success = TRUE, de_result = do.call(rbind, res_list), skipped_info = NULL))
    }
  }
  
  list(success = FALSE, de_result = NULL,
       skipped_info = list(
         timepoint = tp, celltype = "ALL (proportion)",
         reason = "Limma failed for proportion analysis",
         failed_covariates = character(0),
         n_samples = nrow(prop_mat),
         group_counts = paste(names(group_counts), as.integer(group_counts), sep="=", collapse=", ")))
}

# =============================================================================
# UNIFIED STEP 1: Run DE for any comparison type - PARALLELIZED
# Now includes proportion analysis
# =============================================================================
run_step1_comparison <- function(parent_dir, output_dir, comparison_type,
                                 env_lib = "~/R_envs/differential_gene",
                                 batch_keys = c("Sex", "BMI category", "LC/Recovered", "age_cluster"),
                                 heatmap_gene_counts = c(10, 20, 30, 40, 50),
                                 make_boxplots = TRUE, max_boxplots = 100,
                                 min_group_per_level = 2,
                                 min_cov_per_level = 2,
                                 min_resid_df = 2,
                                 strict_batch_control = TRUE) {
  
  setup_pipeline_env(env_lib)
  config <- get_comparison_config(comparison_type)
  struc <- discover_structure(parent_dir)
  
  message("\n", strrep("=", 70))
  message("COMPARISON: ", config$display_name)
  message("Found timepoints: ", paste(struc$timepoints, collapse = ", "))
  message("Found ", length(struc$all_celltypes), " unique cell types")
  message(strrep("=", 70))
  
  step1_dir <- file.path(output_dir, "step1_cross_group", config$subdir)
  
  # =========================================================================
  # GENE EXPRESSION ANALYSIS (parallelized over celltype x timepoint)
  # =========================================================================
  tasks <- list()
  for (tp in struc$timepoints) {
    for (ct in struc$celltype_map[[tp]]) {
      tasks[[length(tasks) + 1]] <- list(tp = tp, ct = ct)
    }
  }
  message(sprintf("Processing %d cell type x timepoint combinations in parallel (gene expression)...", length(tasks)))
  
  all_outputs <- future_lapply(tasks, function(task) {
    tp <- task$tp
    ct <- task$ct
    
    result <- list(success = FALSE, de_result = NULL, skipped_info = NULL)
    
    tryCatch({
      dat <- load_celltype_data(file.path(parent_dir, tp, ct))
      if (is.null(dat)) {
        result$skipped_info <- list(
          timepoint = tp, celltype = ct,
          reason = "Data loading failed or insufficient samples",
          failed_covariates = character(0),
          n_samples = "NA", group_counts = "NA")
        return(result)
      }
      
      group <- config$extract_fn(dat$meta)
      if (is.null(group) || length(unique(group)) < 2) {
        result$skipped_info <- list(
          timepoint = tp, celltype = ct,
          reason = "Could not extract group or insufficient group levels",
          failed_covariates = character(0),
          n_samples = nrow(dat$expr), group_counts = "NA")
        return(result)
      }
      
      group_counts <- table(group)
      if (any(group_counts < min_group_per_level)) {
        result$skipped_info <- list(
          timepoint = tp, celltype = ct,
          reason = sprintf("Insufficient samples per group (min required: %d)", min_group_per_level),
          failed_covariates = character(0),
          n_samples = nrow(dat$expr),
          group_counts = paste(names(group_counts), as.integer(group_counts), sep="=", collapse=", "))
        return(result)
      }
      
      outdir_base <- file.path(step1_dir, tp, ct)
      dir.create(outdir_base, recursive = TRUE, showWarnings = FALSE)
      
      # Strict batch check
      if (isTRUE(strict_batch_control) && !isTRUE(config$is_multigroup)) {
        batch_check <- check_all_batch_covariates(
          group = group, meta_df = dat$meta, batch_keys = batch_keys,
          exclude_keys = config$exclude_from_cov, min_per_level = min_cov_per_level)
        
        if (!batch_check$can_model) {
          writeLines(c(
            "SKIPPED: Cannot control for all batch covariates",
            paste0("Comparison: ", config$name),
            paste0("Timepoint: ", tp),
            paste0("Celltype: ", ct),
            paste0("N samples: ", nrow(dat$expr)),
            paste0("Group counts: ", paste(names(group_counts), as.integer(group_counts), sep="=", collapse=", ")),
            "", "Batch covariate check details:", batch_check$details, "",
            paste0("Failed covariates: ", paste(batch_check$failed_covariates, collapse=", "))
          ), file.path(outdir_base, "SKIP_REASON.txt"))
          
          result$skipped_info <- list(
            timepoint = tp, celltype = ct,
            reason = "Cannot control for all batch covariates",
            failed_covariates = batch_check$failed_covariates,
            n_samples = nrow(dat$expr),
            group_counts = paste(names(group_counts), as.integer(group_counts), sep="=", collapse=", "))
          return(result)
        }
      }
      
      cov_keys_use <- batch_keys[!batch_keys %in% config$exclude_from_cov]
      cov_df <- prepare_covariates(dat$meta, cov_keys_use, exclude_key = NULL)
      
      filt <- apply_covariate_filter(dat$expr, group, cov_df)
      if (is.null(filt)) {
        result$skipped_info <- list(
          timepoint = tp, celltype = ct,
          reason = "Covariate filtering removed too many samples",
          failed_covariates = character(0),
          n_samples = nrow(dat$expr),
          group_counts = paste(names(group_counts), as.integer(group_counts), sep="=", collapse=", "))
        return(result)
      }
      
      expr_mat <- filt$expr
      group <- filt$group
      cov_df <- filt$cov
      
      group_counts <- table(group)
      if (any(group_counts < min_group_per_level)) {
        result$skipped_info <- list(
          timepoint = tp, celltype = ct,
          reason = "Insufficient samples per group after covariate filtering",
          failed_covariates = character(0),
          n_samples = nrow(expr_mat),
          group_counts = paste(names(group_counts), as.integer(group_counts), sep="=", collapse=", "))
        return(result)
      }
      
      ds <- build_safe_design(group, cov_df, min_resid_df = min_resid_df)
      design <- ds$design
      cov_used <- ds$cov_used
      cov_notes <- ds$notes
      
      if (is.null(design)) {
        writeLines(c(
          "SKIPPED: design build failed",
          paste0("Comparison: ", config$name),
          paste0("Timepoint: ", tp),
          paste0("Celltype: ", ct),
          paste0("N samples: ", nrow(expr_mat)),
          paste0("Group counts: ", paste(names(group_counts), as.integer(group_counts), sep="=", collapse=", ")),
          "", "Notes:", cov_notes
        ), file.path(outdir_base, "SKIP_REASON.txt"))
        
        result$skipped_info <- list(
          timepoint = tp, celltype = ct,
          reason = "Design matrix build failed",
          failed_covariates = character(0),
          n_samples = nrow(expr_mat),
          group_counts = paste(names(group_counts), as.integer(group_counts), sep="=", collapse=", "))
        return(result)
      }
      
      writeLines(c(
        paste0("Timepoint: ", tp),
        paste0("Celltype: ", ct),
        paste0("Comparison: ", config$name),
        paste0("N samples: ", nrow(expr_mat)),
        paste0("Group counts: ", paste(names(group_counts), as.integer(group_counts), sep="=", collapse=", ")),
        "", "Covariate gating notes:", cov_notes, "",
        paste0("Covariates used: ", if (!is.null(cov_used) && ncol(cov_used) > 0) paste(colnames(cov_used), collapse = ", ") else "OFF"),
        paste0("Design cols: ", paste(colnames(design), collapse = ", "))
      ), file.path(outdir_base, "COVARIATE_DECISIONS.txt"))
      
      # Helper: run one contrast
      .run_one_contrast <- function(level1, level2, coef_name, up_label, down_label) {
        level1_col <- paste0("group", level1)
        level2_col <- paste0("group", level2)
        if (!all(c(level1_col, level2_col) %in% colnames(design))) return(NULL)
        
        contrast_formula <- paste0(level2_col, " - ", level1_col)
        contrast <- makeContrasts(contrasts = contrast_formula, levels = design)
        colnames(contrast) <- coef_name
        
        res <- tryCatch(run_limma(expr_mat, design, contrast, coef_name), error = function(e) NULL)
        if (is.null(res)) return(NULL)
        
        res <- add_de_status_generic(res, up_label = up_label, down_label = down_label)
        
        # Two-tier significance
        sig05_all <- get_sig_genes(res, fdr_thresh = 0.05)
        sig05_strict <- get_sig_genes_strict(res, fdr_thresh = 0.05, logfc_thresh = 1)
        sig05_fdr_only <- get_sig_genes_fdr_only(res, fdr_thresh = 0.05, logfc_thresh = 1)
        sig10 <- get_sig_genes(res, fdr_thresh = 0.10)
        
        outdir_this <- file.path(outdir_base, coef_name)
        dir.create(outdir_this, recursive = TRUE, showWarnings = FALSE)
        
        cov_label <- if (!is.null(cov_used) && ncol(cov_used) > 0) paste(colnames(cov_used), collapse = ", ") else "OFF"
        
        save_de_results(
          res, outdir_this,
          paste0(config$display_name, " | ", coef_name, " | ", tp, " | ", ct),
          nrow(expr_mat), table(group), cov_label,
          sig05_all, sig05_strict, sig05_fdr_only, sig10)
        
        fwrite(as.data.frame(design) %>% mutate(sample = rownames(design)) %>% select(sample, everything()),
               file.path(outdir_this, "design_matrix.csv"))
        
        anno_colors_list <- list()
        anno_colors_list[["Group"]] <- config$group_colors
        
        plot_color_map <- c("#E64B35", "#4DBBD5", "#FFA500", "grey70")
        names(plot_color_map) <- c(up_label, down_label, "FDR-only", "NS")
        
        run_plot_suite(
          res, expr_mat, group, file.path(outdir_this, "figures"),
          paste0(ct, " (", tp, ") ", config$display_name, " | ", coef_name),
          sig05_all = sig05_all, sig05_strict = sig05_strict, sig05_fdr_only = sig05_fdr_only,
          sig10 = sig10,
          color_map = plot_color_map, group_color_values = config$group_colors,
          anno_name = "Group", anno_colors_list = anno_colors_list,
          do_boxplots = make_boxplots, max_boxplots = max_boxplots,
          heatmap_gene_counts = heatmap_gene_counts)
        
        res %>%
          mutate(timepoint = tp, celltype = ct, comparison = config$name, contrast = coef_name) %>%
          select(comparison, contrast, timepoint, celltype, everything())
      }
      
      if (!isTRUE(config$is_multigroup)) {
        de_res <- .run_one_contrast(config$level1, config$level2, config$coef_name, config$up_label, config$down_label)
        if (!is.null(de_res)) {
          result$success <- TRUE
          result$de_result <- de_res
        }
      } else {
        res_list <- lapply(config$contrasts, function(cc) {
          .run_one_contrast(cc$level1, cc$level2, cc$coef_name, cc$up_label, cc$down_label)
        })
        res_list <- res_list[!sapply(res_list, is.null)]
        if (length(res_list) > 0) {
          result$success <- TRUE
          result$de_result <- do.call(rbind, res_list)
        }
      }
      
      return(result)
      
    }, error = function(e) {
      result$skipped_info <- list(
        timepoint = tp, celltype = ct,
        reason = paste("Error:", e$message),
        failed_covariates = character(0),
        n_samples = "NA", group_counts = "NA")
      return(result)
    })
    
  }, future.seed = TRUE)
  
  # Separate successful gene results from skipped
  successful_results <- lapply(all_outputs, function(x) if (x$success) x$de_result else NULL)
  successful_results <- successful_results[!sapply(successful_results, is.null)]
  
  skipped_list <- lapply(all_outputs, function(x) x$skipped_info)
  skipped_list <- skipped_list[!sapply(skipped_list, is.null)]
  
  message(sprintf("Gene DE: Completed %d/%d analyses successfully", length(successful_results), length(tasks)))
  message(sprintf("Gene DE: Skipped %d analyses", length(skipped_list)))
  
  # =========================================================================
  # PROPORTION ANALYSIS (per timepoint)
  # =========================================================================
  message("\n--- Running proportion analysis ---")
  
  prop_outputs <- future_lapply(struc$timepoints, function(tp) {
    tryCatch({
      run_proportion_analysis_one_tp(
        parent_dir = parent_dir, tp = tp, config = config,
        batch_keys = batch_keys, output_dir = output_dir,
        strict_batch_control = strict_batch_control,
        min_group_per_level = min_group_per_level,
        min_cov_per_level = min_cov_per_level,
        min_resid_df = min_resid_df)
    }, error = function(e) {
      list(success = FALSE, de_result = NULL,
           skipped_info = list(
             timepoint = tp, celltype = "ALL (proportion)",
             reason = paste("Error:", e$message),
             failed_covariates = character(0),
             n_samples = "NA", group_counts = "NA"))
    })
  }, future.seed = TRUE)
  
  prop_results <- lapply(prop_outputs, function(x) if (x$success) x$de_result else NULL)
  prop_results <- prop_results[!sapply(prop_results, is.null)]
  
  prop_skipped <- lapply(prop_outputs, function(x) x$skipped_info)
  prop_skipped <- prop_skipped[!sapply(prop_skipped, is.null)]
  
  message(sprintf("Proportion: Completed %d/%d timepoints successfully", length(prop_results), length(struc$timepoints)))
  message(sprintf("Proportion: Skipped %d timepoints", length(prop_skipped)))
  
  list(results = successful_results, skipped = skipped_list,
       proportion_results = prop_results, proportion_skipped = prop_skipped,
       config = config, struc = struc)
}

# =============================================================================
# MAIN STEP 1 RUNNER: Run all comparisons - PARALLELIZED
# =============================================================================
run_step1 <- function(parent_dir, output_dir,
                      env_lib = "~/R_envs/differential_gene",
                      batch_keys = c("Sex", "BMI category", "LC/Recovered", "age_cluster"),
                      comparisons = c("LC_vs_Recovered", "BMI_Elevated_vs_Normal", "Sex_Male_vs_Female", "Age_Young_Middle_Old"),
                      heatmap_gene_counts = c(10, 20, 30, 40, 50),
                      make_boxplots = TRUE, max_boxplots = 100,
                      n_workers = NULL,
                      memory_per_worker = "4GB",
                      strict_batch_control = TRUE) {
  
  setup_pipeline_env(env_lib)
  setup_parallel(n_workers = n_workers, memory_per_worker = memory_per_worker)
  struc <- discover_structure(parent_dir)
  
  message("\n", strrep("=", 70))
  message("STEP 1: Cross-group Differential Expression + Proportion Analysis (PARALLEL)")
  message("Comparisons to run: ", paste(comparisons, collapse = ", "))
  message("Strict batch control: ", strict_batch_control)
  message(strrep("=", 70))
  
  all_outputs_combined <- list()
  
  for (comp in comparisons) {
    message("\n\n", strrep("#", 70))
    message("Starting comparison: ", comp)
    message(strrep("#", 70))
    
    output <- run_step1_comparison(
      parent_dir = parent_dir, output_dir = output_dir, comparison_type = comp,
      env_lib = env_lib, batch_keys = batch_keys,
      heatmap_gene_counts = heatmap_gene_counts,
      make_boxplots = make_boxplots, max_boxplots = max_boxplots,
      strict_batch_control = strict_batch_control)
    
    all_outputs_combined[[comp]] <- output
  }
  
  summary_dir <- file.path(output_dir, "summary")
  dir.create(summary_dir, recursive = TRUE, showWarnings = FALSE)
  
  message("\n\n", strrep("=", 70))
  message("GENERATING COMPREHENSIVE SUMMARIES FOR EACH COMPARISON")
  message(strrep("=", 70))
  
  for (comp in names(all_outputs_combined)) {
    output <- all_outputs_combined[[comp]]
    config <- get_comparison_config(comp)
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
        step1_dir_root = output_dir
      )
    }, error = function(e) {
      message("  Warning: Error generating summary for ", config$display_name, ": ", e$message)
    })
  }
  
  # -----------------------------------------------------------------------------
  # Generate overall cross-comparison summary
  # -----------------------------------------------------------------------------
  message("\n--- Generating overall summary ---")
  
  overall_summary <- c(
    paste(rep("=", 70), collapse = ""),
    "OVERALL DIFFERENTIAL EXPRESSION + PROPORTION ANALYSIS SUMMARY",
    paste(rep("=", 70), collapse = ""),
    "",
    sprintf("Analysis date: %s", Sys.time()),
    sprintf("Input directory: %s", parent_dir),
    sprintf("Output directory: %s", output_dir),
    "",
    paste(rep("-", 70), collapse = ""),
    "COMPARISONS PERFORMED:",
    paste(rep("-", 70), collapse = "")
  )
  
  for (comp in names(all_outputs_combined)) {
    output <- all_outputs_combined[[comp]]
    config <- get_comparison_config(comp)
    n_completed <- length(output$results)
    n_skipped <- length(output$skipped)
    
    if (n_completed > 0) {
      combined <- do.call(rbind, output$results)
      n_sig_all <- sum(combined$FDR < 0.05, na.rm = TRUE)
      n_sig_strict <- sum(combined$FDR < 0.05 & abs(combined$logFC) > 1, na.rm = TRUE)
      n_sig_fdr_only <- sum(combined$FDR < 0.05 & abs(combined$logFC) <= 1, na.rm = TRUE)
    } else {
      n_sig_all <- 0; n_sig_strict <- 0; n_sig_fdr_only <- 0
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
                         sprintf("    Gene DE:"),
                         sprintf("      - Completed analyses: %d", n_completed),
                         sprintf("      - Skipped analyses: %d", n_skipped),
                         sprintf("      - Total sig (FDR<0.05): %d", n_sig_all),
                         sprintf("        FDR+logFC: %d | FDR-only: %d", n_sig_strict, n_sig_fdr_only),
                         sprintf("    Proportion DE:"),
                         sprintf("      - Completed timepoints: %d", n_prop_completed),
                         sprintf("      - Skipped timepoints: %d", n_prop_skipped),
                         sprintf("      - Significant proportions (FDR<0.05): %d", n_prop_sig)
    )
  }
  
  writeLines(overall_summary, file.path(summary_dir, "overall_summary.txt"))
  
  plan(sequential)
  message("\n\nStep 1 complete for all comparisons (gene expression + proportions).")
  message("Output directory: ", output_dir)
  message("Summary directory: ", summary_dir)
  
  invisible(all_outputs_combined)
}

# =============================================================================
# RUN STEP 1
# =============================================================================
if (sys.nframe() == 0) {
  run_step1(
    parent_dir = "/dcs07/hongkai/data/harry/result/long_covid/subset/sample_pseudobulk_differential_gene/B_cell",
    output_dir = "/dcs07/hongkai/data/harry/result/long_covid/subset/sample_pseudobulk_differential_gene/B_analysis/step1",
    env_lib    = "~/R_envs/differential_gene",
    batch_keys = c("Sex", "BMI category", "LC/Recovered", "age_cluster"),
    comparisons = c("LC_vs_Recovered", "BMI_Elevated_vs_Normal", "Sex_Male_vs_Female", "Age_Young_Middle_Old"),
    heatmap_gene_counts = c(10, 20, 30),
    make_boxplots = TRUE,
    max_boxplots = 100,
    n_workers = NULL,
    memory_per_worker = "30GB",
    strict_batch_control = TRUE
  )
}