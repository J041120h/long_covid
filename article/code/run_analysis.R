#!/usr/bin/env Rscript
# =============================================================================
# Long COVID scRNA-seq: Comparison with Published Paper (Yin et al. 2024)
#
# Paper: "Long COVID manifests with T cell dysregulation, inflammation and
#         an uncoordinated adaptive immune response to SARS-CoV-2"
#         Nature Immunology 25, 218-225 (2024)
#
# METHODOLOGY NOTE:
#
#   All CyTOF protein markers in the paper are standard protein-coding genes
#   with a 1-to-1 gene-to-protein mapping (CXCR4 protein = CXCR4 gene, etc.).
#   Checking whether the same markers are elevated at the mRNA level in our
#   scRNA-seq is a valid and recognised approach ("transcriptomic validation
#   of CyTOF/proteomics findings"). mRNA and protein levels correlate
#   reasonably well for surface immune receptors.
#
#   The real comparability concern is the CELL POPULATION, not mRNA vs protein:
#     Paper: isolates SARS-CoV-2-specific T cells only (~1-5% of all T cells)
#            via ex-vivo peptide stimulation + cytokine sorting
#     Ours:  measures all T cells in PBMC without antigen selection
#   → If CXCR4/PD-1 signal is enriched in the 1-5% antigen-specific cells,
#     it will be diluted ~20-100x when averaged over all T cells in our data.
#
#   Tier 1 (green)  — paper used scRNA-seq mRNA, same cell population:
#     ALAS2, OR7D2, CXCR4/CCR6 mRNA (Extended Data Fig. 9)
#   Tier 2 (orange) — paper used CyTOF on antigen-specific cells; we test
#     the same genes at mRNA level in all cells (signal expected to be diluted):
#
# COHORT NOTE (added 2026-04):
#
#   The paper's bulk RNA-seq + scRNA-seq subset was FEMALE-ONLY at month 8
#   postinfection (T2). Quote: "We limited these studies to females because
#   individuals with high levels of OR7D2 or ALAS2 were mostly female (the top
#   five OR7D2 expressors were female, as were four of the top six ALAS2
#   expressors). For comparison, we included four randomly selected females
#   from the R specimens."
#
#   For valid comparison with the paper's RNA-seq findings, we restrict our
#   data to FEMALE samples only. The closest timepoint to paper's month 8 in
#   our cohort is 6_month (we report both 1_month and 6_month for context).
#
#   Cohort selection is controlled by the --cohort flag:
#     --cohort=female   (default)   paper-aligned analysis
#     --cohort=all                  full cohort (legacy behaviour)
#     --cohort=male                 male-only sensitivity check
#
# OUTPUTS:
#   figures/ 01_cell_proportions.png      -- claim 1 (proportion changes)
#            02_scrna_genes_heatmap.png   -- claims 2-4 (scRNA-comparable genes)
#            03_ALAS2_OR7D2_expression.png-- claims 2-3 (paper's primary scRNA hit)
#            04_tissue_homing_CD4.png    -- claim 4/6 (mRNA proxy for CyTOF)
#            05_exhaustion_CD8.png       -- claim 7 (mRNA proxy for CyTOF)
#            06_volcano_key_celltypes.png -- overall DEG landscape
#   tables/  all_DE_results.csv
#            scrna_comparable_genes_DE.csv
#            cytof_proxy_genes_DE.csv
#            proportion_summary.csv
#            significant_genes.csv
#   summary/ analysis_summary.txt
# =============================================================================

# -- Libraries ----------------------------------------------------------------
sys_lib  <- "/jhpce/shared/community/core/conda_R/4.4.x/R/lib64/R/site-library"
user_lib <- path.expand("~/R_envs/differential_gene")
.libPaths(c(user_lib, sys_lib))

suppressPackageStartupMessages({
  library(data.table); library(dplyr); library(tidyr); library(tibble)
  library(ggplot2);    library(ggrepel); library(pheatmap)
  library(RColorBrewer); library(scales)
})

# -- Cohort flag --------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
COHORT <- "female"  # paper-aligned default
for (a in args) {
  if (grepl("^--cohort=", a)) COHORT <- tolower(sub("^--cohort=", "", a))
}
stopifnot(COHORT %in% c("female", "all", "male"))
cat(sprintf("[cohort] running with COHORT = %s\n", COHORT))

# -- Paths --------------------------------------------------------------------
DE_ROOT  <- "/dcs07/hongkai/data/harry/result/long_covid/sample_pseudobulk_differential_analysis/step1/step1_cross_group/LC_vs_Recovered"
PROP_DIR <- "/dcs07/hongkai/data/harry/result/long_covid/sample_pseudobulk_differential_analysis/different_time_point"
META_CSV <- "/dcs07/hongkai/data/harry/result/long_covid/sample_pseudobulk_differential_analysis/Whole_Pseudobulk/required/pseudobulk_metadata.csv"

result_suffix <- if (COHORT == "all") "" else paste0("_", COHORT)
RES_ROOT <- paste0("/users/hjiang/GenoDistance/long_covid/article/results", result_suffix)
OUT_FIG  <- file.path(RES_ROOT, "figures")
OUT_TBL  <- file.path(RES_ROOT, "tables")
OUT_SUM  <- file.path(RES_ROOT, "summary")
for (d in c(OUT_FIG, OUT_TBL, OUT_SUM)) dir.create(d, recursive = TRUE, showWarnings = FALSE)

ARTICLE_ROOT <- "/users/hjiang/GenoDistance/long_covid/article"

TIMEPOINTS <- c("1_month", "6_month")
GROUP_PAL  <- c("LC" = "#E64B35", "Recovered" = "#4DBBD5")

COHORT_BANNER <- switch(
  COHORT,
  female = "[FEMALE-ONLY cohort | paper-aligned: paper used female-only bulk/scRNA-seq at 8mo postinfection]",
  male   = "[MALE-ONLY cohort | sensitivity check; paper did NOT analyse males in bulk/scRNA-seq]",
  all    = "[FULL cohort | paper used female-only — direct comparison is in the female-only run]"
)

# -- Gene sets by comparability tier ------------------------------------------

# TIER 1: Paper's actual scRNA-seq findings (Extended Data Fig. 9)
# These are the ONLY mRNA-level claims in the paper — directly comparable to us
SCRNA_GENES <- c(
  "ALAS2",   # Paper's primary DE gene in LC (Extended Data Fig. 9 volcano)
  "OR7D2",   # Paper's second primary DE gene in LC (Extended Data Fig. 9)
  "CXCR4",   # Validated at mRNA level in CD4+ T in Extended Data Fig. 9
  "CXCR5",   # Validated at mRNA level in Extended Data Fig. 9
  "CCR6",    # Validated at mRNA level in Extended Data Fig. 9
  "CCR4"     # Tissue-homing receptor, mRNA level
)

# TIER 2: CyTOF protein markers — tested as mRNA proxies (weaker comparison)
# Paper Fig. 2: CD4+ tissue-homing surface proteins
CYTOF_CD4_GENES <- c("CXCR4", "CXCR5", "CCR4", "CCR6", "ITGB1", "CD40LG")

# Paper Fig. 3: CD8+ exhaustion surface proteins
CYTOF_CD8_GENES <- c("PDCD1", "CTLA4", "TIGIT", "HAVCR2", "LAG3",
                      "GZMA", "GZMB", "GZMK", "PRF1")

ALL_GENES <- unique(c(SCRNA_GENES, CYTOF_CD4_GENES, CYTOF_CD8_GENES))

# =============================================================================
# LOAD DATA
# =============================================================================
cat("[1/6] Loading data...\n")

meta <- as.data.frame(fread(META_CSV))

# -- Cohort sample selection --------------------------------------------------
# Save the subset metadata so the user can see exactly which samples are in scope.
cohort_meta <- if (COHORT == "all") {
  meta
} else {
  cohort_label <- if (COHORT == "female") "Female" else "Male"
  meta[meta$Sex == cohort_label, ]
}
cohort_samples <- unique(as.character(cohort_meta$filename_sample_id))

cohort_csv <- file.path(ARTICLE_ROOT, paste0(COHORT, "_subset_samples.csv"))
fwrite(cohort_meta, cohort_csv)
cat(sprintf("  cohort=%s: %d sample-timepoint rows, %d unique subjects -> %s\n",
            COHORT, nrow(cohort_meta), length(cohort_samples), cohort_csv))

# Per-timepoint LC vs Rec counts in the cohort (sanity check for power)
for (tp in TIMEPOINTS) {
  tp_num <- as.integer(gsub("_month", "", tp))
  sub <- cohort_meta[cohort_meta$month == tp_num, ]
  n_lc  <- sum(sub$`LC/Recovered` == "LC",        na.rm = TRUE)
  n_rec <- sum(sub$`LC/Recovered` == "Recovered", na.rm = TRUE)
  cat(sprintf("    %s: n_LC=%d, n_Recovered=%d\n", tp, n_lc, n_rec))
}

# -- DE results --------------------------------------------------------------
# For COHORT="all": load pre-computed DE files (limma/edgeR pipeline).
# For COHORT="female"/"male": pre-computed DE used the WHOLE cohort and is not
# valid for a sex-stratified comparison. We recompute a cohort-restricted DE
# from the pseudobulk expression matrices using a per-gene t-test on the log
# pseudobulk values (logFC = mean(LC) - mean(Rec); FDR via BH). With ~4 vs 4
# samples this is underpowered for genome-wide FDR<0.05 hits but gives valid
# effect-size and direction estimates for the targeted paper-gene comparisons.
load_precomputed_de <- function() {
  de_list <- list()
  for (tp in TIMEPOINTS) {
    ct_dirs <- list.dirs(file.path(DE_ROOT, tp), recursive = FALSE, full.names = FALSE)
    ct_dirs <- ct_dirs[!grepl("^_", ct_dirs)]
    for (ct in ct_dirs) {
      path <- file.path(DE_ROOT, tp, ct, "LC_vs_Recovered", "DE_results.csv")
      if (!file.exists(path)) next
      df <- tryCatch(as.data.frame(fread(path)), error = function(e) NULL)
      if (is.null(df) || nrow(df) == 0) next
      df$timepoint <- tp
      df$cell_type <- trimws(gsub("_", " ", ct))
      de_list[[paste(tp, ct)]] <- df
    }
  }
  bind_rows(de_list)
}

compute_cohort_de <- function() {
  de_list <- list()
  for (tp in TIMEPOINTS) {
    tp_num <- gsub("_month", "", tp)
    ct_dirs <- list.dirs(file.path(PROP_DIR, tp), recursive = FALSE, full.names = TRUE)
    for (ct_dir in ct_dirs) {
      expr_path <- file.path(ct_dir, "pseudobulk_expression.csv")
      if (!file.exists(expr_path)) next
      df <- tryCatch(as.data.frame(fread(expr_path)), error = function(e) NULL)
      if (is.null(df) || nrow(df) == 0) next

      raw_ids   <- as.character(df[[1]])
      sample_id <- gsub(paste0("-M", tp_num, "$"), "", raw_ids)
      gene_mat  <- as.matrix(df[, -1, drop = FALSE])
      rownames(gene_mat) <- raw_ids

      keep <- sample_id %in% cohort_samples
      if (sum(keep) < 4) next
      gene_mat  <- gene_mat[keep, , drop = FALSE]
      sub_meta  <- cohort_meta[match(sample_id[keep], as.character(cohort_meta$filename_sample_id)), ]
      grp <- sub_meta$`LC/Recovered`

      lc_idx  <- which(grp == "LC")
      rec_idx <- which(grp == "Recovered")
      if (length(lc_idx) < 2 || length(rec_idx) < 2) next

      mean_lc  <- colMeans(gene_mat[lc_idx,  , drop = FALSE])
      mean_rec <- colMeans(gene_mat[rec_idx, , drop = FALSE])
      logFC    <- mean_lc - mean_rec  # pseudobulk values are already log-scale

      pvals <- vapply(seq_len(ncol(gene_mat)), function(j) {
        x <- gene_mat[lc_idx, j];  y <- gene_mat[rec_idx, j]
        if (length(unique(c(x, y))) < 2) return(NA_real_)
        tryCatch(t.test(x, y)$p.value, error = function(e) NA_real_)
      }, numeric(1))

      fdr <- p.adjust(pvals, method = "BH")
      ct_clean <- trimws(gsub("_", " ", basename(ct_dir)))

      de_list[[paste(tp, basename(ct_dir))]] <- data.frame(
        gene = colnames(gene_mat),
        logFC = logFC, AveExpr = (mean_lc + mean_rec) / 2,
        PValue = pvals, FDR = fdr,
        timepoint = tp, cell_type = ct_clean,
        stringsAsFactors = FALSE
      )
    }
    cat(sprintf("    DE recomputed for %s\n", tp))
  }
  bind_rows(de_list)
}

if (COHORT == "all") {
  cat("  Loading pre-computed DE (full-cohort limma pipeline)...\n")
  all_de <- load_precomputed_de()
} else {
  cat(sprintf("  Recomputing DE on %s subset from pseudobulk expression (t-test, ~few minutes)...\n", COHORT))
  all_de <- compute_cohort_de()
}
fwrite(all_de, file.path(OUT_TBL, "all_DE_results.csv"))

sig_genes <- all_de %>% filter(FDR < 0.05) %>% arrange(FDR)
fwrite(sig_genes, file.path(OUT_TBL, "significant_genes.csv"))

# Tier-1 genes
scrna_de <- all_de %>% filter(gene %in% SCRNA_GENES)
fwrite(scrna_de, file.path(OUT_TBL, "scrna_comparable_genes_DE.csv"))

# Tier-2 genes
cytof_de <- all_de %>% filter(gene %in% c(CYTOF_CD4_GENES, CYTOF_CD8_GENES))
fwrite(cytof_de, file.path(OUT_TBL, "cytof_proxy_genes_DE.csv"))

cat(sprintf("  %d total tests | %d FDR<0.05\n", nrow(all_de), nrow(sig_genes)))

# Load proportions
prop_list <- list()
for (tp in c("1_month", "3_month", "6_month")) {
  tp_num <- gsub("_month", "", tp)
  path   <- file.path(PROP_DIR, tp, "celltype_proportions.csv")
  if (!file.exists(path)) next
  mat <- as.data.frame(fread(path))
  rownames(mat) <- mat[[1]]; mat <- mat[, -1, drop = FALSE]
  long <- mat %>%
    rownames_to_column("cell_type") %>%
    pivot_longer(-cell_type, names_to = "sample_raw", values_to = "proportion") %>%
    mutate(sample_id = gsub(paste0("-M", tp_num, "$"), "", sample_raw),
           timepoint = tp)
  prop_list[[tp]] <- long
}
prop_all <- bind_rows(prop_list) %>%
  left_join(meta %>%
              mutate(sample_id = as.character(filename_sample_id)) %>%
              select(sample_id, `LC/Recovered`, Sex) %>%
              distinct(),  # one row per subject — LC/Sex are invariant across timepoints
            by = "sample_id") %>%
  filter(!is.na(`LC/Recovered`))

if (COHORT != "all") {
  sex_keep <- if (COHORT == "female") "Female" else "Male"
  prop_all <- prop_all %>% filter(Sex == sex_keep)
}

# Load pseudobulk expression for selected genes
expr_list <- list()
for (tp in TIMEPOINTS) {
  tp_num  <- gsub("_month", "", tp)
  for (ct_dir in list.dirs(file.path(PROP_DIR, tp), recursive=FALSE, full.names=TRUE)) {
    path <- file.path(ct_dir, "pseudobulk_expression.csv")
    if (!file.exists(path)) next
    df <- tryCatch(as.data.frame(fread(path)), error = function(e) NULL)
    if (is.null(df)) next
    id_col    <- colnames(df)[1]
    gene_cols <- intersect(colnames(df), ALL_GENES)
    if (length(gene_cols) == 0) next
    df[, c(id_col, gene_cols)] %>%
      pivot_longer(-all_of(id_col), names_to = "gene", values_to = "expression") %>%
      rename(sample_raw = !!id_col) %>%
      mutate(sample_id = gsub(paste0("-M", tp_num, "$"), "", sample_raw),
             cell_type  = trimws(gsub("_", " ", basename(ct_dir))),
             timepoint  = tp) -> long
    expr_list[[paste(tp, basename(ct_dir))]] <- long
  }
}
expr_all <- bind_rows(expr_list) %>%
  left_join(meta %>%
              mutate(sample_id = as.character(filename_sample_id)) %>%
              select(sample_id, `LC/Recovered`, Sex) %>%
              distinct(),  # one row per subject — LC/Sex are invariant across timepoints
            by = "sample_id") %>%
  filter(!is.na(`LC/Recovered`))

if (COHORT != "all") {
  sex_keep <- if (COHORT == "female") "Female" else "Male"
  expr_all <- expr_all %>% filter(Sex == sex_keep)
}

cat(sprintf("  After cohort filter: prop_all=%d rows | expr_all=%d rows | all_de=%d rows\n",
            nrow(prop_all), nrow(expr_all), nrow(all_de)))

# =============================================================================
# FIGURE 1 — Cell Type Proportions (scRNA-seq comparable, Claim 1)
# =============================================================================
cat("[2/6] Fig 1: Cell proportions...\n")

prop_12 <- prop_all %>% filter(timepoint %in% c("1_month", "6_month"))

wilcox_p <- prop_12 %>%
  group_by(cell_type, timepoint) %>%
  summarise(
    p_val = tryCatch({
      lc  <- proportion[`LC/Recovered` == "LC"]
      rec <- proportion[`LC/Recovered` == "Recovered"]
      if (length(lc) >= 3 && length(rec) >= 3) wilcox.test(lc, rec)$p.value else NA_real_
    }, error = function(e) NA_real_),
    y_max = max(proportion, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(label = case_when(
    !is.na(p_val) & p_val < 0.001 ~ "***",
    !is.na(p_val) & p_val < 0.01  ~ "**",
    !is.na(p_val) & p_val < 0.05  ~ "*",
    !is.na(p_val) & p_val < 0.10  ~ ".",
    TRUE ~ ""
  ))

prop_summary <- prop_12 %>%
  group_by(cell_type, timepoint, `LC/Recovered`) %>%
  summarise(mean = round(mean(proportion, na.rm=TRUE), 4),
            sd   = round(sd(proportion,   na.rm=TRUE), 4),
            n    = n(), .groups = "drop") %>%
  pivot_wider(names_from = `LC/Recovered`, values_from = c(mean, sd, n)) %>%
  left_join(wilcox_p %>% select(cell_type, timepoint, p_val), by = c("cell_type","timepoint"))
fwrite(prop_summary, file.path(OUT_TBL, "proportion_summary.csv"))

p1 <- ggplot(prop_12,
             aes(x = `LC/Recovered`, y = proportion, fill = `LC/Recovered`)) +
  geom_boxplot(alpha = 0.55, outlier.shape = NA, linewidth = 0.4, width = 0.5) +
  geom_jitter(aes(color = `LC/Recovered`), width = 0.14, size = 1.5, alpha = 0.75) +
  geom_text(data = wilcox_p, inherit.aes = FALSE,
            aes(x = 1.5, y = y_max * 1.12, label = label), size = 4.5) +
  facet_grid(cell_type ~ timepoint, scales = "free_y") +
  scale_fill_manual(values  = GROUP_PAL) +
  scale_color_manual(values = GROUP_PAL) +
  labs(title    = "Cell Type Proportions: Long COVID vs Recovered",
       subtitle = paste0(COHORT_BANNER,
                  "\nDirectly comparable to paper Extended Data Fig. 1 (scRNA-seq clusters)",
                  "\n*** p<0.001  ** p<0.01  * p<0.05  . p<0.10  (Wilcoxon)"),
       x = NULL, y = "Proportion of PBMCs") +
  theme_bw(base_size = 10) +
  theme(legend.position = "none",
        strip.text.y = element_text(size = 7, angle = 0),
        strip.text.x = element_text(size = 9),
        axis.text.x  = element_text(size = 8))

ggsave(file.path(OUT_FIG, "01_cell_proportions.png"),
       p1, width = 8, height = max(10, length(unique(prop_12$cell_type)) * 1.1), dpi = 150)
cat("  Saved: 01_cell_proportions.png\n")

# =============================================================================
# FIGURE 2 — ALAS2 and OR7D2: Paper's Primary scRNA-seq Hit (Claim 2-3)
# =============================================================================
cat("[3/6] Fig 2: ALAS2 / OR7D2 expression...\n")

alas2_or7d2 <- expr_all %>%
  filter(gene %in% c("ALAS2", "OR7D2")) %>%
  mutate(gene = factor(gene, levels = c("ALAS2", "OR7D2")))

# Wilcoxon p-value per gene x cell type x timepoint
alas2_pvals <- alas2_or7d2 %>%
  group_by(gene, cell_type, timepoint) %>%
  summarise(
    p_val = tryCatch({
      lc  <- expression[`LC/Recovered` == "LC"]
      rec <- expression[`LC/Recovered` == "Recovered"]
      if (length(lc) >= 3 && length(rec) >= 3) wilcox.test(lc, rec)$p.value else NA_real_
    }, error = function(e) NA_real_),
    y_max = max(expression, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(label = case_when(
    !is.na(p_val) & p_val < 0.05 ~ paste0("p=", round(p_val, 3)),
    !is.na(p_val) & p_val < 0.15 ~ paste0("p=", round(p_val, 2)),
    TRUE ~ ""
  ))

if (nrow(alas2_or7d2) > 0) {
  p2 <- ggplot(alas2_or7d2,
               aes(x = `LC/Recovered`, y = expression, fill = `LC/Recovered`)) +
    geom_boxplot(alpha = 0.5, outlier.shape = NA, linewidth = 0.4, width = 0.5) +
    geom_jitter(aes(color = `LC/Recovered`), width = 0.15, size = 1.3, alpha = 0.7) +
    geom_text(data = alas2_pvals %>% filter(label != ""),
              inherit.aes = FALSE,
              aes(x = 1.5, y = y_max * 1.1, label = label),
              size = 3, color = "red") +
    facet_grid(gene ~ cell_type + timepoint, scales = "free_y") +
    scale_fill_manual(values  = GROUP_PAL) +
    scale_color_manual(values = GROUP_PAL) +
    labs(title    = "ALAS2 and OR7D2 Expression: LC vs Recovered",
         subtitle = paste0(COHORT_BANNER,
                    "\nPaper's two primary scRNA-seq DE genes (Extended Data Fig. 9)",
                    "\nDirectly comparable: both paper and our data use scRNA-seq mRNA"),
         x = NULL, y = "Pseudobulk expression") +
    theme_bw(base_size = 9) +
    theme(legend.position = "bottom", legend.title = element_blank(),
          strip.text   = element_text(size = 7),
          axis.text.x  = element_text(size = 7, angle = 30, hjust = 1))

  n_ct <- length(unique(paste(alas2_or7d2$cell_type, alas2_or7d2$timepoint)))
  ggsave(file.path(OUT_FIG, "02_ALAS2_OR7D2_expression.png"),
         p2, width = max(14, n_ct * 1.4 + 3), height = 7, dpi = 150, limitsize = FALSE)
  cat("  Saved: 02_ALAS2_OR7D2_expression.png\n")
}

# =============================================================================
# FIGURE 3 — Tissue-Homing mRNA in CD4+ T (Claim 4: scRNA validated in paper)
# =============================================================================
cat("[4/6] Fig 3 & 4: Tissue-homing + exhaustion expression...\n")

# Find CD4+ T cell types in data
ct_all <- unique(expr_all$cell_type)
cd4_cts <- ct_all[grepl("CD4|Memory|Naive CD4", ct_all, ignore.case = TRUE)]
cd8_cts <- ct_all[grepl("CD8|Naive CD8", ct_all, ignore.case = TRUE)]
mono_cts <- ct_all[grepl("Monocyte|Mono", ct_all, ignore.case = TRUE)]

make_expr_plot <- function(data, genes, cell_types, title, subtitle, filename) {
  sub <- data %>%
    filter(gene %in% genes, cell_type %in% cell_types) %>%
    mutate(gene = factor(gene, levels = genes[genes %in% unique(.$gene)]))
  if (nrow(sub) == 0) { cat("  WARNING: no data for", filename, "\n"); return() }

  # p-values
  pvals <- sub %>%
    group_by(gene, cell_type, timepoint) %>%
    summarise(
      p = tryCatch({
        lc  <- expression[`LC/Recovered` == "LC"]
        rc  <- expression[`LC/Recovered` == "Recovered"]
        if (length(lc) >= 3 && length(rc) >= 3) wilcox.test(lc, rc)$p.value else NA_real_
      }, error = function(e) NA_real_),
      ymax = max(expression, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(lab = ifelse(!is.na(p) & p < 0.10, paste0("p=", round(p, 2)), ""))

  p <- ggplot(sub, aes(x = `LC/Recovered`, y = expression, fill = `LC/Recovered`)) +
    geom_boxplot(alpha = 0.5, outlier.shape = NA, linewidth = 0.4, width = 0.5) +
    geom_jitter(aes(color = `LC/Recovered`), width = 0.15, size = 1.2, alpha = 0.7) +
    geom_text(data = pvals %>% filter(lab != ""), inherit.aes = FALSE,
              aes(x = 1.5, y = ymax * 1.08, label = lab), size = 2.8, color = "red") +
    facet_grid(gene ~ cell_type + timepoint, scales = "free_y") +
    scale_fill_manual(values  = GROUP_PAL) +
    scale_color_manual(values = GROUP_PAL) +
    labs(title = title, subtitle = subtitle, x = NULL, y = "Pseudobulk expression") +
    theme_bw(base_size = 8) +
    theme(legend.position = "bottom", legend.title = element_blank(),
          strip.text   = element_text(size = 6.5),
          strip.text.y = element_text(face = "italic"),
          axis.text.x  = element_text(size = 7, angle = 30, hjust = 1))

  ng <- length(unique(sub$gene))
  nc <- length(unique(paste(sub$cell_type, sub$timepoint)))
  ggsave(file.path(OUT_FIG, filename),
         p, width = max(10, nc * 1.5 + 3), height = max(6, ng * 1.1 + 3),
         dpi = 150, limitsize = FALSE)
  cat("  Saved:", filename, "\n")
}

# Fig 3: Tissue-homing in CD4+ (mRNA proxy for paper's CyTOF + scRNA validation)
make_expr_plot(
  expr_all,
  genes      = CYTOF_CD4_GENES,
  cell_types = cd4_cts,
  title    = "Tissue-Homing Receptor mRNA in CD4+ T Cells: LC vs Recovered",
  subtitle = paste0(COHORT_BANNER,
              "\nPaper (Fig. 2) shows these as PROTEINS by CyTOF on antigen-specific CD4+ T cells",
              "\nExtended Data Fig. 9 validates CXCR4/CCR6 mRNA — our comparison is at mRNA level"),
  filename = "03_tissue_homing_CD4.png"
)

# Fig 4: CD8+ exhaustion + cytotoxic (mRNA proxy for paper's CyTOF)
make_expr_plot(
  expr_all,
  genes      = CYTOF_CD8_GENES,
  cell_types = cd8_cts,
  title    = "CD8+ T Cell Exhaustion + Cytotoxic Markers: LC vs Recovered",
  subtitle = paste0(COHORT_BANNER,
              "\nPaper (Fig. 3) shows PDCD1/CTLA4/TIGIT as PROTEINS by CyTOF on antigen-specific CD8+ T",
              "\nOur data: mRNA in all CD8+ T cells (no antigen-specificity filtering)"),
  filename = "04_exhaustion_cytotoxic_CD8.png"
)

# =============================================================================
# FIGURE 5 — logFC Heatmap: both tiers clearly labelled
# =============================================================================
cat("[5/6] Fig 5: logFC heatmap...\n")

tier_genes <- c(SCRNA_GENES, CYTOF_CD8_GENES[!CYTOF_CD8_GENES %in% SCRNA_GENES],
                CYTOF_CD4_GENES[!CYTOF_CD4_GENES %in% SCRNA_GENES])
tier_genes <- unique(tier_genes)

lfc_wide <- all_de %>%
  filter(gene %in% tier_genes) %>%
  mutate(col_id = paste0(gsub("_month", "M", timepoint), "\n", cell_type)) %>%
  select(gene, col_id, logFC) %>%
  pivot_wider(names_from = col_id, values_from = logFC, values_fill = 0) %>%
  column_to_rownames("gene")

gene_tier <- data.frame(
  Tier = ifelse(rownames(lfc_wide) %in% SCRNA_GENES,
                "1: scRNA-seq (directly comparable)",
                ifelse(rownames(lfc_wide) %in% CYTOF_CD4_GENES,
                       "2: CyTOF CD4+ protein proxy",
                       "2: CyTOF CD8+ protein proxy")),
  row.names = rownames(lfc_wide)
)
annot_col <- list(Tier = c(
  "1: scRNA-seq (directly comparable)" = "#2ca25f",
  "2: CyTOF CD4+ protein proxy"        = "#fdae6b",
  "2: CyTOF CD8+ protein proxy"        = "#9ecae1"
))

mat <- pmax(pmin(as.matrix(lfc_wide), 2.5), -2.5)

# Only label genes where |logFC| > 0.3
display_lab <- ifelse(abs(mat) > 0.3,
                      sprintf("%.2f", mat), "")
dim(display_lab) <- dim(mat); dimnames(display_lab) <- dimnames(mat)

png(file.path(OUT_FIG, "05_lfc_heatmap_both_tiers.png"),
    width  = max(1400, ncol(mat) * 80 + 400),
    height = max(900,  nrow(mat) * 42 + 280),
    res = 150)
tryCatch(
  pheatmap(mat,
           color             = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
           breaks            = seq(-2.5, 2.5, length.out = 101),
           display_numbers   = display_lab,
           number_color      = "black",
           fontsize_number   = 6.5,
           annotation_row    = gene_tier,
           annotation_colors = annot_col,
           cluster_rows      = FALSE,
           cluster_cols      = TRUE,
           fontsize          = 9,
           fontsize_row      = 9,
           fontsize_col      = 7,
           main = paste0(COHORT_BANNER,
                        "\nlogFC (LC vs Recovered) — Green rows: paper's actual scRNA-seq genes",
                        "\nOrange/Blue rows: CyTOF protein markers tested here as mRNA proxies",
                        "\nValues shown where |logFC| > 0.3"),
           border_color      = "grey85"),
  error = function(e) message("Heatmap error: ", e$message)
)
dev.off()
cat("  Saved: 05_lfc_heatmap_both_tiers.png\n")

# =============================================================================
# FIGURE 6 — Volcano plots (3 key cell types)
# =============================================================================
cat("[6/6] Fig 6: Volcano plots...\n")

ct_all_names <- unique(all_de$cell_type)
key_cts <- c(
  ct_all_names[grepl("Memory CD4", ct_all_names, ignore.case=TRUE)][1],
  ct_all_names[grepl("^CD8 T", ct_all_names, ignore.case=TRUE)][1],
  ct_all_names[grepl("CD14.*Mono", ct_all_names, ignore.case=TRUE)][1]
)
key_cts <- key_cts[!is.na(key_cts)]

volc_data <- all_de %>%
  filter(cell_type %in% key_cts) %>%
  mutate(
    neg_log10_fdr = pmin(-log10(FDR + 1e-300), 30),
    tier = case_when(
      gene %in% SCRNA_GENES    ~ "Paper scRNA-seq gene",
      gene %in% CYTOF_CD4_GENES | gene %in% CYTOF_CD8_GENES ~ "Paper CyTOF gene (mRNA proxy)",
      FDR < 0.05               ~ "Significant (FDR<0.05)",
      TRUE                     ~ "NS"
    ),
    label = ifelse(
      gene %in% tier_genes | FDR < 0.10, gene, NA_character_
    )
  )

color_map <- c(
  "Paper scRNA-seq gene"          = "#2ca25f",
  "Paper CyTOF gene (mRNA proxy)" = "#fdae6b",
  "Significant (FDR<0.05)"        = "#E64B35",
  "NS"                            = "grey75"
)

p6 <- ggplot(volc_data, aes(x = logFC, y = neg_log10_fdr)) +
  geom_point(data = subset(volc_data, tier == "NS"),
             color = "grey75", alpha = 0.3, size = 0.6) +
  geom_point(data = subset(volc_data, tier != "NS"),
             aes(color = tier, shape = timepoint), alpha = 0.9, size = 2.5) +
  geom_text_repel(aes(label = label, color = tier), size = 2.4,
                  max.overlaps = 15, segment.size = 0.25, na.rm = TRUE,
                  segment.color = "grey40", box.padding = 0.3,
                  show.legend = FALSE) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color="grey30", linewidth=0.35) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color="grey30", linewidth=0.35) +
  facet_wrap(~ cell_type, ncol = length(key_cts)) +
  scale_color_manual(values = color_map, name = NULL) +
  scale_shape_manual(values = c("1_month"=16, "6_month"=17), name = "Timepoint") +
  labs(title    = "Differential Expression Landscape: LC vs Recovered",
       subtitle = paste0(COHORT_BANNER,
                  "\nGreen = paper's scRNA-seq genes (directly comparable)",
                  "\nOrange = paper's CyTOF protein markers tested as mRNA",
                  "\nCircle = 1 month  |  Triangle = 6 month",
                  if (COHORT != "all") "\nNOTE: cohort DE recomputed by t-test on log pseudobulk; underpowered with n~4 per group" else ""),
       x = "log2 Fold Change (LC vs Recovered)", y = expression(-log[10](FDR))) +
  theme_bw(base_size = 10) +
  theme(legend.position = "bottom",
        strip.text      = element_text(size = 9, face = "bold"))

ggsave(file.path(OUT_FIG, "06_volcano_key_celltypes.png"),
       p6, width = max(14, length(key_cts) * 5), height = 6.5, dpi = 150)
cat("  Saved: 06_volcano_key_celltypes.png\n")

# =============================================================================
# SUMMARY REPORT
# =============================================================================
cat("Writing summary...\n")

# ALAS2 / OR7D2 specific results
alas2_summary <- scrna_de %>%
  filter(gene %in% c("ALAS2", "OR7D2")) %>%
  select(gene, cell_type, timepoint, logFC, FDR) %>%
  mutate(direction = ifelse(logFC > 0, "UP in LC", "DOWN in LC"),
         logFC = round(logFC, 3),
         FDR   = signif(FDR, 3))

prop_sig <- prop_summary %>%
  filter(!is.na(p_val), p_val < 0.10) %>%
  arrange(p_val)

sink(file.path(OUT_SUM, "analysis_summary.txt"))
cat("===========================================================================\n")
cat("LONG COVID scRNA-seq: COMPARISON WITH YIN ET AL. 2024\n")
cat("Nature Immunology 25, 218-225 (2024)\n")
cat("===========================================================================\n\n")

cat(sprintf("COHORT: %s\n", toupper(COHORT)))
cat(sprintf("  %s\n", COHORT_BANNER))
cat(sprintf("  Subset metadata file: %s\n", cohort_csv))
cat(sprintf("  Subjects in cohort: %d\n", length(cohort_samples)))
for (tp in TIMEPOINTS) {
  tp_num <- as.integer(gsub("_month", "", tp))
  sub <- cohort_meta[cohort_meta$month == tp_num, ]
  n_lc  <- sum(sub$`LC/Recovered` == "LC", na.rm = TRUE)
  n_rec <- sum(sub$`LC/Recovered` == "Recovered", na.rm = TRUE)
  cat(sprintf("    %s: n_LC=%d, n_Recovered=%d\n", tp, n_lc, n_rec))
}
if (COHORT != "all") {
  cat("  DE source: t-test on log pseudobulk expression (cohort-restricted, recomputed in-script)\n")
} else {
  cat("  DE source: pre-computed limma/edgeR pipeline (full cohort)\n")
}
cat("\n")

cat("METHODOLOGICAL ALIGNMENT\n")
cat("---------------------------------------------------------------------------\n")
cat("TIER 1 — Directly comparable (paper used scRNA-seq mRNA, Extended Data Fig. 9):\n")
cat("  Genes: ALAS2, OR7D2, CXCR4, CXCR5, CCR6, CCR4\n\n")
cat("TIER 2 — Weak proxy only (paper used CyTOF PROTEIN on antigen-specific T cells):\n")
cat("  CD4+ T panel: CXCR4, CXCR5, CCR4, CCR6, ITGB1, CD40LG (paper Fig. 2)\n")
cat("  CD8+ T panel: PDCD1, CTLA4, TIGIT, HAVCR2, LAG3, GZMA, GZMB, GZMK, PRF1 (paper Fig. 3)\n")
cat("  NOTE: mRNA does not equal protein; our cells are not antigen-selected.\n\n")

cat("OVERALL DE RESULTS\n")
cat("---------------------------------------------------------------------------\n")
cat(sprintf("  Total tests: %d\n", nrow(all_de)))
cat(sprintf("  FDR < 0.05:  %d genes\n\n", nrow(sig_genes)))

cat("TIER 1: ALAS2 AND OR7D2 (paper's primary scRNA-seq hits)\n")
cat("---------------------------------------------------------------------------\n")
if (nrow(alas2_summary) > 0) {
  for (i in seq_len(nrow(alas2_summary))) {
    r <- alas2_summary[i, ]
    cat(sprintf("  %-6s | %-25s | %-7s | logFC=%+.3f | FDR=%.3f | %s\n",
                r$gene, r$cell_type, r$timepoint, r$logFC, r$FDR, r$direction))
  }
  n_up   <- sum(alas2_summary$logFC > 0)
  n_down <- sum(alas2_summary$logFC < 0)
  cat(sprintf("\n  Summary: %d cell-type tests UP in LC, %d DOWN in LC\n", n_up, n_down))
  cat(sprintf("  Paper found: ALAS2 UP in LC (bulk scRNA-seq)\n"))
} else {
  cat("  No ALAS2/OR7D2 data found in expression files.\n")
}

cat("\nCELL TYPE PROPORTIONS (directly comparable to paper Extended Data Fig. 1)\n")
cat("---------------------------------------------------------------------------\n")
for (tp in c("1_month", "6_month")) {
  sub <- prop_summary %>% filter(timepoint == tp) %>% arrange(p_val)
  cat(sprintf("\n%s:\n", tp))
  for (i in seq_len(nrow(sub))) {
    r <- sub[i, ]
    sig <- if (!is.na(r$p_val) && r$p_val < 0.05) " *" else
           if (!is.na(r$p_val) && r$p_val < 0.10) " ." else ""
    lc_v  <- if (!is.na(r$mean_LC))        round(r$mean_LC, 4)        else "NA"
    rec_v <- if (!is.na(r$mean_Recovered))  round(r$mean_Recovered, 4) else "NA"
    dir   <- if (!is.na(r$mean_LC) && !is.na(r$mean_Recovered))
               ifelse(r$mean_LC > r$mean_Recovered, "↑", "↓") else "?"
    pstr  <- if (!is.na(r$p_val)) sprintf("p=%.3f", r$p_val) else "p=NA"
    cat(sprintf("  %s %-28s LC=%-6s Rec=%-6s %s%s\n",
                dir, r$cell_type, lc_v, rec_v, pstr, sig))
  }
}

cat("\nTIER 2: CyTOF PROTEIN PROXIES (logFC direction only — weak evidence)\n")
cat("---------------------------------------------------------------------------\n")
cat("Paper Fig. 2 CD4+ genes (CXCR4/CXCR5/CCR4/CCR6 — tested as mRNA in Memory CD4+ T):\n")
cd4_sub <- cytof_de %>%
  filter(gene %in% CYTOF_CD4_GENES, grepl("Memory CD4|CD4", cell_type, ignore.case=TRUE)) %>%
  group_by(gene) %>%
  summarise(mean_lfc = round(mean(logFC), 3), .groups = "drop") %>%
  mutate(dir = ifelse(mean_lfc > 0, "UP in LC (consistent)", "DOWN in LC (inconsistent)"))
for (i in seq_len(nrow(cd4_sub))) {
  cat(sprintf("  %-10s mean logFC = %+.3f  %s\n",
              cd4_sub$gene[i], cd4_sub$mean_lfc[i], cd4_sub$dir[i]))
}
cat("\nPaper Fig. 3 CD8+ genes (exhaustion/cytotoxic — tested as mRNA in CD8+ T cells):\n")
cd8_sub <- cytof_de %>%
  filter(gene %in% CYTOF_CD8_GENES, grepl("CD8", cell_type, ignore.case=TRUE)) %>%
  group_by(gene) %>%
  summarise(mean_lfc = round(mean(logFC), 3), .groups = "drop") %>%
  mutate(dir = ifelse(mean_lfc > 0, "UP in LC", "DOWN in LC"))
for (i in seq_len(nrow(cd8_sub))) {
  expected <- if (cd8_sub$gene[i] == "GZMK") "DOWN" else "UP"
  consistent <- if ((cd8_sub$mean_lfc[i] > 0 && expected == "UP") ||
                    (cd8_sub$mean_lfc[i] < 0 && expected == "DOWN")) "(consistent)" else "(inconsistent)"
  cat(sprintf("  %-10s mean logFC = %+.3f  %s  paper expected: %s\n",
              cd8_sub$gene[i], cd8_sub$mean_lfc[i], consistent, expected))
}

cat("\nOUTPUT FILES\n")
cat("---------------------------------------------------------------------------\n")
cat("figures/01_cell_proportions.png         -- Cell proportion boxplots + p-values\n")
cat("figures/02_ALAS2_OR7D2_expression.png   -- Paper's primary scRNA-seq DE genes\n")
cat("figures/03_tissue_homing_CD4.png        -- Homing receptor mRNA in CD4+ T\n")
cat("figures/04_exhaustion_cytotoxic_CD8.png -- Exhaustion/cytotoxic mRNA in CD8+ T\n")
cat("figures/05_lfc_heatmap_both_tiers.png   -- logFC heatmap (tiers colour-coded)\n")
cat("figures/06_volcano_key_celltypes.png    -- Volcano for 3 key cell types\n")
cat("tables/all_DE_results.csv\n")
cat("tables/scrna_comparable_genes_DE.csv    -- Tier 1 genes only\n")
cat("tables/cytof_proxy_genes_DE.csv         -- Tier 2 genes only\n")
cat("tables/proportion_summary.csv\n")
cat("tables/significant_genes.csv\n")
cat("===========================================================================\n")
sink()

cat("\n=== Analysis complete ===\n")
cat(sprintf("Cohort: %s\nResults: %s/\n", COHORT, RES_ROOT))
