#!/usr/bin/env Rscript
# =============================================================================
# Step 2 INTERACTION Analysis runner for T / B cell SUBSET pseudobulk
# -----------------------------------------------------------------------------
# Sources the original step-2 interaction code (which in turn sources
# differential_gene_LC_month_step_1.R and differential_gene_differnet_step_2.R).
# The existing entry-point block is guarded by `if (sys.nframe() == 0)`, so
# sourcing only brings in function definitions.
#
# Then calls run_step2_interaction() with input/output paths pointing to
#     /dcs07/hongkai/data/harry/result/long_covid/subset/{B,T}_LC_recovered_decouple/cell_type/
#
# Writes to a fresh ``interaction_step2/`` folder under each subset root —
# existing results are not overwritten.
#
# Usage:
#   Rscript run_step2_interaction_TB_subset.R            # runs B and T
#   Rscript run_step2_interaction_TB_subset.R B          # only B subset
#   Rscript run_step2_interaction_TB_subset.R T          # only T subset
# =============================================================================

source("/users/hjiang/r/differential_gene_differnet_step_2_interaction.R")

SUBSET_ROOTS <- list(
  B = "/dcs07/hongkai/data/harry/result/long_covid/subset/B_LC_recovered_decouple",
  T = "/dcs07/hongkai/data/harry/result/long_covid/subset/T_LC_recovered_decouple"
)

RESOLUTIONS <- c("cell_type")

args <- commandArgs(trailingOnly = TRUE)
subsets <- if (length(args) == 0) names(SUBSET_ROOTS) else args

for (subset in subsets) {
  if (!subset %in% names(SUBSET_ROOTS)) {
    stop("Unknown subset: ", subset, ". Must be one of: ",
         paste(names(SUBSET_ROOTS), collapse = ", "))
  }

  root <- SUBSET_ROOTS[[subset]]
  message("\n", strrep("#", 72))
  message("[Interaction Step 2]  subset: ", subset, "   root: ", root)
  message(strrep("#", 72))

  for (res in RESOLUTIONS) {
    lc_dir        <- file.path(root, res, "LC")
    recovered_dir <- file.path(root, res, "Recovered")
    merged_dir    <- file.path(root, res, "merged")
    output_dir    <- file.path(root, res, "interaction_step2")

    if (!dir.exists(merged_dir)) {
      if (dir.exists(lc_dir) || dir.exists(recovered_dir)) {
        message("Building merged tree under ", merged_dir, " ...")
        merge_lc_recovered_pseudobulk(lc_dir, recovered_dir, merged_dir,
                                       verbose = TRUE)
      } else {
        message("SKIP ", subset, " / ", res,
                ": no LC/ or Recovered/ trees found under ", root, "/", res)
        next
      }
    }

    message("\n", strrep("=", 70))
    message("[Interaction Step 2]  ", subset, " / ", res)
    message("  parent_dir : ", merged_dir)
    message("  output_dir : ", output_dir)
    message(strrep("=", 70))

    run_step2_interaction(
      parent_dir          = merged_dir,
      output_dir          = output_dir,
      env_lib             = "~/R_envs/differential_gene",
      batch_keys          = c("Sex", "BMI category", "age_cluster"),
      heatmap_gene_counts = c(10, 20, 30),
      make_boxplots       = TRUE,
      max_boxplots        = 100,
      min_per_tp          = 2,
      min_resid_df        = 2,
      n_workers           = NULL,
      memory_per_worker   = "30GB",
      sig_mode            = "strict",
      logfc_thresh        = 0.5
    )
  }
}

message("\nAll step-2 interaction runs complete.")
