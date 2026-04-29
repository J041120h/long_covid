#!/usr/bin/env Rscript
# =============================================================================
# Step 1 INTERACTION Analysis runner for T / B cell SUBSET pseudobulk
# -----------------------------------------------------------------------------
# This script sources the original step-1 interaction code, which only defines
# functions when sourced (the top-level `if (sys.nframe() == 0)` entry point
# is skipped), and then invokes run_step1_interaction() with input/output paths
# pointing to
#     /dcs07/hongkai/data/harry/result/long_covid/subset/{B,T}_LC_recovered_decouple/cell_type/
#
# No existing results are overwritten: this writes to a fresh
# ``interaction_step1/`` folder under each subset root.
#
# Usage:
#   Rscript run_step1_interaction_TB_subset.R            # runs B and T
#   Rscript run_step1_interaction_TB_subset.R B          # only B subset
#   Rscript run_step1_interaction_TB_subset.R T          # only T subset
# =============================================================================

source("/users/hjiang/r/differential_gene_differnet_step_1_interaction.R")

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
  message("[Interaction Step 1]  subset: ", subset,  "   root: ", root)
  message(strrep("#", 72))

  for (res in RESOLUTIONS) {
    parent_dir <- file.path(root, res, "merged")
    output_dir <- file.path(root, res, "interaction_step1")

    if (!dir.exists(parent_dir)) {
      message("SKIP ", subset, " / ", res,
              ": merged pseudobulk not found at ", parent_dir)
      next
    }

    message("\n", strrep("=", 70))
    message("[Interaction Step 1]  ", subset, " / ", res)
    message("  parent_dir : ", parent_dir)
    message("  output_dir : ", output_dir)
    message(strrep("=", 70))

    run_step1_interaction(
      parent_dir             = parent_dir,
      output_dir             = output_dir,
      env_lib                = "~/R_envs/differential_gene",
      batch_keys             = c("Sex", "BMI category"),
      logfc_thresh           = 0.5,
      sig_mode               = "strict",
      min_per_cell           = 2,
      min_resid_df           = 2,
      min_gene_variance      = 1e-10,
      min_expressed_fraction = 0.1,
      max_boxplots           = 50,
      n_workers              = NULL,
      memory_per_worker      = "30GB",
      debug                  = FALSE
    )
  }
}

message("\nAll step-1 interaction runs complete.")
