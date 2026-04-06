#!/usr/bin/env python3
"""
Pseudobulk for LC/Recovered-separated LC_month analysis.

Generates one pseudobulk dataset per (resolution, LC_group, month) triple,
writing output to:
    {ROOT}/{resolution}/{LC_group}/{month}/{cluster}/

where {cluster} is the sanitized Leiden cluster label (e.g. "4", "9").

Output directory structure (per resolution):
    LC_recovered_decouple/
      leiden_0.25/
        LC/
          1/  3/  6/       <- timepoints; discovered by R downstream scripts
            {cluster}/
              pseudobulk.h5ad
              pseudobulk_expression.csv
              pseudobulk_metadata.csv
          step1/           <- written by differential_gene_LC_month_step_1.R
          step2/           <- written by differential_gene_LC_month_step_2.R
        Recovered/
          1/  3/  6/
            {cluster}/
              ...
        merged/            <- written by differential_gene_LC_month_step_1.R
          1/  3/  6/
            {cluster}/
              pseudobulk_expression.csv  (LC + Recovered row-bound)
              pseudobulk_metadata.csv
        step1/             <- combined LC+Recovered step1 output
        step2/             <- combined LC+Recovered step2 output
      leiden_1/
        (same structure)

Usage:
    python run_pseudobulk_LC_month.py
"""

import os
import sys
import itertools

import scanpy as sc

# Allow direct import from same directory (pseudobulk.py lives alongside this file)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pseudobulk import compute_pseudobulk_per_celltype

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ADATA_PATH  = "/dcs07/hongkai/data/harry/result/long_covid/analysis/preprocess/adata_sample.h5ad"
ROOT        = "/dcs07/hongkai/data/harry/result/long_covid/LC_recovered_decouple"

RESOLUTIONS = ["leiden_0.25", "leiden_1"]
LC_GROUPS   = ["LC", "Recovered"]
MONTHS      = ["1", "3", "6"]   # stored as strings in adata.obs["month"]

# Column names in adata.obs
SAMPLE_COL    = "sample"
LC_COL        = "LC/Recovered"   # values: "LC", "Recovered"
MONTH_COL     = "month"          # values: "1", "3", "6" (category dtype)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Loading AnnData from:", ADATA_PATH)
    adata = sc.read_h5ad(ADATA_PATH)
    print(f"Loaded: {adata.n_obs:,} cells  x  {adata.n_vars:,} genes")
    print(f"Resolutions : {RESOLUTIONS}")
    print(f"LC groups   : {LC_GROUPS}")
    print(f"Months      : {MONTHS}")
    print("=" * 60)

    total = len(RESOLUTIONS) * len(LC_GROUPS) * len(MONTHS)
    idx   = 0

    for resolution, lc_group, month in itertools.product(RESOLUTIONS, LC_GROUPS, MONTHS):
        idx  += 1
        tag   = f"{resolution}/{lc_group}/month_{month}"
        label = f"[{idx}/{total}]  {tag}"

        # -------------------------------------------------------------------
        # Filter to this (LC_group, month) combination
        # adata.obs[MONTH_COL] is category dtype; compare as string
        # -------------------------------------------------------------------
        mask = (
            (adata.obs[LC_COL].astype(str) == lc_group) &
            (adata.obs[MONTH_COL].astype(str) == month)
        )
        adata_subset = adata[mask].copy()

        n_cells   = adata_subset.n_obs
        n_samples = adata_subset.obs[SAMPLE_COL].nunique()

        if n_cells == 0:
            print(f"\n{label}  ->  SKIP (0 cells)")
            continue

        print(f"\n{'=' * 60}")
        print(f"{label}")
        print(f"  cells   : {n_cells:,}")
        print(f"  samples : {n_samples}")
        print(f"  output  : {ROOT}/{resolution}/{lc_group}/{month}/")
        print(f"{'=' * 60}")

        output_dir = os.path.join(ROOT, resolution, lc_group, month)

        compute_pseudobulk_per_celltype(
            adata_subset,
            sample_col                    = SAMPLE_COL,
            celltype_col                  = resolution,
            group_col                     = None,
            keep_groups                   = None,
            batch_col                     = None,
            covariates                    = None,
            normalize                     = True,
            log_transform                 = True,
            output_dir                    = output_dir,
            prefix                        = "pseudobulk",
            use_gpu                       = True,
            verbose                       = True,
            save_global_proportions_anndata = True,
        )

    print("\n" + "=" * 60)
    print("All pseudobulk runs complete.")
    print(f"Root output: {ROOT}")
    print("=" * 60)
