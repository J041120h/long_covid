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
        merged/            <- written by this script (row-bound LC + Recovered)
          1/  3/  6/
            {cluster}/
              pseudobulk_expression.csv  (LC + Recovered row-bound)
              pseudobulk_metadata.csv
            cell_type_proportions.csv    <- cell type proportions per sample
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

import pandas as pd
import numpy as np
import scanpy as sc

# Allow direct import from same directory (pseudobulk.py lives alongside this file)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pseudobulk import compute_pseudobulk_per_celltype

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ADATA_PATH  = "/dcs07/hongkai/data/harry/result/long_covid/analysis/preprocess/adata_sample.h5ad"
ROOT        = "/dcs07/hongkai/data/harry/result/long_covid/LC_recovered_decouple"

RESOLUTIONS = ["cell_type_0.25", "cell_type_1"]
LC_GROUPS   = ["LC", "Recovered"]
MONTHS      = ["1", "3", "6"]   # stored as strings in adata.obs["month"]

# Column names in adata.obs
SAMPLE_COL    = "sample"
LC_COL        = "LC/Recovered"   # values: "LC", "Recovered"
MONTH_COL     = "month"          # values: "1", "3", "6" (category dtype)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def compute_cell_type_proportions(adata, sample_col, celltype_col):
    """
    Compute cell type proportions for each sample.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    sample_col : str
        Column name for sample identifiers.
    celltype_col : str
        Column name for cell type labels.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with samples as rows and cell types as columns,
        containing proportion values (0-1).
    """
    # Create a crosstab of sample x cell type counts
    counts = pd.crosstab(
        adata.obs[sample_col],
        adata.obs[celltype_col]
    )
    
    # Convert to proportions (normalize by row)
    proportions = counts.div(counts.sum(axis=1), axis=0)
    
    # Reset index to make sample a column
    proportions.index.name = sample_col
    
    return proportions


def compute_cell_type_proportions_with_metadata(adata, sample_col, celltype_col, 
                                                  lc_col, month_col):
    """
    Compute cell type proportions with associated metadata.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    sample_col : str
        Column name for sample identifiers.
    celltype_col : str
        Column name for cell type labels.
    lc_col : str
        Column name for LC/Recovered status.
    month_col : str
        Column name for month/timepoint.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with proportions and metadata columns.
    """
    # Compute proportions
    proportions = compute_cell_type_proportions(adata, sample_col, celltype_col)
    
    # Get metadata for each sample (take first occurrence)
    metadata = adata.obs.groupby(sample_col).first()[[lc_col, month_col]].copy()
    
    # Also compute total cell counts per sample
    cell_counts = adata.obs[sample_col].value_counts()
    metadata['total_cells'] = cell_counts
    
    # Merge proportions with metadata
    result = proportions.join(metadata)
    
    # Reorder columns: metadata first, then cell types
    celltype_cols = [col for col in proportions.columns]
    meta_cols = [lc_col, month_col, 'total_cells']
    result = result[meta_cols + celltype_cols]
    
    return result


def merge_lc_recovered_pseudobulk(lc_dir, recovered_dir, merged_dir, verbose=True):
    """
    Row-bind LC and Recovered per-cluster pseudobulk CSVs into a merged tree.

    Reads pseudobulk_expression.csv + pseudobulk_metadata.csv from
        {lc_dir}/{month}/{cluster}/  and  {recovered_dir}/{month}/{cluster}/
    and writes row-bound versions to
        {merged_dir}/{month}/{cluster}/

    Gene columns are intersected so the merged expression matrix is consistent.
    If only one side exists, the single-side files are copied as-is.
    """
    exclude_dirs = {"step1", "step2", "merged"}

    def _list_subdirs(d):
        if not os.path.isdir(d):
            return []
        return [
            name for name in os.listdir(d)
            if os.path.isdir(os.path.join(d, name)) and name not in exclude_dirs
        ]

    lc_tps  = _list_subdirs(lc_dir)
    rec_tps = _list_subdirs(recovered_dir)
    timepoints = sorted(set(lc_tps) | set(rec_tps))

    if not timepoints:
        if verbose:
            print(f"  [merge] No timepoint subdirectories found under "
                  f"{lc_dir} or {recovered_dir}")
        return

    for tp in timepoints:
        lc_cts  = _list_subdirs(os.path.join(lc_dir, tp))
        rec_cts = _list_subdirs(os.path.join(recovered_dir, tp))
        celltypes = sorted(set(lc_cts) | set(rec_cts))

        for ct in celltypes:
            lc_ef  = os.path.join(lc_dir,        tp, ct, "pseudobulk_expression.csv")
            lc_mf  = os.path.join(lc_dir,        tp, ct, "pseudobulk_metadata.csv")
            rec_ef = os.path.join(recovered_dir, tp, ct, "pseudobulk_expression.csv")
            rec_mf = os.path.join(recovered_dir, tp, ct, "pseudobulk_metadata.csv")

            has_lc  = os.path.exists(lc_ef)  and os.path.exists(lc_mf)
            has_rec = os.path.exists(rec_ef) and os.path.exists(rec_mf)
            if not has_lc and not has_rec:
                continue

            out_dir = os.path.join(merged_dir, tp, ct)
            os.makedirs(out_dir, exist_ok=True)

            if has_lc and has_rec:
                expr_lc  = pd.read_csv(lc_ef)
                expr_rec = pd.read_csv(rec_ef)
                meta_lc  = pd.read_csv(lc_mf)
                meta_rec = pd.read_csv(rec_mf)

                id_col = expr_lc.columns[0]
                common_genes = [
                    g for g in expr_lc.columns[1:] if g in set(expr_rec.columns[1:])
                ]

                if len(common_genes) < 10:
                    if verbose:
                        print(f"  [WARN] Too few common genes for {tp}/{ct} "
                              f"({len(common_genes)}) — skipping merge")
                    continue

                cols = [id_col] + common_genes
                expr_merged = pd.concat([expr_lc[cols], expr_rec[cols]],
                                        axis=0, ignore_index=True)
                meta_merged = pd.concat([meta_lc, meta_rec],
                                        axis=0, ignore_index=True, sort=False)

                if verbose:
                    print(f"  Merged {tp}/{ct}: "
                          f"{len(expr_merged)} samples, "
                          f"{len(common_genes)} genes")

            elif has_lc:
                expr_merged = pd.read_csv(lc_ef)
                meta_merged = pd.read_csv(lc_mf)
                if verbose:
                    print(f"  LC-only {tp}/{ct}: {len(expr_merged)} samples")
            else:
                expr_merged = pd.read_csv(rec_ef)
                meta_merged = pd.read_csv(rec_mf)
                if verbose:
                    print(f"  Recovered-only {tp}/{ct}: {len(expr_merged)} samples")

            expr_merged.to_csv(os.path.join(out_dir, "pseudobulk_expression.csv"),
                               index=False)
            meta_merged.to_csv(os.path.join(out_dir, "pseudobulk_metadata.csv"),
                               index=False)


def save_merged_cell_type_proportions(adata, resolution, month, output_dir):
    """
    Compute and save cell type proportions for merged (LC + Recovered) data.
    
    Parameters
    ----------
    adata : AnnData
        Full annotated data matrix.
    resolution : str
        Resolution column name (e.g., 'leiden_0.25').
    month : str
        Month/timepoint value.
    output_dir : str
        Directory to save the proportions CSV.
    """
    # Filter to this month (both LC and Recovered)
    mask = adata.obs[MONTH_COL].astype(str) == month
    adata_month = adata[mask]
    
    if adata_month.n_obs == 0:
        print(f"  SKIP proportions for month {month} (0 cells)")
        return
    
    # Compute proportions with metadata
    proportions_df = compute_cell_type_proportions_with_metadata(
        adata_month,
        sample_col=SAMPLE_COL,
        celltype_col=resolution,
        lc_col=LC_COL,
        month_col=MONTH_COL
    )
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, "cell_type_proportions.csv")
    proportions_df.to_csv(output_path)
    
    print(f"  Saved cell type proportions: {output_path}")
    print(f"    Samples: {len(proportions_df)}, Cell types: {len(proportions_df.columns) - 3}")
    
    return proportions_df


def save_global_cell_type_proportions(adata, resolution, output_dir):
    """
    Compute and save global cell type proportions across all timepoints.
    
    Parameters
    ----------
    adata : AnnData
        Full annotated data matrix.
    resolution : str
        Resolution column name (e.g., 'leiden_0.25').
    output_dir : str
        Directory to save the proportions CSV.
    """
    # Compute proportions with metadata for all samples
    proportions_df = compute_cell_type_proportions_with_metadata(
        adata,
        sample_col=SAMPLE_COL,
        celltype_col=resolution,
        lc_col=LC_COL,
        month_col=MONTH_COL
    )
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, "cell_type_proportions_all.csv")
    proportions_df.to_csv(output_path)
    
    print(f"  Saved global cell type proportions: {output_path}")
    print(f"    Samples: {len(proportions_df)}, Cell types: {len(proportions_df.columns) - 3}")
    
    # Also save summary statistics
    summary_path = os.path.join(output_dir, "cell_type_proportions_summary.csv")
    
    # Get cell type columns (exclude metadata)
    celltype_cols = [col for col in proportions_df.columns 
                    if col not in [LC_COL, MONTH_COL, 'total_cells']]
    
    # Compute summary by LC group and month
    summary_list = []
    for (lc_group, month), group_df in proportions_df.groupby([LC_COL, MONTH_COL]):
        for ct in celltype_cols:
            summary_list.append({
                LC_COL: lc_group,
                MONTH_COL: month,
                'cell_type': ct,
                'mean_proportion': group_df[ct].mean(),
                'std_proportion': group_df[ct].std(),
                'median_proportion': group_df[ct].median(),
                'min_proportion': group_df[ct].min(),
                'max_proportion': group_df[ct].max(),
                'n_samples': len(group_df)
            })
    
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved proportions summary: {summary_path}")
    
    return proportions_df


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

    # -----------------------------------------------------------------------
    # Merge LC + Recovered pseudobulk CSVs into the merged/ tree
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Building merged/ pseudobulk tree (LC + Recovered row-bound)...")
    print("=" * 60)

    for resolution in RESOLUTIONS:
        print(f"\n--- Resolution: {resolution} ---")
        merge_lc_recovered_pseudobulk(
            lc_dir        = os.path.join(ROOT, resolution, "LC"),
            recovered_dir = os.path.join(ROOT, resolution, "Recovered"),
            merged_dir    = os.path.join(ROOT, resolution, "merged"),
            verbose       = True,
        )

    # -----------------------------------------------------------------------
    # Generate merged cell type proportions for downstream analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating merged cell type proportions...")
    print("=" * 60)

    for resolution in RESOLUTIONS:
        print(f"\n--- Resolution: {resolution} ---")

        # Save global proportions (all months combined)
        merged_base_dir = os.path.join(ROOT, resolution, "merged")
        save_global_cell_type_proportions(adata, resolution, merged_base_dir)

        # Save per-month proportions
        for month in MONTHS:
            merged_month_dir = os.path.join(ROOT, resolution, "merged", month)
            save_merged_cell_type_proportions(adata, resolution, month, merged_month_dir)

    print("\n" + "=" * 60)
    print("All pseudobulk runs complete.")
    print(f"Root output: {ROOT}")
    print("=" * 60)
    print("\nGenerated files in merged folders:")
    print("  - {month}/{cluster}/pseudobulk_expression.csv (row-bound LC + Recovered)")
    print("  - {month}/{cluster}/pseudobulk_metadata.csv")
    print("  - {month}/cell_type_proportions.csv (per month)")
    print("  - cell_type_proportions_all.csv (all months)")
    print("  - cell_type_proportions_summary.csv (summary statistics)")
    print("=" * 60)