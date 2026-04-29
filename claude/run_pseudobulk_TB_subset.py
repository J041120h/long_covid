#!/usr/bin/env python3
"""
Pseudobulk for T / B cell subset h5ad files.

This mirrors run_pseudobulk_LC_month.py but for the B/T sub-clustered data in
/dcs07/hongkai/data/harry/result/long_covid/subset/.

Key differences from the general pipeline:
  - Input h5ad has a single cell-type annotation column ``cell_type`` (no
    resolution loop).
  - Raw counts are stored in ``adata.layers['counts']`` (not ``adata.X``).
  - Outputs are written under
        /dcs07/hongkai/data/harry/result/long_covid/subset/{B,T}_LC_recovered_decouple/cell_type/
    to avoid overwriting the pre-existing
    ``sample_pseudobulk_differential_gene/`` analysis.

Usage:
    python run_pseudobulk_TB_subset.py --subset B
    python run_pseudobulk_TB_subset.py --subset T
    python run_pseudobulk_TB_subset.py --subset both        # default
"""

import os
import sys
import argparse
import itertools

import numpy as np
import pandas as pd
import scanpy as sc

# pseudobulk.py does `import rapids_singlecell` inside a try/except that only
# catches ImportError. On CPU-only or mismatched-driver nodes the rapids stack
# raises CUDARuntimeError / RuntimeError during import, which escapes that
# guard. Pre-import those modules here and short-circuit sys.modules on any
# failure so pseudobulk.py's `import ... except ImportError` path takes over.
for _gpu_mod in ("cupy", "rapids_singlecell"):
    try:
        __import__(_gpu_mod)
    except Exception as _e:
        print(f"[run_pseudobulk_TB_subset] disabling {_gpu_mod} "
              f"({type(_e).__name__}): {_e}", file=sys.stderr)
        sys.modules[_gpu_mod] = None

# Allow import of the original pseudobulk module without copying it.
sys.path.insert(0, "/users/hjiang/GenoDistance/long_covid")
from pseudobulk import compute_pseudobulk_per_celltype


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUBSET_CONFIGS = {
    "B": {
        "adata_path": "/dcs07/hongkai/data/harry/result/long_covid/subset/B_clean_subclusterclean.h5ad",
        "root":       "/dcs07/hongkai/data/harry/result/long_covid/subset/B_LC_recovered_decouple",
    },
    "T": {
        "adata_path": "/dcs07/hongkai/data/harry/result/long_covid/subset/T_clean_subclusterclean.h5ad",
        "root":       "/dcs07/hongkai/data/harry/result/long_covid/subset/T_LC_recovered_decouple",
    },
}

# Only one "resolution" for the subset analyses
RESOLUTIONS = ["cell_type"]
LC_GROUPS   = ["LC", "Recovered"]
MONTHS      = ["1", "3", "6"]

# Only one LC patient is available at month 3, which is not enough to estimate
# any LC x (Age/Month) interaction. Drop the (LC, 3) arm so merged/ and all
# downstream interaction analyses ignore it.
EXCLUDE_LC_MONTH = {("LC", "3")}

SAMPLE_COL    = "sample"
LC_COL        = "LC/Recovered"
MONTH_COL     = "month"
CELLTYPE_COL  = "manual_cell_type"
COUNTS_LAYER  = "counts"


# ---------------------------------------------------------------------------
# Helpers (identical in behavior to run_pseudobulk_LC_month.py)
# ---------------------------------------------------------------------------
def compute_cell_type_proportions(adata, sample_col, celltype_col):
    counts = pd.crosstab(adata.obs[sample_col], adata.obs[celltype_col])
    proportions = counts.div(counts.sum(axis=1), axis=0)
    proportions.index.name = sample_col
    return proportions


def compute_cell_type_proportions_with_metadata(adata, sample_col, celltype_col,
                                                 lc_col, month_col):
    proportions = compute_cell_type_proportions(adata, sample_col, celltype_col)
    metadata = adata.obs.groupby(sample_col).first()[[lc_col, month_col]].copy()
    cell_counts = adata.obs[sample_col].value_counts()
    metadata['total_cells'] = cell_counts
    result = proportions.join(metadata)
    celltype_cols = [col for col in proportions.columns]
    meta_cols = [lc_col, month_col, 'total_cells']
    result = result[meta_cols + celltype_cols]
    return result


def merge_lc_recovered_pseudobulk(lc_dir, recovered_dir, merged_dir, verbose=True):
    """Row-bind LC and Recovered per-cluster pseudobulk CSVs into a merged tree."""
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
    mask = adata.obs[MONTH_COL].astype(str) == month
    adata_month = adata[mask]
    if adata_month.n_obs == 0:
        print(f"  SKIP proportions for month {month} (0 cells)")
        return
    proportions_df = compute_cell_type_proportions_with_metadata(
        adata_month,
        sample_col=SAMPLE_COL,
        celltype_col=resolution,
        lc_col=LC_COL,
        month_col=MONTH_COL,
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cell_type_proportions.csv")
    proportions_df.to_csv(output_path)
    print(f"  Saved cell type proportions: {output_path}")
    print(f"    Samples: {len(proportions_df)}, "
          f"Cell types: {len(proportions_df.columns) - 3}")
    return proportions_df


def save_global_cell_type_proportions(adata, resolution, output_dir):
    proportions_df = compute_cell_type_proportions_with_metadata(
        adata,
        sample_col=SAMPLE_COL,
        celltype_col=resolution,
        lc_col=LC_COL,
        month_col=MONTH_COL,
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cell_type_proportions_all.csv")
    proportions_df.to_csv(output_path)
    print(f"  Saved global cell type proportions: {output_path}")
    print(f"    Samples: {len(proportions_df)}, "
          f"Cell types: {len(proportions_df.columns) - 3}")

    summary_path = os.path.join(output_dir, "cell_type_proportions_summary.csv")
    celltype_cols = [col for col in proportions_df.columns
                     if col not in [LC_COL, MONTH_COL, 'total_cells']]
    summary_list = []
    for (lc_group, month), group_df in proportions_df.groupby([LC_COL, MONTH_COL]):
        for ct in celltype_cols:
            summary_list.append({
                LC_COL: lc_group,
                MONTH_COL: month,
                'cell_type': ct,
                'mean_proportion':   group_df[ct].mean(),
                'std_proportion':    group_df[ct].std(),
                'median_proportion': group_df[ct].median(),
                'min_proportion':    group_df[ct].min(),
                'max_proportion':    group_df[ct].max(),
                'n_samples':         len(group_df),
            })
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved proportions summary: {summary_path}")
    return proportions_df


# ---------------------------------------------------------------------------
# Subset runner
# ---------------------------------------------------------------------------
def run_for_subset(subset_name, use_gpu=True):
    cfg = SUBSET_CONFIGS[subset_name]
    adata_path = cfg["adata_path"]
    root       = cfg["root"]

    print("=" * 60)
    print(f"Subset     : {subset_name}")
    print(f"AnnData in : {adata_path}")
    print(f"Output root: {root}")
    print("=" * 60)

    adata = sc.read_h5ad(adata_path)
    print(f"Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # Raw counts live in layers['counts']; swap into X so pseudobulk aggregates counts
    if COUNTS_LAYER not in adata.layers:
        raise RuntimeError(
            f"Layer '{COUNTS_LAYER}' not found in {adata_path}. "
            f"Available layers: {list(adata.layers.keys())}"
        )
    adata.X = adata.layers[COUNTS_LAYER]

    # Coerce month column to string (it is stored as category with numeric levels)
    adata.obs[MONTH_COL] = adata.obs[MONTH_COL].astype(str)

    # Make cell_type a string to avoid pandas groupby surprises on Categorical
    adata.obs[CELLTYPE_COL] = adata.obs[CELLTYPE_COL].astype(str)

    total = len(RESOLUTIONS) * len(LC_GROUPS) * len(MONTHS)
    idx = 0
    for resolution, lc_group, month in itertools.product(RESOLUTIONS, LC_GROUPS, MONTHS):
        idx += 1
        tag   = f"{resolution}/{lc_group}/month_{month}"
        label = f"[{idx}/{total}]  {tag}"

        if (lc_group, month) in EXCLUDE_LC_MONTH:
            print(f"\n{label}  ->  SKIP (excluded: insufficient samples)")
            continue

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
        print(f"  output  : {root}/{resolution}/{lc_group}/{month}/")
        print(f"{'=' * 60}")

        output_dir = os.path.join(root, resolution, lc_group, month)

        compute_pseudobulk_per_celltype(
            adata_subset,
            sample_col     = SAMPLE_COL,
            celltype_col   = CELLTYPE_COL,
            group_col      = None,
            keep_groups    = None,
            batch_col      = None,
            covariates     = None,
            normalize      = True,
            log_transform  = True,
            output_dir     = output_dir,
            prefix         = "pseudobulk",
            use_gpu        = use_gpu,
            verbose        = True,
            save_global_proportions_anndata = True,
        )

    # Merge LC + Recovered into merged/ tree
    print("\n" + "=" * 60)
    print("Building merged/ pseudobulk tree (LC + Recovered row-bound)...")
    print("=" * 60)
    for resolution in RESOLUTIONS:
        print(f"\n--- Resolution: {resolution} ---")
        merge_lc_recovered_pseudobulk(
            lc_dir        = os.path.join(root, resolution, "LC"),
            recovered_dir = os.path.join(root, resolution, "Recovered"),
            merged_dir    = os.path.join(root, resolution, "merged"),
            verbose       = True,
        )

    # Global + per-month cell-type proportion CSVs (downstream R needs them)
    print("\n" + "=" * 60)
    print("Generating merged cell type proportions...")
    print("=" * 60)
    for resolution in RESOLUTIONS:
        print(f"\n--- Resolution: {resolution} ---")
        merged_base_dir = os.path.join(root, resolution, "merged")
        save_global_cell_type_proportions(adata, resolution, merged_base_dir)
        for month in MONTHS:
            merged_month_dir = os.path.join(root, resolution, "merged", month)
            save_merged_cell_type_proportions(adata, resolution, month, merged_month_dir)

    print("\n" + "=" * 60)
    print(f"Subset {subset_name}: pseudobulk complete.")
    print(f"Root output: {root}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=["B", "T", "both"], default="both",
                        help="Which subset to process (default: both).")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration (falls back to CPU).")
    args = parser.parse_args()

    subsets = ["B", "T"] if args.subset == "both" else [args.subset]
    for s in subsets:
        run_for_subset(s, use_gpu=not args.no_gpu)


if __name__ == "__main__":
    main()
