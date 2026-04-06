#!/usr/bin/env python3
"""
Improved pseudobulk computation (per-sample per-cell-type) WITH cell proportions:
- Per-sample per-cell-type pseudobulk aggregation
- Flexible sample filtering by group
- Optional normalization and ComBat batch correction
- Organized output: separate subfolder per cell type
- GPU-accelerated normalization (optional)
- NEW: compute/save cell-type proportions (CSV) and store in AnnData (.uns)

Outputs (in output_dir):
- cell_type subfolders:
    - pseudobulk.h5ad
    - pseudobulk_expression.csv
    - pseudobulk_metadata.csv   (includes pb.obs['n_cells'] for that cell type)
- NEW (global):
    - celltype_counts.csv       (cell_types x samples)
    - celltype_proportions.csv  (cell_types x samples)
    - OPTIONAL: celltype_proportions.h5ad (samples x cell_types)
"""

import os
import warnings
import contextlib
import io
import gc
from typing import Optional, Union, List, Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse, csr_matrix

# Try to import GPU libraries
try:
    import cupy as cp
    import rapids_singlecell as rsc
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# =============================================================================
# Utility Functions
# =============================================================================

def _as_list(x: Optional[Union[str, List[str]]]) -> List[str]:
    """Convert None/str/list to list."""
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def _sanitize_celltype_name(cell_type: str) -> str:
    """Sanitize a cell type string for filesystem paths."""
    return cell_type.replace("/", "_").replace(" ", "_")


def clear_gpu_memory():
    """Clear GPU memory if available."""
    if not GPU_AVAILABLE:
        return
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def to_gpu(adata: sc.AnnData) -> sc.AnnData:
    """Move AnnData to GPU."""
    if GPU_AVAILABLE:
        rsc.get.anndata_to_GPU(adata)
    return adata


def to_cpu(adata: sc.AnnData) -> sc.AnnData:
    """Move AnnData to CPU."""
    if GPU_AVAILABLE:
        rsc.get.anndata_to_CPU(adata)
    return adata


# =============================================================================
# Sample Filtering
# =============================================================================

def filter_samples(
    adata: sc.AnnData,
    sample_col: str = "sample",
    group_col: Optional[str] = None,
    keep_groups: Optional[Union[str, List[str]]] = None,
    verbose: bool = False
) -> sc.AnnData:
    """Filter samples based on group membership."""
    if group_col is None or keep_groups is None:
        if verbose:
            print("No sample filtering applied")
        return adata

    if group_col not in adata.obs.columns:
        if verbose:
            print(f"Warning: group column '{group_col}' not found, skipping filtering")
        return adata

    keep_groups_list = _as_list(keep_groups)

    mask = adata.obs[group_col].isin(keep_groups_list)
    n_cells_before = adata.n_obs
    n_samples_before = adata.obs[sample_col].nunique()

    adata_filtered = adata[mask].copy()

    n_cells_after = adata_filtered.n_obs
    n_samples_after = adata_filtered.obs[sample_col].nunique()

    if verbose:
        print(f"Sample filtering by '{group_col}' in {keep_groups_list}:")
        print(f"  Cells: {n_cells_before} -> {n_cells_after}")
        print(f"  Samples: {n_samples_before} -> {n_samples_after}")

    return adata_filtered


# =============================================================================
# Cell counts / proportions (GLOBAL)
# =============================================================================

def compute_celltype_counts_and_proportions(
    adata: sc.AnnData,
    sample_col: str = "sample",
    celltype_col: str = "cell_type"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cell-type counts and proportions per sample from cell-level AnnData.

    Returns
    -------
    counts_df : DataFrame (cell_types x samples)
    props_df  : DataFrame (cell_types x samples)
    """
    samples = sorted(adata.obs[sample_col].unique())
    cell_types = sorted(adata.obs[celltype_col].unique())

    counts = pd.crosstab(adata.obs[celltype_col], adata.obs[sample_col])
    counts = counts.reindex(index=cell_types, columns=samples, fill_value=0)

    totals = counts.sum(axis=0)
    totals[totals == 0] = 1  # avoid div-by-zero
    props = (counts / totals).astype(float)

    return counts, props


# =============================================================================
# Pseudobulk Aggregation (Per-Sample Per-Cell-Type)
# =============================================================================

def aggregate_to_pseudobulk_per_celltype(
    adata: sc.AnnData,
    sample_col: str = "sample",
    celltype_col: str = "cell_type",
    verbose: bool = False
) -> Dict[str, sc.AnnData]:
    """
    Aggregate single-cell counts to pseudobulk per sample AND per cell type.

    Returns
    -------
    Dict[str, sc.AnnData]
        Dictionary mapping cell_type -> pseudobulk AnnData
    """
    samples = sorted(adata.obs[sample_col].unique())
    cell_types = sorted(adata.obs[celltype_col].unique())
    n_samples = len(samples)
    n_cells = adata.n_obs

    if verbose:
        print(f"Aggregating {n_cells} cells -> {n_samples} samples x {len(cell_types)} cell types")

    sample_to_idx = {s: i for i, s in enumerate(samples)}
    pseudobulk_dict: Dict[str, sc.AnnData] = {}

    # Precompute sample-level metadata once (from full adata)
    sample_metadata = {}
    for col in adata.obs.columns:
        if col in [sample_col, celltype_col]:
            continue
        grouped = adata.obs.groupby(sample_col)[col].apply(lambda x: x.dropna().unique())
        if grouped.apply(lambda u: len(u) <= 1).all():
            sample_metadata[col] = grouped.apply(lambda u: u[0] if len(u) > 0 else np.nan)

    for ct in cell_types:
        ct_mask = adata.obs[celltype_col] == ct
        ct_adata = adata[ct_mask]
        n_ct_cells = ct_adata.n_obs

        if n_ct_cells == 0:
            if verbose:
                print(f"  {ct}: 0 cells, skipping")
            continue

        cell_to_sample = ct_adata.obs[sample_col].map(sample_to_idx).values

        indicator = csr_matrix(
            (np.ones(n_ct_cells, dtype=np.float32), (cell_to_sample, np.arange(n_ct_cells))),
            shape=(n_samples, n_ct_cells)
        )

        X = ct_adata.X
        if issparse(X):
            pb_matrix = (indicator @ X).toarray().astype(np.float32)
        else:
            pb_matrix = (indicator @ X).astype(np.float32)

        pb = sc.AnnData(
            X=pb_matrix,
            obs=pd.DataFrame(index=samples),
            var=adata.var.copy()
        )
        pb.obs.index.name = "sample"

        if sample_metadata:
            pb.obs = pb.obs.join(pd.DataFrame(sample_metadata), how="left")

        # Store per-sample cell counts for this cell type
        cell_counts = indicator.sum(axis=1).A.flatten()
        pb.obs["n_cells"] = cell_counts

        pseudobulk_dict[ct] = pb

        if verbose:
            print(f"  {ct}: {n_ct_cells} cells -> {n_samples} samples")

    if verbose:
        print(f"Created pseudobulk for {len(pseudobulk_dict)} cell types")

    return pseudobulk_dict


# =============================================================================
# Batch Correction
# =============================================================================

def apply_combat(
    adata: sc.AnnData,
    batch_col: Union[str, List[str]],
    covariates: Optional[List[str]] = None,
    verbose: bool = False
) -> bool:
    """Apply ComBat batch correction (in-place)."""
    batch_cols = _as_list(batch_col)
    batch_cols = [col for col in batch_cols if col in adata.obs.columns]

    if not batch_cols:
        if verbose:
            print("    No valid batch columns found, skipping correction")
        return False

    if len(batch_cols) == 1:
        batch_key = batch_cols[0]
        if verbose:
            print(f"    Using batch column: {batch_key}")
    else:
        batch_key = "_combined_batch_"
        adata.obs[batch_key] = adata.obs[batch_cols].astype(str).agg("|".join, axis=1)
        if verbose:
            print(f"    Combined {len(batch_cols)} batch columns: {batch_cols}")

    n_batches = adata.obs[batch_key].nunique()
    if n_batches <= 1:
        if verbose:
            print(f"    Only {n_batches} batch found, skipping correction")
        return False

    batch_counts = adata.obs[batch_key].value_counts()
    if batch_counts.min() < 2:
        if verbose:
            print(f"    Minimum batch size is {batch_counts.min()} (need >=2), skipping ComBat")
        return False

    try:
        if verbose:
            print(f"    Applying ComBat correction for {n_batches} batches")
            if covariates:
                print(f"      Preserving covariates: {covariates}")

        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sc.pp.combat(adata, key=batch_key, covariates=covariates, inplace=True)

        if verbose:
            print("    ComBat correction completed")
        return True

    except Exception as e:
        if verbose:
            print(f"    ComBat failed: {type(e).__name__}: {e}")
        return False


# =============================================================================
# Processing Pipeline
# =============================================================================

def process_pseudobulk(
    pb: sc.AnnData,
    cell_type: str,
    batch_col: Optional[Union[str, List[str]]] = None,
    covariates: Optional[Union[str, List[str]]] = None,
    normalize: bool = False,
    log_transform: bool = False,
    use_gpu: bool = False,
    verbose: bool = False
) -> sc.AnnData:
    """Process a single cell type's pseudobulk AnnData."""
    if verbose:
        print(f"  Processing {cell_type}:")

    sc.pp.filter_genes(pb, min_cells=1)
    if verbose:
        print(f"    After filtering: {pb.n_vars} genes")

    if normalize:
        if use_gpu and GPU_AVAILABLE:
            if verbose:
                print("    Normalizing to CPM (GPU-accelerated)")
            clear_gpu_memory()
            to_gpu(pb)
            rsc.pp.normalize_total(pb, target_sum=1e6)
            if log_transform:
                if verbose:
                    print("    Log-transforming (GPU-accelerated)")
                rsc.pp.log1p(pb)
            to_cpu(pb)
            clear_gpu_memory()
        else:
            if verbose:
                print("    Normalizing to CPM")
            sc.pp.normalize_total(pb, target_sum=1e6)
            if log_transform:
                if verbose:
                    print("    Log-transforming (log1p)")
                sc.pp.log1p(pb)

    if batch_col:
        covariate_list = [c for c in _as_list(covariates) if c in pb.obs.columns]
        apply_combat(pb, batch_col, covariate_list, verbose)

    return pb


# =============================================================================
# Save Functions
# =============================================================================

def save_pseudobulk(
    pb: sc.AnnData,
    cell_type: str,
    output_dir: str,
    prefix: str = "pseudobulk",
    verbose: bool = False
):
    """Save pseudobulk data for one cell type."""
    ct_dir = os.path.join(output_dir, _sanitize_celltype_name(cell_type))
    os.makedirs(ct_dir, exist_ok=True)

    h5ad_path = os.path.join(ct_dir, f"{prefix}.h5ad")
    pb.write_h5ad(h5ad_path)
    if verbose:
        print(f"    Saved AnnData: {h5ad_path}")

    csv_path = os.path.join(ct_dir, f"{prefix}_expression.csv")
    expr_df = pd.DataFrame(
        pb.X if not issparse(pb.X) else pb.X.toarray(),
        index=pb.obs.index,
        columns=pb.var.index
    )
    expr_df.to_csv(csv_path)
    if verbose:
        print(f"    Saved expression CSV: {csv_path}")

    if pb.obs.shape[1] > 0:
        meta_path = os.path.join(ct_dir, f"{prefix}_metadata.csv")
        pb.obs.to_csv(meta_path)
        if verbose:
            print(f"    Saved metadata CSV: {meta_path}")


def save_celltype_summary_tables(
    counts_df: pd.DataFrame,
    props_df: pd.DataFrame,
    output_dir: str,
    verbose: bool = False
):
    """Save global counts/proportions tables (cell_types x samples)."""
    os.makedirs(output_dir, exist_ok=True)

    counts_path = os.path.join(output_dir, "celltype_counts.csv")
    props_path = os.path.join(output_dir, "celltype_proportions.csv")

    counts_df.to_csv(counts_path)
    props_df.to_csv(props_path)

    if verbose:
        print(f"Saved global counts: {counts_path}")
        print(f"Saved global proportions: {props_path}")


def save_celltype_proportions_anndata(
    props_df: pd.DataFrame,
    output_dir: str,
    verbose: bool = False
):
    """
    OPTIONAL: Save proportions as an AnnData:
    - obs = samples
    - var = cell_types
    - X   = proportions (samples x cell_types)
    """
    os.makedirs(output_dir, exist_ok=True)
    X = props_df.T.values.astype(np.float32)  # samples x cell_types
    ad = sc.AnnData(
        X=X,
        obs=pd.DataFrame(index=props_df.columns),
        var=pd.DataFrame(index=props_df.index),
    )
    ad.obs.index.name = "sample"
    ad.var.index.name = "cell_type"
    ad.uns["note"] = "Cell-type proportions per sample (computed from cell-level adata.obs)."

    path = os.path.join(output_dir, "celltype_proportions.h5ad")
    ad.write_h5ad(path)

    if verbose:
        print(f"Saved proportions AnnData: {path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def compute_pseudobulk_per_celltype(
    adata: sc.AnnData,
    sample_col: str = "sample",
    celltype_col: str = "cell_type",
    group_col: Optional[str] = None,
    keep_groups: Optional[Union[str, List[str]]] = None,
    batch_col: Optional[Union[str, List[str]]] = None,
    covariates: Optional[Union[str, List[str]]] = None,
    output_dir: str = "./",
    prefix: str = "pseudobulk",
    normalize: bool = False,
    log_transform: bool = False,
    use_gpu: bool = False,
    verbose: bool = False,
    save_global_proportions_anndata: bool = True
) -> Dict[str, sc.AnnData]:
    """
    Compute per-sample per-cell-type pseudobulk + save cell-type proportions.

    NEW:
    - Computes global cell-type counts/proportions from the *filtered* adata
    - Saves:
        output_dir/celltype_counts.csv
        output_dir/celltype_proportions.csv
      and optionally:
        output_dir/celltype_proportions.h5ad
    - Stores in each pb AnnData:
        pb.uns["celltype_counts"]      (DataFrame cell_types x samples)
        pb.uns["celltype_proportions"] (DataFrame cell_types x samples)
    """
    if verbose:
        print("=== Pseudobulk Computation (Per-Sample Per-Cell-Type) ===")
        if use_gpu and GPU_AVAILABLE:
            print("GPU acceleration: ENABLED")
        elif use_gpu and not GPU_AVAILABLE:
            print("GPU acceleration: REQUESTED but not available (falling back to CPU)")
        print(f"Input: {adata.n_obs} cells x {adata.n_vars} genes")

    # Step 1: Filter samples (if requested)
    adata_filtered = filter_samples(
        adata,
        sample_col=sample_col,
        group_col=group_col,
        keep_groups=keep_groups,
        verbose=verbose
    )

    if verbose:
        print(f"\nProcessing: {adata_filtered.n_obs} cells from {adata_filtered.obs[sample_col].nunique()} samples")

    # NEW Step 1b: Compute global counts/proportions from filtered data
    counts_df, props_df = compute_celltype_counts_and_proportions(
        adata_filtered,
        sample_col=sample_col,
        celltype_col=celltype_col
    )
    save_celltype_summary_tables(counts_df, props_df, output_dir=output_dir, verbose=verbose)
    if save_global_proportions_anndata:
        save_celltype_proportions_anndata(props_df, output_dir=output_dir, verbose=verbose)

    # Step 2: Aggregate to pseudobulk per cell type
    if verbose:
        print("\nStep 1: Aggregating to pseudobulk...")

    pseudobulk_dict = aggregate_to_pseudobulk_per_celltype(
        adata_filtered,
        sample_col=sample_col,
        celltype_col=celltype_col,
        verbose=verbose
    )

    # Step 3: Process each cell type
    if verbose:
        print("\nStep 2: Processing cell types...")

    processed_dict: Dict[str, sc.AnnData] = {}

    for ct, pb in pseudobulk_dict.items():
        pb_processed = process_pseudobulk(
            pb,
            cell_type=ct,
            batch_col=batch_col,
            covariates=covariates,
            normalize=normalize,
            log_transform=log_transform,
            use_gpu=use_gpu,
            verbose=verbose
        )

        # NEW: store global proportions + counts in each AnnData
        pb_processed.uns["celltype_counts"] = counts_df
        pb_processed.uns["celltype_proportions"] = props_df

        processed_dict[ct] = pb_processed

    # Step 4: Save outputs
    if verbose:
        print("\nStep 3: Saving outputs...")

    os.makedirs(output_dir, exist_ok=True)

    for ct, pb in processed_dict.items():
        save_pseudobulk(
            pb,
            cell_type=ct,
            output_dir=output_dir,
            prefix=prefix,
            verbose=verbose
        )

    if verbose:
        print("\n=== Complete ===")
        print(f"Processed {len(processed_dict)} cell types")
        print(f"Output directory: {output_dir}")

    return processed_dict


if __name__ == "__main__":
    # Example usage
    adata = sc.read_h5ad(
        "/dcs07/hongkai/data/harry/result/long_covid/analysis/preprocess/adata_filtered.h5ad"
    )

    pseudobulk_dict = compute_pseudobulk_per_celltype(
        adata,
        sample_col="sample",
        celltype_col="cell_type",     # replace if needed
        group_col='month',            # optional
        keep_groups=["6"],            # optional
        batch_col=None,
        covariates=None,
        normalize=True,
        log_transform=True,
        output_dir= "/dcs07/hongkai/data/harry/result/long_covid/sample_pseudobulk_differential_analysis/different_time_point/6_month",
        prefix="pseudobulk",
        use_gpu=True,
        verbose=True,
        save_global_proportions_anndata=True
    )
