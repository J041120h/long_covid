#!/usr/bin/env python3
"""
Script to recover raw counts from original AnnData and add them to preprocessed AnnData.

This script takes a preprocessed AnnData (which lost raw counts during preprocessing)
and the original AnnData (with raw counts), then creates a new AnnData with:
- All processed layers, obsm, varm, obs, var, uns from preprocessed data
- Raw counts from the original data stored in .X (overwriting normalized data)
"""

import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

# =============================================================================
# USER CONFIGURATION - MODIFY THESE PATHS
# =============================================================================

# Path to preprocessed AnnData (missing raw counts)
PREPROCESSED_PATH = "/dcs07/hongkai/data/jiatong/hvg_2000_res_0.8/lineage_T_cluster/processed_reintegrated_manual_annotated_clean.h5ad"

# Path to original AnnData (with raw counts)
ORIGINAL_PATH = "/dcl01/hongkai/data/data/hjiang/Data/long_covid/long_covid.h5ad"

# Output path for new AnnData with recovered raw counts
OUTPUT_PATH = "/dcs07/hongkai/data/harry/result/long_covid/analysis/preprocess/T_count_for_pseudobulk.h5ad"

# =============================================================================
# OPTIONAL CONFIGURATION
# =============================================================================

# Layer name in original AnnData containing raw counts
# Set to None to use .X from original
RAW_LAYER = None

# If True, store raw counts in .raw attribute
STORE_AS_RAW = True

# If True, also store the original normalized data as a layer before overwriting
SAVE_NORMALIZED_AS_LAYER = True

# Name of the layer to store normalized data (if SAVE_NORMALIZED_AS_LAYER is True)
NORMALIZED_LAYER_NAME = "normalized"

# If True, .raw will contain all genes from original
# If False, .raw will only contain genes that survived filtering
KEEP_ALL_GENES_IN_RAW = True

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_anndata(filepath: str) -> ad.AnnData:
    """Load an AnnData object from file."""
    print(f"Loading: {filepath}")
    adata = ad.read_h5ad(filepath)
    print(f"  Shape: {adata.shape}")
    print(f"  Obs columns: {list(adata.obs.columns)[:10]}...")
    print(f"  Var columns: {list(adata.var.columns)[:10]}...")
    print(f"  Layers: {list(adata.layers.keys())}")
    print(f"  Has .raw: {adata.raw is not None}")
    return adata


def validate_compatibility(original: ad.AnnData, preprocessed: ad.AnnData) -> dict:
    """
    Validate that original and preprocessed AnnData are compatible.
    Returns info about how to match cells/genes.
    """
    info = {
        'cells_match': False,
        'genes_match': False,
        'cells_subset': False,
        'genes_subset': False,
        'common_cells': [],
        'common_genes': [],
    }
    
    # Check cell (obs) compatibility
    orig_cells = set(original.obs_names)
    prep_cells = set(preprocessed.obs_names)
    
    common_cells = prep_cells.intersection(orig_cells)
    info['common_cells'] = list(common_cells)
    
    if orig_cells == prep_cells:
        info['cells_match'] = True
        print("✓ Cells match exactly between original and preprocessed")
    elif prep_cells.issubset(orig_cells):
        info['cells_subset'] = True
        n_removed = len(orig_cells) - len(prep_cells)
        print(f"✓ Preprocessed cells are a subset of original ({n_removed} cells were filtered)")
    elif len(common_cells) > 0:
        missing = prep_cells - orig_cells
        print(f"⚠ Warning: {len(missing)} cells in preprocessed not found in original")
        print(f"  Found {len(common_cells)} common cells")
        print(f"  Examples of missing cells: {list(missing)[:5]}")
    else:
        print("✗ ERROR: No common cells found!")
        print(f"  Original cell examples: {list(orig_cells)[:5]}")
        print(f"  Preprocessed cell examples: {list(prep_cells)[:5]}")
    
    # Check gene (var) compatibility
    orig_genes = set(original.var_names)
    prep_genes = set(preprocessed.var_names)
    
    common_genes = prep_genes.intersection(orig_genes)
    info['common_genes'] = list(common_genes)
    
    if orig_genes == prep_genes:
        info['genes_match'] = True
        print("✓ Genes match exactly between original and preprocessed")
    elif prep_genes.issubset(orig_genes):
        info['genes_subset'] = True
        n_removed = len(orig_genes) - len(prep_genes)
        print(f"✓ Preprocessed genes are a subset of original ({n_removed} genes were filtered)")
    elif len(common_genes) > 0:
        missing = prep_genes - orig_genes
        print(f"⚠ Warning: {len(missing)} genes in preprocessed not found in original")
        print(f"  Found {len(common_genes)} common genes")
    else:
        print("✗ ERROR: No common genes found!")
    
    return info


def recover_raw_counts(
    original: ad.AnnData,
    preprocessed: ad.AnnData,
    raw_layer: str = None,
    store_as_raw: bool = True,
    save_normalized: bool = True,
    normalized_layer_name: str = "normalized",
    keep_all_genes: bool = True
) -> ad.AnnData:
    """
    Create new AnnData with raw counts in .X (overwriting normalized data).
    
    Parameters
    ----------
    original : AnnData
        Original AnnData with raw counts in .X or specified layer
    preprocessed : AnnData
        Preprocessed AnnData (missing raw counts)
    raw_layer : str, optional
        Layer name in original containing raw counts. If None, uses .X
    store_as_raw : bool
        If True, store raw counts in .raw attribute
    save_normalized : bool
        If True, save the original normalized .X as a layer before overwriting
    normalized_layer_name : str
        Name for the layer to store normalized data
    keep_all_genes : bool
        If True, .raw will contain all genes from original (not just filtered ones)
    
    Returns
    -------
    AnnData
        New AnnData with raw counts in .X
    """
    
    # Get cells that are in both
    prep_cells = set(preprocessed.obs_names)
    orig_cells = set(original.obs_names)
    common_cells = list(prep_cells.intersection(orig_cells))
    
    # Maintain order from preprocessed
    common_cells_ordered = [c for c in preprocessed.obs_names if c in orig_cells]
    
    if len(common_cells_ordered) != preprocessed.n_obs:
        print(f"\n⚠ Warning: Only {len(common_cells_ordered)}/{preprocessed.n_obs} cells found in original")
        print("  Creating new AnnData with only common cells")
        # Subset preprocessed to only common cells
        new_adata = preprocessed[common_cells_ordered, :].copy()
    else:
        new_adata = preprocessed.copy()
    
    # Get the raw count matrix source info
    if raw_layer is not None:
        if raw_layer not in original.layers:
            raise ValueError(f"Layer '{raw_layer}' not found in original. "
                           f"Available layers: {list(original.layers.keys())}")
        print(f"\nUsing layer '{raw_layer}' from original as raw counts")
    else:
        print("\nUsing .X from original as raw counts")
    
    # Subset original to matching cells (maintain preprocessed order)
    orig_subset = original[common_cells_ordered, :].copy()
    
    # Save normalized data as a layer before overwriting (optional)
    if save_normalized:
        print(f"\nSaving current .X (normalized data) to layer '{normalized_layer_name}'")
        new_adata.layers[normalized_layer_name] = new_adata.X.copy()
        print(f"✓ Saved normalized data to layer '{normalized_layer_name}'")
    
    # Overwrite .X with raw counts
    prep_genes = list(preprocessed.var_names)
    orig_genes = set(original.var_names)
    common_genes = [g for g in prep_genes if g in orig_genes]
    
    if len(common_genes) != len(prep_genes):
        print(f"\n⚠ Warning: Only {len(common_genes)}/{len(prep_genes)} genes found in original")
        print("  Missing genes will have zero counts")
        
    # Get raw counts for common genes in preprocessed order
    orig_for_X = original[common_cells_ordered, :][:, common_genes]
    
    if raw_layer is not None:
        raw_counts = orig_for_X.layers[raw_layer].copy()
    else:
        raw_counts = orig_for_X.X.copy()
    
    # If some genes are missing, create a matrix with zeros for missing genes
    if len(common_genes) != len(prep_genes):
        from scipy import sparse
        
        # Create full matrix
        if sparse.issparse(raw_counts):
            full_matrix = sparse.lil_matrix((len(common_cells_ordered), len(prep_genes)))
        else:
            full_matrix = np.zeros((len(common_cells_ordered), len(prep_genes)))
        
        # Fill in values for common genes
        gene_idx_map = {g: i for i, g in enumerate(prep_genes)}
        for i, g in enumerate(common_genes):
            if sparse.issparse(raw_counts):
                full_matrix[:, gene_idx_map[g]] = raw_counts[:, i].toarray().flatten()
            else:
                full_matrix[:, gene_idx_map[g]] = raw_counts[:, i]
        
        if sparse.issparse(raw_counts):
            full_matrix = full_matrix.tocsr()
        
        new_adata.X = full_matrix
    else:
        new_adata.X = raw_counts
    
    print(f"✓ Overwrote .X with raw counts")
    
    # Store as .raw attribute
    if store_as_raw:
        if keep_all_genes:
            # Keep all genes from original in .raw
            print(f"\nStoring raw counts with all {orig_subset.n_vars} original genes in .raw")
            
            if raw_layer is not None:
                raw_X = orig_subset.layers[raw_layer].copy()
            else:
                raw_X = orig_subset.X.copy()
            
            raw_adata = ad.AnnData(
                X=raw_X,
                obs=pd.DataFrame(index=orig_subset.obs_names),
                var=orig_subset.var.copy()
            )
            new_adata.raw = raw_adata
            
        else:
            # Only keep genes that are in preprocessed
            genes_in_original = [g for g in prep_genes if g in original.var_names]
            
            print(f"\nStoring raw counts with {len(genes_in_original)} filtered genes in .raw")
            
            orig_filtered = orig_subset[:, genes_in_original].copy()
            
            if raw_layer is not None:
                raw_X = orig_filtered.layers[raw_layer].copy()
            else:
                raw_X = orig_filtered.X.copy()
            
            raw_adata = ad.AnnData(
                X=raw_X,
                obs=pd.DataFrame(index=orig_filtered.obs_names),
                var=orig_filtered.var.copy()
            )
            new_adata.raw = raw_adata
        
        print(f"✓ Added raw counts to .raw attribute")
    
    return new_adata


def main():
    """Main function to run the raw count recovery."""
    
    print("=" * 70)
    print("RAW COUNT RECOVERY SCRIPT")
    print("=" * 70)
    
    # Validate paths
    print("\nValidating paths...")
    
    if not os.path.exists(PREPROCESSED_PATH):
        sys.exit(f"ERROR: Preprocessed file not found:\n  {PREPROCESSED_PATH}")
    print(f"✓ Preprocessed file exists")
    
    if not os.path.exists(ORIGINAL_PATH):
        sys.exit(f"ERROR: Original file not found:\n  {ORIGINAL_PATH}")
    print(f"✓ Original file exists")
    
    # Create output directory if needed
    output_dir = Path(OUTPUT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {output_dir}")
    
    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    print("\n[1/2] Loading preprocessed AnnData...")
    preprocessed = load_anndata(PREPROCESSED_PATH)
    
    print("\n[2/2] Loading original AnnData...")
    original = load_anndata(ORIGINAL_PATH)
    
    # Validate compatibility
    print("\n" + "=" * 70)
    print("VALIDATING COMPATIBILITY")
    print("=" * 70)
    
    info = validate_compatibility(original, preprocessed)
    
    if len(info['common_cells']) == 0:
        sys.exit("\nERROR: No common cells found. Cannot proceed.")
    
    # Process
    print("\n" + "=" * 70)
    print("RECOVERING RAW COUNTS")
    print("=" * 70)
    
    new_adata = recover_raw_counts(
        original=original,
        preprocessed=preprocessed,
        raw_layer=RAW_LAYER,
        store_as_raw=STORE_AS_RAW,
        save_normalized=SAVE_NORMALIZED_AS_LAYER,
        normalized_layer_name=NORMALIZED_LAYER_NAME,
        keep_all_genes=KEEP_ALL_GENES_IN_RAW
    )
    
    # Save
    print("\n" + "=" * 70)
    print("SAVING OUTPUT")
    print("=" * 70)
    
    print(f"\nSaving to: {OUTPUT_PATH}")
    new_adata.write_h5ad(OUTPUT_PATH)
    print("✓ File saved successfully")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nNew AnnData properties:")
    print(f"  Shape: {new_adata.shape}")
    print(f"  Obs columns: {list(new_adata.obs.columns)[:10]}...")
    print(f"  Var columns: {list(new_adata.var.columns)[:10]}...")
    print(f"  Layers: {list(new_adata.layers.keys())}")
    print(f"  Obsm keys: {list(new_adata.obsm.keys())}")
    print(f"  Has .raw: {new_adata.raw is not None}")
    
    if new_adata.raw is not None:
        print(f"  .raw shape: {new_adata.raw.shape}")
    
    # Show .X statistics (now contains raw counts)
    X_data = new_adata.X
    print(f"\n  .X (raw counts) statistics:")
    if hasattr(X_data, 'toarray'):
        # Sparse matrix
        print(f"    Type: sparse matrix")
        print(f"    Dtype: {X_data.dtype}")
        print(f"    Non-zero elements: {X_data.nnz}")
        sample = X_data[:5, :5].toarray()
    else:
        print(f"    Type: dense array")
        print(f"    Dtype: {X_data.dtype}")
        sample = X_data[:5, :5]
    print(f"    Sample (5x5):\n{sample}")
    
    if NORMALIZED_LAYER_NAME in new_adata.layers:
        print(f"\n  Layer '{NORMALIZED_LAYER_NAME}' (normalized data) preserved")
    
    print(f"\n✓ Output saved to: {OUTPUT_PATH}")
    print("\nDone!")


if __name__ == "__main__":
    main()