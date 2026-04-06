import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
import rapids_singlecell as rsc
from harmony import harmonize
from scipy import sparse
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.safe_save import safe_h5ad_write, ensure_cpu_arrays
from utils.random_seed import set_global_seed
from utils.merge_sample_meta import merge_sample_metadata


# =============================================================================
# USER CONFIGURATION - MODIFY THESE VARIABLES
# =============================================================================

H5AD_PATH = "/dcl01/hongkai/data/data/hjiang/Data/long_covid/long_covid.h5ad"
OUTPUT_DIR = "/dcs07/hongkai/data/harry/result/long_covid/analysis"
SAMPLE_META_PATH = "/dcl01/hongkai/data/data/hjiang/Data/long_covid/sample_meta.csv"
CELL_META_PATH = None

RUN_STEPS = "all"
VERBOSE = True
SAVE = True

SAMPLE_COLUMN = "sample"
BATCH_KEY = None

MIN_CELLS = 500
MIN_FEATURES = 500
PCT_MITO_CUTOFF = 20
EXCLUDE_GENES = None
VARS_TO_REGRESS = None

NUM_FEATURES = 2000
NUM_PCS = 30
NUM_HARMONY = 30

# Can be a single value or a list of values
CLUSTER_RESOLUTION = [0.25, 0.5, 0.75, 1.0]
USE_REP = "X_pca_harmony"

RUN_CELLTYPIST = True
CELLTYPIST_MODEL = "/users/hjiang/GenoDistance/long_covid/PaediatricAdult_COVID19_PBMC.pkl"

COMPUTE_UMAP = True
GENERATE_PLOTS = True
FIND_MARKERS = True

# Marker gene visualization parameters
N_TOP_MARKERS = 10  # Number of top markers per cluster for dot plot
N_MARKER_UMAP = 10  # Number of top markers per cluster for UMAP feature plots

# Common marker gene file path (optional)
COMMON_MARKER_GENE_PATH = "/users/hjiang/GenoDistance/long_covid/cell_type_marker_gene.csv"

# Parallelization settings
N_JOBS_PLOTTING = 8  # Number of parallel workers for UMAP plotting


# =============================================================================
# HELPER FUNCTION: Identify BCR and TCR genes
# =============================================================================
def _get_ig_tcr_mask(var_names):
    """
    Get boolean mask for IG (BCR) and TCR genes.

    BCR genes: start with "IG" (case-sensitive, as in original)
    TCR genes: start with "TRA" or "TRB" (case-insensitive)

    Returns:
        ig_mask: boolean array for IG genes
        tcr_mask: boolean array for TCR genes
        combined_mask: boolean array for IG or TCR genes
    """
    # IG genes (BCR) - case sensitive as in original
    ig_mask = var_names.str.startswith("IG")

    # TCR genes - case insensitive
    var_names_upper = var_names.str.upper()
    tcr_mask = var_names_upper.str.startswith("TRA") | var_names_upper.str.startswith("TRB")

    # Combined mask
    combined_mask = ig_mask | tcr_mask

    return ig_mask, tcr_mask, combined_mask


# =============================================================================
# STEP 1: QC and Filtering
# =============================================================================
def qc_and_filter(
    adata,
    output_dir=None,
    sample_column="sample",
    sample_meta_path=None,
    cell_meta_path=None,
    batch_key="batch",
    min_cells=500,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    vars_to_regress=None,
    verbose=True,
):
    if verbose:
        print("=== Step 1: QC and Filtering ===")
        print(f"Input shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

    if cell_meta_path is None:
        if sample_column not in adata.obs.columns:
            if verbose:
                print(f"   ℹ️ No '{sample_column}' column; inferring from obs_names")
            adata.obs[sample_column] = adata.obs_names.str.split(":").str[0]
    else:
        if verbose:
            print(f"   📄 Merging cell-level metadata from: {cell_meta_path}")
        cell_meta = pd.read_csv(cell_meta_path).set_index("barcode")
        adata.obs = adata.obs.join(cell_meta, how="left")
        if sample_column not in adata.obs.columns:
            adata.obs[sample_column] = adata.obs_names.str.split(":").str[0]

    if sample_meta_path is not None:
        if verbose:
            print("   📄 Merging sample-level metadata...")
        adata = merge_sample_metadata(
            adata=adata,
            metadata_path=sample_meta_path,
            sample_column=sample_column,
            verbose=verbose,
        )

    vars_to_regress = vars_to_regress or []
    flat_vars = []
    for v in vars_to_regress:
        if isinstance(v, (list, tuple, np.ndarray, pd.Index)):
            flat_vars.extend(map(str, list(v)))
        else:
            flat_vars.append(str(v))

    vars_to_regress_for_harmony = flat_vars.copy()
    if sample_column not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append(sample_column)

    flat_batch_keys = []
    if batch_key:
        if isinstance(batch_key, (list, tuple, np.ndarray, pd.Index)):
            flat_batch_keys.extend(map(str, list(batch_key)))
        else:
            flat_batch_keys.append(str(batch_key))

    required = list(dict.fromkeys(flat_vars + flat_batch_keys))
    missing_vars = sorted(set(required) - set(map(str, adata.obs.columns)))
    if missing_vars:
        raise KeyError(f"Missing variables in adata.obs: {missing_vars}")

    if adata.X.dtype != np.float32:
        if issparse(adata.X):
            adata.X = adata.X.astype(np.float32)
        else:
            adata.X = np.asarray(adata.X, dtype=np.float32)

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_features)

    if verbose:
        print(f"   After gene/cell filter: {adata.shape[0]} cells × {adata.shape[1]} genes")

    mt_mask = adata.var_names.str.startswith(("MT-", "mt-"))
    adata.var["mt"] = mt_mask
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], log1p=False, inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < pct_mito_cutoff].copy()

    if verbose:
        print(f"   After mito filter (<{pct_mito_cutoff}%): {adata.shape[0]} cells")

    mt_genes = adata.var_names[adata.var_names.str.startswith("MT-")]
    genes_to_exclude = set(mt_genes) | set(exclude_genes or [])
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()

    n_before = adata.shape[1]
    ensg_mask = adata.var_names.str.startswith("ENSG")
    ensg_genes = adata.var_names[ensg_mask].tolist()
    n_ensg = len(ensg_genes)
    adata = adata[:, ~ensg_mask].copy()

    if verbose:
        print(f"   Removed {n_ensg} pseudogenes (ENSG*)")
        print(f"   Final shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

    if output_dir:
        preprocess_dir = os.path.join(output_dir, "preprocess")
        os.makedirs(preprocess_dir, exist_ok=True)

        summary_path = os.path.join(preprocess_dir, "gene_removal_summary.txt")
        with open(summary_path, "w") as f:
            f.write("=== Gene Removal Summary (Step 1: QC) ===\n\n")
            f.write(f"Total genes before removal: {n_before}\n")
            f.write(f"Pseudogenes removed (ENSG*): {n_ensg}\n")
            f.write(f"Total genes after removal: {adata.shape[1]}\n\n")
            if ensg_genes:
                f.write("Removed ENSG genes:\n")
                for g in ensg_genes[:100]:
                    f.write(f"  {g}\n")
                if len(ensg_genes) > 100:
                    f.write(f"  ... and {len(ensg_genes) - 100} more\n")

        _plot_gene_removal_summary(
            n_total=n_before,
            n_ensg=n_ensg,
            n_ig=0,
            n_tcr=0,
            output_dir=preprocess_dir,
            stage="qc",
            verbose=verbose,
        )

    return adata, vars_to_regress_for_harmony


# =============================================================================
# STEP 2: Normalization and Integration
# =============================================================================
def normalize_and_integrate(
    adata,
    output_dir=None,
    sample_column="sample",
    num_features=2000,
    num_PCs=20,
    num_harmony=30,
    vars_to_regress_for_harmony=None,
    save=True,
    verbose=True,
):
    if verbose:
        print("\n=== Step 2: Normalization and Integration ===")
        print("   Creating adata_cluster and adata_sample copies...")

    adata_cluster = adata.copy()
    adata_sample = adata.copy()

    # Get IG (BCR) and TCR gene masks
    ig_mask, tcr_mask, ig_tcr_mask = _get_ig_tcr_mask(adata_cluster.var_names)

    ig_genes = adata_cluster.var_names[ig_mask].tolist()
    tcr_genes = adata_cluster.var_names[tcr_mask].tolist()
    n_ig = len(ig_genes)
    n_tcr = len(tcr_genes)

    if verbose:
        if n_ig > 0:
            print(f"   Found {n_ig} IG (BCR) genes (will exclude from HVG selection)")
        if n_tcr > 0:
            print(f"   Found {n_tcr} TCR genes (TRA*/TRB*) (will exclude from HVG selection)")
        print(f"   Total immune receptor genes: {n_ig + n_tcr}")

    if output_dir:
        preprocess_dir = os.path.join(output_dir, "preprocess")
        os.makedirs(preprocess_dir, exist_ok=True)

        summary_path = os.path.join(preprocess_dir, "gene_removal_summary.txt")
        with open(summary_path, "a") as f:
            f.write("\n=== Gene Annotation Summary (Step 2: Integration) ===\n\n")
            f.write(f"IG (BCR) genes (excluded from HVG/PCA): {n_ig}\n")
            f.write(f"TCR genes (excluded from HVG/PCA): {n_tcr}\n")
            f.write(f"Total immune receptor genes: {n_ig + n_tcr}\n")
            f.write("Note: ALL genes are kept in adata_cluster (with annotations)\n")
            f.write("Note: HVG marked in adata_cluster.var['highly_variable']\n")
            f.write("Note: IG/TCR marked in adata_cluster.var['is_ig'] and adata_cluster.var['is_tcr']\n\n")

            if ig_genes:
                f.write("IG (BCR) genes:\n")
                for g in ig_genes:
                    f.write(f"  {g}\n")
                f.write("\n")

            if tcr_genes:
                f.write("TCR genes:\n")
                for g in tcr_genes:
                    f.write(f"  {g}\n")

        _plot_gene_removal_summary(
            n_total=adata.shape[1],
            n_ensg=0,
            n_ig=n_ig,
            n_tcr=n_tcr,
            output_dir=preprocess_dir,
            stage="integration",
            verbose=verbose,
        )

    if verbose:
        print("\n--- Processing adata_cluster ---")

    rsc.get.anndata_to_CPU(adata_cluster)

    if verbose:
        print("   Running HVG selection (excluding IG and TCR genes)...")

    # Exclude both IG and TCR genes from HVG selection
    adata_for_hvg = adata_cluster[:, ~ig_tcr_mask].copy()
    sc.pp.highly_variable_genes(
        adata_for_hvg,
        n_top_genes=num_features,
        flavor="seurat_v3",
        batch_key=None,
    )
    hvg_genes = adata_for_hvg.var_names[adata_for_hvg.var["highly_variable"]].tolist()
    del adata_for_hvg

    if verbose:
        print(f"   Selected {len(hvg_genes)} HVG genes")

    # Create subset for PCA (HVG only, no IG/TCR)
    adata_for_pca = adata_cluster[:, adata_cluster.var_names.isin(hvg_genes)].copy()

    if verbose:
        print(f"   adata for PCA: {adata_for_pca.shape[1]} genes")

    rsc.get.anndata_to_GPU(adata_for_pca)
    rsc.pp.normalize_total(adata_for_pca, target_sum=1e4)
    rsc.pp.log1p(adata_for_pca)

    if verbose:
        print("   Running PCA...")
    rsc.pp.pca(adata_for_pca, n_comps=num_PCs)

    if verbose:
        print("   Running Harmony integration...")
        print(f"   Variables to regress: {', '.join(vars_to_regress_for_harmony or [])}")

    Z = harmonize(
        adata_for_pca.obsm["X_pca"],
        adata_for_pca.obs,
        batch_key=vars_to_regress_for_harmony,
        max_iter_harmony=num_harmony,
        use_gpu=True,
    )
    adata_for_pca.obsm["X_pca_harmony"] = Z

    rsc.get.anndata_to_CPU(adata_for_pca)

    # =========================================================================
    # KEY CHANGE: Keep ALL genes in adata_cluster, mark HVG/IG/TCR in .var
    # =========================================================================
    
    # Initialize var annotations
    adata_cluster.var["highly_variable"] = adata_cluster.var_names.isin(hvg_genes)
    adata_cluster.var["is_ig"] = adata_cluster.var_names.str.startswith("IG")
    # TCR annotation (case-insensitive)
    var_names_upper = adata_cluster.var_names.str.upper()
    adata_cluster.var["is_tcr"] = var_names_upper.str.startswith("TRA") | var_names_upper.str.startswith("TRB")
    # Combined immune receptor flag
    adata_cluster.var["is_immune_receptor"] = adata_cluster.var["is_ig"] | adata_cluster.var["is_tcr"]
    # Flag for genes used in PCA (HVG and not immune receptor)
    adata_cluster.var["used_in_pca"] = adata_cluster.var["highly_variable"] & ~adata_cluster.var["is_immune_receptor"]

    # Normalize ALL genes in adata_cluster
    rsc.get.anndata_to_GPU(adata_cluster)
    rsc.pp.normalize_total(adata_cluster, target_sum=1e4)
    rsc.pp.log1p(adata_cluster)
    rsc.get.anndata_to_CPU(adata_cluster)

    # Transfer PCA embeddings from adata_for_pca to adata_cluster
    adata_cluster.obsm["X_pca"] = adata_for_pca.obsm["X_pca"]
    adata_cluster.obsm["X_pca_harmony"] = adata_for_pca.obsm["X_pca_harmony"]
    if "pca" in adata_for_pca.uns:
        adata_cluster.uns["pca"] = adata_for_pca.uns["pca"]
    adata_cluster.uns["pca_genes"] = hvg_genes

    del adata_for_pca

    n_hvg = adata_cluster.var["highly_variable"].sum()
    n_used_pca = adata_cluster.var["used_in_pca"].sum()
    
    if verbose:
        print(f"\n   adata_cluster final: {adata_cluster.shape[1]} genes (ALL genes retained)")
        print(f"      - Highly variable genes (HVG): {n_hvg}")
        print(f"      - IG (BCR) genes: {n_ig}")
        print(f"      - TCR genes: {n_tcr}")
        print(f"      - Genes used in PCA: {n_used_pca}")
        print("   ✓ adata_cluster processing complete")

    if verbose:
        print("\n--- Processing adata_sample ---")

    rsc.get.anndata_to_CPU(adata_sample)

    X_raw = adata_sample.X.copy()
    sc.pp.normalize_total(adata_sample, target_sum=1e4)
    sc.pp.log1p(adata_sample)
    sc.pp.pca(adata_sample, n_comps=num_PCs)
    adata_sample.X = X_raw
    del X_raw

    if verbose:
        print(f"   adata_sample shape: {adata_sample.shape[0]} cells × {adata_sample.shape[1]} genes")
        print("   ✓ adata_sample processing complete (raw counts preserved)")

    if save and output_dir:
        preprocess_dir = os.path.join(output_dir, "preprocess")
        os.makedirs(preprocess_dir, exist_ok=True)
        safe_h5ad_write(adata_cluster, os.path.join(preprocess_dir, "adata_cell.h5ad"), verbose=verbose)
        safe_h5ad_write(adata_sample, os.path.join(preprocess_dir, "adata_sample.h5ad"), verbose=verbose)

    return adata_cluster, adata_sample


# =============================================================================
# STEP 3: Clustering and Annotation
# =============================================================================
def _resolution_to_col_name(resolution):
    """Convert resolution to column name, e.g., 0.18 -> 'leiden_0.18'"""
    res_str = f"{resolution:.2f}".rstrip("0").rstrip(".")
    return f"leiden_{res_str}"


def _resolution_to_folder_name(resolution):
    """Convert resolution to folder name, e.g., 0.18 -> 'resolution_0.18'"""
    res_str = f"{resolution:.2f}".rstrip("0").rstrip(".")
    return f"resolution_{res_str}"


def cluster_and_annotate(
    adata_cluster,
    adata_sample=None,
    output_dir=None,
    sample_column="sample",
    cluster_resolution=0.8,
    use_rep="X_pca_harmony",
    num_PCs=20,
    run_celltypist=True,
    celltypist_model=None,
    compute_umap=True,
    find_markers=True,
    save=True,
    verbose=True,
    generate_plots=True,
    n_top_markers=5,
    n_marker_umap=3,
    common_marker_gene_path=None,
    n_jobs_plotting=8,
):
    if verbose:
        print("\n=== Step 3: Clustering and Cell Type Annotation ===")

    # Handle single or multiple resolutions
    if isinstance(cluster_resolution, (list, tuple)):
        resolutions = list(cluster_resolution)
    else:
        resolutions = [cluster_resolution]

    if verbose:
        print(f"   Resolutions to test: {resolutions}")

    # Build neighbors graph (shared across all resolutions)
    if verbose:
        print("\n--- Computing shared resources ---")

    rsc.get.anndata_to_GPU(adata_cluster)

    if verbose:
        print("   Building neighborhood graph...")
    rsc.pp.neighbors(adata_cluster, use_rep=use_rep, n_pcs=num_PCs, random_state=42)

    # UMAP (shared across all resolutions)
    if compute_umap:
        if verbose:
            print("   Computing UMAP...")
        rsc.tl.umap(adata_cluster, min_dist=0.5)

    # Leiden clustering at all resolutions
    if verbose:
        print("\n--- Clustering at multiple resolutions ---")

    resolution_results = {}

    for res in resolutions:
        col_name = _resolution_to_col_name(res)

        if verbose:
            print(f"   Clustering at resolution {res}...")

        rsc.tl.leiden(
            adata_cluster,
            resolution=res,
            key_added=col_name,
            random_state=42,
        )

        # Convert to 1-indexed string categories
        rsc.get.anndata_to_CPU(adata_cluster)
        adata_cluster.obs[col_name] = ((adata_cluster.obs[col_name].astype(int) + 1).astype(str)).astype("category")
        rsc.get.anndata_to_GPU(adata_cluster)

        n_clusters = adata_cluster.obs[col_name].nunique()
        resolution_results[res] = {"col_name": col_name, "n_clusters": n_clusters}

        if verbose:
            print(f"      {col_name}: {n_clusters} clusters")

    rsc.get.anndata_to_CPU(adata_cluster)
    adata_cluster = ensure_cpu_arrays(adata_cluster)

    # Set default 'leiden' column to first resolution
    default_res = resolutions[0]
    default_col = _resolution_to_col_name(default_res)
    adata_cluster.obs["leiden"] = adata_cluster.obs[default_col].copy()

    if verbose:
        print(f"\n   Default 'leiden' column set to resolution {default_res}")

    # Cell type annotation with celltypist (done once)
    if run_celltypist:
        if verbose:
            print("\n--- Running celltypist annotation ---")
        adata_cluster = _run_celltypist(adata_cluster, model_name=celltypist_model, verbose=verbose)
    else:
        adata_cluster.obs["cell_type"] = adata_cluster.obs["leiden"].copy()

    # Process each resolution: majority voting, markers, visualizations
    if output_dir:
        clustering_dir = os.path.join(output_dir, "clustering")
        os.makedirs(clustering_dir, exist_ok=True)

        all_majority_votes = []

        for res in resolutions:
            col_name = _resolution_to_col_name(res)
            folder_name = _resolution_to_folder_name(res)
            res_dir = os.path.join(clustering_dir, folder_name)
            os.makedirs(res_dir, exist_ok=True)

            if verbose:
                print(f"\n--- Processing resolution {res} ---")

            # Majority voting
            if run_celltypist:
                if verbose:
                    print(f"   Computing majority voting for {col_name}...")

                mv_df = _majority_vote_cell_types(
                    adata_cluster.obs,
                    cluster_col=col_name,
                    celltype_col="cell_type",
                    verbose=verbose,
                )
                mv_df["resolution"] = res
                mv_df["leiden_col"] = col_name
                all_majority_votes.append(mv_df)

                # Save resolution-specific mapping
                mv_path = os.path.join(res_dir, "leiden_celltype_mapping.csv")
                mv_df.to_csv(mv_path, index=False)

            # Find marker genes for this resolution
            if find_markers:
                if verbose:
                    print(f"   Finding marker genes for {col_name}...")

                _find_markers_and_visualize(
                    adata_cluster,
                    groupby=col_name,
                    output_dir=res_dir,
                    n_top_markers=n_top_markers,
                    n_marker_umap=n_marker_umap,
                    verbose=verbose,
                    n_jobs=n_jobs_plotting,
                )

            # Generate UMAP visualization for this resolution
            if generate_plots and compute_umap:
                if verbose:
                    print(f"   Generating UMAP for {col_name}...")
                _plot_single_umap(adata_cluster, col_name, res_dir, verbose=False)

        # Save combined majority voting results
        if run_celltypist and all_majority_votes:
            combined_mv_df = pd.concat(all_majority_votes, ignore_index=True)
            combined_mv_path = os.path.join(clustering_dir, "leiden_celltype_mapping_all.csv")
            combined_mv_df.to_csv(combined_mv_path, index=False)
            if verbose:
                print(f"\n   Saved combined leiden-celltype mappings to {combined_mv_path}")

        # Save resolution comparison
        if len(resolutions) > 1:
            _save_resolution_comparison(adata_cluster, resolutions, resolution_results, clustering_dir, verbose)

        # Generate shared UMAPs (cell type and sample)
        if generate_plots and compute_umap:
            if verbose:
                print("\n--- Generating shared visualizations ---")
            _plot_single_umap(adata_cluster, "cell_type", clustering_dir, verbose)
            _plot_sample_umap(adata_cluster, sample_column, clustering_dir, verbose)

        # Generate common marker gene UMAPs (resolution-independent)
        if common_marker_gene_path and compute_umap:
            if verbose:
                print("\n--- Generating common marker gene UMAPs ---")
            _plot_common_marker_genes(
                adata_cluster,
                marker_gene_path=common_marker_gene_path,
                output_dir=clustering_dir,
                verbose=verbose,
                n_jobs=n_jobs_plotting,
            )

    # Transfer to adata_sample
    if adata_sample is not None:
        if verbose:
            print("\n--- Transferring annotations to adata_sample ---")

        # Transfer all leiden columns + cell_type
        cols_to_transfer = ["cell_type"] + [_resolution_to_col_name(r) for r in resolutions] + ["leiden"]

        for col in cols_to_transfer:
            if col in adata_cluster.obs.columns:
                if adata_sample.n_obs == adata_cluster.n_obs and adata_sample.obs_names.equals(adata_cluster.obs_names):
                    adata_sample.obs[col] = adata_cluster.obs[col].values
                else:
                    common = adata_sample.obs_names.intersection(adata_cluster.obs_names)
                    if len(common) == 0:
                        raise ValueError(f"No common obs_names; cannot transfer {col}")
                    adata_sample.obs[col] = pd.Series(pd.NA, index=adata_sample.obs_names, dtype="object")
                    adata_sample.obs.loc[common, col] = adata_cluster.obs.loc[common, col].values

        # Transfer embeddings
        for key in ["X_umap", "X_pca_harmony"]:
            if key in adata_cluster.obsm:
                if adata_sample.n_obs == adata_cluster.n_obs and adata_sample.obs_names.equals(adata_cluster.obs_names):
                    adata_sample.obsm[key] = adata_cluster.obsm[key].copy()
                else:
                    common = adata_sample.obs_names.intersection(adata_cluster.obs_names)
                    embedding = np.full((adata_sample.n_obs, adata_cluster.obsm[key].shape[1]), np.nan)
                    idx_sample = [adata_sample.obs_names.get_loc(c) for c in common]
                    idx_cluster = [adata_cluster.obs_names.get_loc(c) for c in common]
                    embedding[idx_sample] = adata_cluster.obsm[key][idx_cluster]
                    adata_sample.obsm[key] = embedding

        adata_sample = ensure_cpu_arrays(adata_sample)

    # Save outputs
    if save and output_dir:
        preprocess_dir = os.path.join(output_dir, "preprocess")
        os.makedirs(preprocess_dir, exist_ok=True)

        # Save cell type CSV (with all leiden columns)
        celltype_data = {"cell_id": adata_cluster.obs.index}
        for res in resolutions:
            col_name = _resolution_to_col_name(res)
            celltype_data[col_name] = adata_cluster.obs[col_name].astype(str)
        celltype_data["cell_type"] = adata_cluster.obs["cell_type"].astype(str)

        celltype_df = pd.DataFrame(celltype_data)
        celltype_df.to_csv(os.path.join(preprocess_dir, "cell_type.csv"), index=False)

        # Save gene annotation summary
        gene_summary_path = os.path.join(preprocess_dir, "gene_annotations.csv")
        gene_summary = adata_cluster.var[["highly_variable", "is_ig", "is_tcr", "is_immune_receptor", "used_in_pca"]].copy()
        gene_summary.to_csv(gene_summary_path)
        if verbose:
            print(f"   Saved gene annotations to {gene_summary_path}")

        # Save single adata files
        safe_h5ad_write(adata_cluster, os.path.join(preprocess_dir, "adata_cell.h5ad"), verbose=verbose)
        if adata_sample is not None:
            safe_h5ad_write(adata_sample, os.path.join(preprocess_dir, "adata_sample.h5ad"), verbose=verbose)

    if verbose:
        print(f"\n   ✓ Clustering complete")
        print(f"   Leiden columns: {[_resolution_to_col_name(r) for r in resolutions]}")
        print(f"   Default 'leiden' = {default_col}")
        print(f"   Total genes in adata_cluster: {adata_cluster.shape[1]}")
        print(f"   HVG marked: {adata_cluster.var['highly_variable'].sum()}")

    if adata_sample is None:
        return adata_cluster
    return adata_cluster, adata_sample


# =============================================================================
# Parallel UMAP Plotting Helper Functions
# =============================================================================
def _plot_single_gene_umap_data(args):
    """
    Worker function for parallel UMAP plotting.
    Takes pre-extracted data to avoid passing large adata objects.

    IMPORTANT IMPROVEMENT:
      - Seurat FeaturePlot(order=TRUE) equivalent:
        plot low expression first and high expression last (on top)
    """
    gene, expr, umap_coords, output_path = args

    try:
        expr = np.asarray(expr)
        order = np.argsort(expr, kind="mergesort")  # stable sort
        x = umap_coords[order, 0]
        y = umap_coords[order, 1]
        c = expr[order]

        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(
            x,
            y,
            c=c,
            cmap="viridis",
            s=1,
            alpha=0.7,
            rasterized=True,
            linewidths=0,
        )

        plt.colorbar(scatter, ax=ax, label="Expression", shrink=0.8)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"{gene}", fontsize=14)
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return gene, True, None
    except Exception as e:
        plt.close("all")
        return gene, False, str(e)


def _extract_gene_expression(adata, gene):
    """Extract expression values for a single gene (faster + lower memory)."""
    if gene not in adata.var_names:
        return None

    gene_idx = adata.var_names.get_loc(gene)
    X = adata.X

    if issparse(X):
        # Use toarray() and flatten() for scipy >= 1.11 compatibility
        return np.asarray(X[:, gene_idx].toarray()).flatten().astype(np.float32, copy=False)
    return np.asarray(X[:, gene_idx], dtype=np.float32).reshape(-1)


def _plot_gene_umaps_parallel(adata, genes, output_dir, n_jobs=8, verbose=True):
    """
    Plot multiple gene UMAPs in parallel.

    Returns:
        dict mapping gene -> output_path for successfully plotted genes
    """
    if "X_umap" not in adata.obsm:
        if verbose:
            print("      No UMAP coordinates found, skipping feature plots")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    umap_coords = np.asarray(adata.obsm["X_umap"])
    plot_args = []

    for gene in genes:
        expr = _extract_gene_expression(adata, gene)
        if expr is None:
            continue
        output_path = os.path.join(output_dir, f"umap_{gene}.png")
        plot_args.append((gene, expr, umap_coords, output_path))

    if not plot_args:
        if verbose:
            print("      No valid genes found for UMAP plotting")
        return {}

    if verbose:
        print(f"      Plotting {len(plot_args)} gene UMAPs with {n_jobs} parallel workers...")

    results = {}
    n_success = 0
    n_failed = 0

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for gene, success, error in executor.map(_plot_single_gene_umap_data, plot_args):
            if success:
                results[gene] = os.path.join(output_dir, f"umap_{gene}.png")
                n_success += 1
            else:
                n_failed += 1
                if verbose:
                    print(f"      Warning: Failed to plot {gene}: {error}")

    if verbose:
        print(f"      Successfully plotted {n_success} genes" + (f", {n_failed} failed" if n_failed > 0 else ""))

    return results


# =============================================================================
# Marker Gene Analysis and Visualization
# =============================================================================
def _find_markers_and_visualize(
    adata,
    groupby,
    output_dir,
    n_top_markers=5,
    n_marker_umap=3,
    n_genes_csv=20,
    verbose=True,
    n_jobs=8,
):
    """Find marker genes, save CSV, generate dot plot, and UMAP feature plots organized by cluster."""
    markers_dir = os.path.join(output_dir, "markers")
    os.makedirs(markers_dir, exist_ok=True)

    # Exclude IG and TCR genes from marker finding
    # Use the var annotations if available, otherwise compute mask
    if "is_immune_receptor" in adata.var.columns:
        ig_tcr_mask = adata.var["is_immune_receptor"].values
    else:
        _, _, ig_tcr_mask = _get_ig_tcr_mask(adata.var_names)
    
    adata_for_markers = adata[:, ~ig_tcr_mask].copy()

    if verbose:
        print("      Finding marker genes (excluding IG and TCR genes)...")

    # Run differential expression
    sc.tl.rank_genes_groups(
        adata_for_markers,
        groupby=groupby,
        method="wilcoxon",
        use_raw=False,
    )

    # Extract marker genes for each cluster
    markers_list = []
    clusters = adata_for_markers.obs[groupby].cat.categories

    for cluster in clusters:
        df = sc.get.rank_genes_groups_df(adata_for_markers, group=cluster)
        df = df.head(n_genes_csv)
        df["cluster"] = cluster
        markers_list.append(df)

    markers_df = pd.concat(markers_list, ignore_index=True)

    # Save full marker genes CSV
    markers_path = os.path.join(markers_dir, "marker_genes.csv")
    markers_df.to_csv(markers_path, index=False)
    if verbose:
        print(f"      Saved marker genes to {markers_path}")

    # Get top markers per cluster for visualization
    top_markers_per_cluster = {}
    all_top_markers = []

    for cluster in clusters:
        cluster_markers = markers_df[markers_df["cluster"] == cluster]
        # Filter for significant markers (positive scores, low p-value)
        significant = cluster_markers[(cluster_markers["scores"] > 0) & (cluster_markers["pvals_adj"] < 0.05)]
        top_genes = significant["names"].head(max(n_top_markers, n_marker_umap)).tolist()
        top_markers_per_cluster[cluster] = top_genes
        all_top_markers.extend(top_genes[:n_top_markers])

    # Remove duplicates while preserving order
    unique_markers = list(dict.fromkeys(all_top_markers))

    if verbose:
        print(f"      Selected {len(unique_markers)} unique top markers for dot plot")

    # Generate dot plot (use original adata which has all genes)
    if len(unique_markers) > 0:
        _plot_marker_dotplot(
            adata,  # Use full adata for visualization (has all genes)
            markers=unique_markers,
            groupby=groupby,
            output_dir=markers_dir,
            verbose=verbose,
        )

    # Generate UMAP feature plots organized by cluster (use full adata)
    if "X_umap" in adata.obsm:
        _plot_marker_umap_by_cluster(
            adata,  # Use full adata for visualization
            top_markers_per_cluster=top_markers_per_cluster,
            n_marker_umap=n_marker_umap,
            output_dir=markers_dir,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    # Save top markers summary
    top_markers_summary = []
    for cluster, genes in top_markers_per_cluster.items():
        top_markers_summary.append({"cluster": cluster, "top_markers": ", ".join(genes[:n_top_markers])})

    summary_df = pd.DataFrame(top_markers_summary)
    summary_path = os.path.join(markers_dir, "top_markers_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    if verbose:
        print(f"      Saved top markers summary to {summary_path}")


def _plot_marker_umap_by_cluster(
    adata,
    top_markers_per_cluster,
    n_marker_umap,
    output_dir,
    verbose=True,
    n_jobs=8,
):
    """
    Generate UMAP feature plots organized by cluster folder.
    Each cluster gets its own folder with its top marker gene UMAPs.
    """
    if "X_umap" not in adata.obsm:
        if verbose:
            print("      No UMAP coordinates found, skipping feature plots")
        return

    umap_base_dir = os.path.join(output_dir, "umap_features")
    os.makedirs(umap_base_dir, exist_ok=True)

    # Collect all genes to plot and organize by cluster
    all_genes_to_plot = set()
    gene_to_clusters = {}

    for cluster, genes in top_markers_per_cluster.items():
        valid_genes = [g for g in genes[:n_marker_umap] if g in adata.var_names]
        for gene in valid_genes:
            all_genes_to_plot.add(gene)
            gene_to_clusters.setdefault(gene, []).append(cluster)

    if len(all_genes_to_plot) == 0:
        if verbose:
            print("      No valid genes found for UMAP feature plots")
        return

    if verbose:
        print(f"      Generating UMAP feature plots for {len(all_genes_to_plot)} unique marker genes...")

    # Create cluster directories
    for cluster in top_markers_per_cluster.keys():
        cluster_dir = os.path.join(umap_base_dir, f"cluster_{cluster}")
        os.makedirs(cluster_dir, exist_ok=True)

    # Plot all unique genes in parallel to a temp directory
    temp_plot_dir = os.path.join(umap_base_dir, "_temp_plots")
    os.makedirs(temp_plot_dir, exist_ok=True)

    plotted_genes = _plot_gene_umaps_parallel(
        adata,
        list(all_genes_to_plot),
        temp_plot_dir,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # Distribute plots to cluster folders:
    # Prefer hardlinks (fast, no extra disk). Fallback to copy2.
    import shutil

    for gene, source_path in plotted_genes.items():
        if not os.path.exists(source_path):
            continue
        for cluster in gene_to_clusters.get(gene, []):
            cluster_dir = os.path.join(umap_base_dir, f"cluster_{cluster}")
            dest_path = os.path.join(cluster_dir, f"umap_{gene}.png")
            try:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                os.link(source_path, dest_path)
            except Exception:
                shutil.copy2(source_path, dest_path)

    # Clean up temp directory
    shutil.rmtree(temp_plot_dir, ignore_errors=True)

    if verbose:
        print(f"      Organized UMAP plots into {len(top_markers_per_cluster)} cluster folders")


def _plot_marker_dotplot(adata, markers, groupby, output_dir, verbose=True):
    """Generate dot plot of marker genes across clusters."""
    if len(markers) == 0:
        if verbose:
            print("      No markers to plot in dot plot")
        return

    valid_markers = [g for g in markers if g in adata.var_names]

    if len(valid_markers) == 0:
        if verbose:
            print("      No valid markers found in adata for dot plot")
        return

    if verbose:
        print(f"      Generating dot plot with {len(valid_markers)} markers...")

    n_clusters = adata.obs[groupby].nunique()
    n_markers = len(valid_markers)

    fig_width = max(10, n_markers * 0.4 + 3)
    fig_height = max(6, n_clusters * 0.4 + 2)

    try:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sc.pl.dotplot(
            adata,
            var_names=valid_markers,
            groupby=groupby,
            dendrogram=False,
            standard_scale="var",
            cmap="Reds",
            ax=ax,
            show=False,
        )

        plt.title("Marker Gene Expression by Cluster", fontsize=14, pad=20)
        plt.tight_layout()

        out_path = os.path.join(output_dir, "marker_dotplot.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        if verbose:
            print(f"      Saved dot plot to {out_path}")

    except Exception as e:
        if verbose:
            print(f"      Warning: Could not generate dot plot: {e}")
        plt.close("all")

    _plot_grouped_dotplot(adata, groupby, output_dir, verbose)


def _plot_grouped_dotplot(adata, groupby, output_dir, n_genes=3, verbose=True):
    """Generate dot plot with markers grouped by their top cluster."""
    if "rank_genes_groups" not in adata.uns:
        return

    try:
        fig_width = 14
        fig_height = max(8, adata.obs[groupby].nunique() * 0.5 + 2)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sc.pl.rank_genes_groups_dotplot(
            adata,
            n_genes=n_genes,
            groupby=groupby,
            dendrogram=False,
            standard_scale="var",
            cmap="Reds",
            ax=ax,
            show=False,
        )

        plt.title(f"Top {n_genes} Markers per Cluster", fontsize=14, pad=20)
        plt.tight_layout()

        out_path = os.path.join(output_dir, "marker_dotplot_grouped.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        if verbose:
            print(f"      Saved grouped dot plot to {out_path}")

    except Exception as e:
        if verbose:
            print(f"      Warning: Could not generate grouped dot plot: {e}")
        plt.close("all")


# =============================================================================
# Common Marker Gene Visualization
# =============================================================================
def _plot_common_marker_genes(
    adata,
    marker_gene_path,
    output_dir,
    verbose=True,
    n_jobs=8,
):
    """
    Generate UMAP feature plots for common marker genes from a CSV file.
    Organizes plots by cell type in a resolution-independent folder.
    """
    if not os.path.exists(marker_gene_path):
        if verbose:
            print(f"      Warning: Common marker gene file not found: {marker_gene_path}")
        return

    if "X_umap" not in adata.obsm:
        if verbose:
            print("      No UMAP coordinates found, skipping common marker plots")
        return

    try:
        marker_df = pd.read_csv(marker_gene_path)
    except Exception as e:
        if verbose:
            print(f"      Warning: Could not read marker gene file: {e}")
        return

    if verbose:
        print(f"      Loaded common marker genes for {len(marker_df)} cell types")

    common_markers_dir = os.path.join(output_dir, "common_markers")
    os.makedirs(common_markers_dir, exist_ok=True)

    cell_type_col = "cell_type"
    if cell_type_col not in marker_df.columns:
        possible_cols = [c for c in marker_df.columns if "cell" in c.lower() or "type" in c.lower()]
        cell_type_col = possible_cols[0] if possible_cols else marker_df.columns[0]

    marker_cols = [c for c in marker_df.columns if c != cell_type_col]

    all_genes = set()
    gene_to_cell_types = {}
    cell_type_to_genes = {}

    for _, row in marker_df.iterrows():
        cell_type = str(row[cell_type_col]).strip()
        cell_type_clean = cell_type.replace("/", "_").replace(" ", "_").replace(":", "_")

        cell_type_to_genes[cell_type_clean] = []

        for col in marker_cols:
            gene = row.get(col)
            if pd.notna(gene) and gene:
                gene = str(gene).strip()
                if gene in adata.var_names:
                    all_genes.add(gene)
                    gene_to_cell_types.setdefault(gene, []).append(cell_type_clean)
                    cell_type_to_genes[cell_type_clean].append(gene)

    if len(all_genes) == 0:
        if verbose:
            print("      No common marker genes found in adata")
        return

    if verbose:
        print(f"      Found {len(all_genes)} valid common marker genes in adata")

    for cell_type_clean in cell_type_to_genes.keys():
        cell_type_dir = os.path.join(common_markers_dir, cell_type_clean)
        os.makedirs(cell_type_dir, exist_ok=True)

    temp_plot_dir = os.path.join(common_markers_dir, "_temp_plots")
    os.makedirs(temp_plot_dir, exist_ok=True)

    plotted_genes = _plot_gene_umaps_parallel(
        adata,
        list(all_genes),
        temp_plot_dir,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    import shutil

    for gene, source_path in plotted_genes.items():
        if not os.path.exists(source_path):
            continue
        for cell_type_clean in gene_to_cell_types.get(gene, []):
            cell_type_dir = os.path.join(common_markers_dir, cell_type_clean)
            dest_path = os.path.join(cell_type_dir, f"umap_{gene}.png")
            try:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                os.link(source_path, dest_path)
            except Exception:
                shutil.copy2(source_path, dest_path)

    shutil.rmtree(temp_plot_dir, ignore_errors=True)

    summary_data = []
    for cell_type_clean, genes in cell_type_to_genes.items():
        summary_data.append(
            {"cell_type": cell_type_clean, "n_markers": len(genes), "markers": ", ".join(genes)}
        )

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(common_markers_dir, "common_markers_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    if verbose:
        print(f"      Organized common marker UMAPs into {len(cell_type_to_genes)} cell type folders")
        print(f"      Saved common markers summary to {summary_path}")


# =============================================================================
# Other Helper Functions
# =============================================================================
def _save_resolution_comparison(adata, resolutions, resolution_results, output_dir, verbose=True):
    """Save resolution comparison summary and plots."""
    summary_data = []
    for res in resolutions:
        col_name = _resolution_to_col_name(res)
        n_clusters = resolution_results[res]["n_clusters"]
        cluster_sizes = adata.obs[col_name].value_counts()

        summary_data.append(
            {
                "resolution": res,
                "leiden_column": col_name,
                "n_clusters": n_clusters,
                "min_cluster_size": cluster_sizes.min(),
                "max_cluster_size": cluster_sizes.max(),
                "median_cluster_size": int(cluster_sizes.median()),
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "resolution_comparison.csv")
    summary_df.to_csv(summary_path, index=False)

    if verbose:
        print(f"\n   Saved resolution comparison to {summary_path}")
        print("\n   Resolution Comparison:")
        print("   " + "-" * 70)
        print(summary_df.to_string(index=False))
        print("   " + "-" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(summary_df["resolution"], summary_df["n_clusters"], "o-", color="steelblue", linewidth=2, markersize=10)

    for _, row in summary_df.iterrows():
        ax.annotate(
            f"{row['n_clusters']}",
            (row["resolution"], row["n_clusters"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    ax.set_xlabel("Resolution", fontsize=12)
    ax.set_ylabel("Number of Clusters", fontsize=12)
    ax.set_title("Leiden Clusters vs Resolution", fontsize=14)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "resolution_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"   Saved resolution comparison plot to {plot_path}")


def _plot_single_umap(adata, groupby, output_dir, verbose=True):
    """Plot a single UMAP with labels."""
    if groupby not in adata.obs.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    categories = adata.obs[groupby].astype("category").cat.categories
    n_cats = len(categories)

    if n_cats <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_cats, 1)))
    else:
        colors = plt.cm.gist_ncar(np.linspace(0, 0.95, n_cats))
    color_map = dict(zip(categories, colors))

    for cat in categories:
        mask = adata.obs[groupby] == cat
        coords = adata.obsm["X_umap"][mask]
        ax.scatter(coords[:, 0], coords[:, 1], c=[color_map[cat]], s=1, alpha=0.6, label=cat)
        centroid = coords.mean(axis=0)
        ax.annotate(
            str(cat)[:30],
            centroid,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"UMAP colored by {groupby}")
    if n_cats <= 30:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, markerscale=3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"umap_{groupby}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"   Saved UMAP to {out_path}")


def _plot_sample_umap(adata, sample_column, output_dir, verbose=True):
    """Plot UMAP colored by sample origin."""
    if sample_column not in adata.obs.columns:
        if verbose:
            print(f"   Warning: '{sample_column}' column not found, skipping sample UMAP")
        return

    if "X_umap" not in adata.obsm:
        if verbose:
            print("   Warning: No UMAP coordinates found, skipping sample UMAP")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    samples = adata.obs[sample_column].astype("category").cat.categories
    n_samples = len(samples)

    if n_samples <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_samples))
    elif n_samples <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_samples))
    else:
        colors = plt.cm.gist_ncar(np.linspace(0, 0.95, n_samples))

    color_map = dict(zip(samples, colors))

    for sample in samples:
        mask = adata.obs[sample_column] == sample
        coords = adata.obsm["X_umap"][mask]
        ax.scatter(coords[:, 0], coords[:, 1], c=[color_map[sample]], s=1, alpha=0.5, label=sample)

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"UMAP colored by {sample_column}")

    if n_samples <= 50:
        ncol = max(1, n_samples // 20 + 1)
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=6,
            markerscale=3,
            ncol=ncol,
        )
    else:
        if verbose:
            print(f"   Note: {n_samples} samples, legend omitted for clarity")

    plt.tight_layout()

    out_path = os.path.join(output_dir, f"umap_{sample_column}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"   Saved sample UMAP to {out_path}")


def _run_celltypist(adata, model_name=None, verbose=True):
    import celltypist
    from celltypist import models

    if model_name is None:
        model_name = "Immune_All_Low.pkl"
        if verbose:
            print(f"   Using default celltypist model: {model_name}")

    if os.path.exists(str(model_name)):
        model = models.Model.load(model_name)
    else:
        models.download_models(model=model_name)
        model = models.Model.load(model_name)

    predictions = celltypist.annotate(adata, model=model, majority_voting=False)
    adata.obs["cell_type"] = predictions.predicted_labels["predicted_labels"].values

    if verbose:
        n_types = adata.obs["cell_type"].nunique()
        print(f"   ✓ Celltypist assigned {n_types} cell types")

    return adata


def _majority_vote_cell_types(obs_df, cluster_col, celltype_col, verbose=True):
    results = []

    if verbose:
        print("   " + "-" * 60)

    for cluster in sorted(obs_df[cluster_col].unique(), key=lambda x: int(x)):
        mask = obs_df[cluster_col] == cluster
        celltype_counts = obs_df.loc[mask, celltype_col].value_counts()
        majority_type = celltype_counts.index[0]
        majority_count = celltype_counts.iloc[0]
        total_count = celltype_counts.sum()
        percentage = (majority_count / total_count) * 100
        top_types = celltype_counts.head(3).to_dict()

        results.append(
            {
                "leiden": cluster,
                "majority_cell_type": majority_type,
                "majority_count": majority_count,
                "total_cells": total_count,
                "majority_percentage": round(percentage, 2),
                "top_cell_types": str(top_types),
            }
        )

        if verbose:
            print(
                f"   Cluster {cluster:>2}: {majority_type:<30} "
                f"({majority_count:>5}/{total_count:<5}, {percentage:>5.1f}%)"
            )

    if verbose:
        print("   " + "-" * 60)

    return pd.DataFrame(results)


def _plot_gene_removal_summary(n_total, n_ensg, n_ig, n_tcr, output_dir, stage, verbose=True):
    """Plot gene removal/annotation summary pie chart."""
    fig, ax = plt.subplots(figsize=(10, 7))

    labels, sizes, colors = [], [], []
    n_remaining = n_total - n_ensg - n_ig - n_tcr

    if n_ensg > 0:
        labels.append(f"Pseudogenes (ENSG): {n_ensg}")
        sizes.append(n_ensg)
        colors.append("#ff6b6b")
    if n_ig > 0:
        labels.append(f"IG genes (BCR, annotated): {n_ig}")
        sizes.append(n_ig)
        colors.append("#ffd93d")
    if n_tcr > 0:
        labels.append(f"TCR genes (annotated): {n_tcr}")
        sizes.append(n_tcr)
        colors.append("#ffa94d")
    labels.append(f"Other genes: {n_remaining}")
    sizes.append(n_remaining)
    colors.append("#6bcb77")

    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)

    title_lines = [f"Gene Summary ({stage})", f"Total: {n_total} genes"]
    if n_ig > 0 or n_tcr > 0:
        title_lines.append(f"Immune receptor genes (IG + TCR): {n_ig + n_tcr}")
        title_lines.append("(All genes retained, IG/TCR annotated)")
    ax.set_title("\n".join(title_lines))

    out_path = os.path.join(output_dir, f"gene_summary_{stage}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"   Saved gene summary plot to {out_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def preprocess_pipeline(
    h5ad_path=None,
    output_dir=None,
    sample_meta_path=None,
    cell_meta_path=None,
    sample_column="sample",
    batch_key="batch",
    min_cells=500,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    vars_to_regress=None,
    num_features=2000,
    num_PCs=20,
    num_harmony=30,
    cluster_resolution=0.8,
    use_rep="X_pca_harmony",
    run_celltypist=True,
    celltypist_model=None,
    run_steps="all",
    adata_input=None,
    adata_cluster_input=None,
    adata_sample_input=None,
    vars_to_regress_for_harmony=None,
    compute_umap=True,
    generate_plots=True,
    find_markers=True,
    n_top_markers=5,
    n_marker_umap=3,
    common_marker_gene_path=None,
    n_jobs_plotting=8,
    save=True,
    verbose=True,
):
    set_global_seed(seed=42, verbose=verbose)
    start_time = time.time()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if run_steps == "all":
        steps_to_run = [1, 2, 3]
    elif run_steps == "qc":
        steps_to_run = [1]
    elif run_steps == "integrate":
        steps_to_run = [2]
    elif run_steps == "cluster":
        steps_to_run = [3]
    elif isinstance(run_steps, (list, tuple)):
        steps_to_run = list(run_steps)
    else:
        raise ValueError(f"Invalid run_steps: {run_steps}")

    if verbose:
        print("=" * 70)
        print("SINGLE-CELL PREPROCESSING PIPELINE")
        print("=" * 70)
        print(f"Steps: {steps_to_run}")
        if isinstance(cluster_resolution, (list, tuple)):
            print(f"Resolutions: {cluster_resolution}")
        if common_marker_gene_path:
            print(f"Common marker genes: {common_marker_gene_path}")
        print(f"Parallel workers for plotting: {n_jobs_plotting}")
        print("=" * 70)

    result = {}

    # Step 1
    if 1 in steps_to_run:
        if h5ad_path is None:
            raise ValueError("h5ad_path required for step 1")

        if verbose:
            print("\n" + "=" * 70)
            print("STEP 1: QC AND FILTERING")
            print("=" * 70)

        adata = sc.read_h5ad(h5ad_path)

        adata_filtered, vars_harmony = qc_and_filter(
            adata=adata,
            output_dir=output_dir,
            sample_column=sample_column,
            sample_meta_path=sample_meta_path,
            cell_meta_path=cell_meta_path,
            batch_key=batch_key,
            min_cells=min_cells,
            min_features=min_features,
            pct_mito_cutoff=pct_mito_cutoff,
            exclude_genes=exclude_genes,
            vars_to_regress=vars_to_regress,
            verbose=verbose,
        )

        result["adata_filtered"] = adata_filtered
        result["vars_to_regress_for_harmony"] = vars_harmony

        if save and output_dir and steps_to_run == [1]:
            preprocess_dir = os.path.join(output_dir, "preprocess")
            os.makedirs(preprocess_dir, exist_ok=True)
            safe_h5ad_write(
                adata_filtered,
                os.path.join(preprocess_dir, "adata_filtered.h5ad"),
                verbose=verbose,
            )

    # Step 2
    if 2 in steps_to_run:
        if 1 in steps_to_run:
            adata_for_step2 = result["adata_filtered"]
            vars_harmony = result["vars_to_regress_for_harmony"]
        elif adata_input is not None:
            adata_for_step2 = adata_input
            vars_harmony = vars_to_regress_for_harmony or [sample_column]
        else:
            raise ValueError("adata_input required for step 2 when skipping step 1")

        if verbose:
            print("\n" + "=" * 70)
            print("STEP 2: NORMALIZATION AND INTEGRATION")
            print("=" * 70)

        save_step2 = save and (3 not in steps_to_run)

        adata_cluster, adata_sample = normalize_and_integrate(
            adata=adata_for_step2,
            output_dir=output_dir,
            sample_column=sample_column,
            num_features=num_features,
            num_PCs=num_PCs,
            num_harmony=num_harmony,
            vars_to_regress_for_harmony=vars_harmony,
            save=save_step2,
            verbose=verbose,
        )

        result["adata_cluster"] = adata_cluster
        result["adata_sample"] = adata_sample

    # Step 3
    if 3 in steps_to_run:
        if 2 in steps_to_run:
            adata_cluster = result["adata_cluster"]
            adata_sample = result["adata_sample"]
        elif adata_cluster_input is not None:
            adata_cluster = adata_cluster_input
            adata_sample = adata_sample_input
        else:
            raise ValueError("adata_cluster_input required for step 3 when skipping step 2")

        if verbose:
            print("\n" + "=" * 70)
            print("STEP 3: CLUSTERING AND ANNOTATION")
            print("=" * 70)

        outputs = cluster_and_annotate(
            adata_cluster=adata_cluster,
            adata_sample=adata_sample,
            output_dir=output_dir,
            sample_column=sample_column,
            cluster_resolution=cluster_resolution,
            use_rep=use_rep,
            num_PCs=num_PCs,
            run_celltypist=run_celltypist,
            celltypist_model=celltypist_model,
            compute_umap=compute_umap,
            find_markers=find_markers,
            save=save,
            verbose=verbose,
            generate_plots=generate_plots,
            n_top_markers=n_top_markers,
            n_marker_umap=n_marker_umap,
            common_marker_gene_path=common_marker_gene_path,
            n_jobs_plotting=n_jobs_plotting,
        )

        if adata_sample is not None:
            result["adata_cluster"], result["adata_sample"] = outputs
        else:
            result["adata_cluster"] = outputs

    elapsed = time.time() - start_time
    if verbose:
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Runtime: {elapsed:.2f}s")
        print(f"Results: {list(result.keys())}")
        if "adata_cluster" in result:
            print(f"adata_cluster: {result['adata_cluster'].shape}")
            print(f"   Total genes: {result['adata_cluster'].shape[1]}")
            if "highly_variable" in result["adata_cluster"].var.columns:
                print(f"   HVG genes: {result['adata_cluster'].var['highly_variable'].sum()}")
            if "is_ig" in result["adata_cluster"].var.columns:
                print(f"   IG genes: {result['adata_cluster'].var['is_ig'].sum()}")
            if "is_tcr" in result["adata_cluster"].var.columns:
                print(f"   TCR genes: {result['adata_cluster'].var['is_tcr'].sum()}")
            leiden_cols = [c for c in result["adata_cluster"].obs.columns if c.startswith("leiden_")]
            print(f"   Leiden columns: {leiden_cols}")
        if "adata_sample" in result:
            print(f"adata_sample: {result['adata_sample'].shape}")
        print("=" * 70)

    return result


# =============================================================================
# RUN PIPELINE
# =============================================================================
if __name__ == "__main__":
    result = preprocess_pipeline(
        h5ad_path=H5AD_PATH,
        output_dir=OUTPUT_DIR,
        sample_meta_path=SAMPLE_META_PATH,
        cell_meta_path=CELL_META_PATH,
        sample_column=SAMPLE_COLUMN,
        batch_key=BATCH_KEY,
        min_cells=MIN_CELLS,
        min_features=MIN_FEATURES,
        pct_mito_cutoff=PCT_MITO_CUTOFF,
        exclude_genes=EXCLUDE_GENES,
        vars_to_regress=VARS_TO_REGRESS,
        num_features=NUM_FEATURES,
        num_PCs=NUM_PCS,
        num_harmony=NUM_HARMONY,
        cluster_resolution=CLUSTER_RESOLUTION,
        use_rep=USE_REP,
        run_celltypist=RUN_CELLTYPIST,
        celltypist_model=CELLTYPIST_MODEL,
        run_steps=RUN_STEPS,
        compute_umap=COMPUTE_UMAP,
        generate_plots=GENERATE_PLOTS,
        find_markers=FIND_MARKERS,
        n_top_markers=N_TOP_MARKERS,
        n_marker_umap=N_MARKER_UMAP,
        common_marker_gene_path=COMMON_MARKER_GENE_PATH,
        n_jobs_plotting=N_JOBS_PLOTTING,
        save=SAVE,
        verbose=VERBOSE,
    )