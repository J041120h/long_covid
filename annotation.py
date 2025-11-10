#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Optional

import numpy as np
import scanpy as sc
import anndata as ad
import celltypist
from celltypist import models
import matplotlib
import matplotlib.pyplot as plt

# Non-interactive backend for HPC
matplotlib.use("Agg")


def find_markers_and_heatmaps(
    adata: ad.AnnData,
    output_dir: str,
    cluster_key: str = "leiden",
    n_top: int = 10,
) -> None:
    """
    Find marker genes per cluster using rank_genes_groups and generate heatmaps:
      - One combined heatmap for all clusters
      - One heatmap per cluster.
    """
    marker_dir = os.path.join(output_dir, "markers")
    os.makedirs(marker_dir, exist_ok=True)

    if cluster_key not in adata.obs.columns:
        raise KeyError(f"Cluster key '{cluster_key}' not found in adata.obs")

    print(f"[INFO] Finding marker genes for clusters in '{cluster_key}'")

    # Use raw data if available (better for DE)
    use_raw = adata.raw is not None
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        use_raw=use_raw,
        method="wilcoxon",
    )

    rank = adata.uns["rank_genes_groups"]
    groups = rank["names"].dtype.names

    # Collect top genes per cluster
    top_genes_all = []
    top_genes_per_cluster = {}
    for g in groups:
        genes_g = list(rank["names"][g][:n_top])
        top_genes_per_cluster[g] = genes_g
        top_genes_all.extend(genes_g)
    # Unique list while preserving order
    top_genes_all = list(dict.fromkeys(top_genes_all))

    # --- Combined heatmap across all clusters ---
    print("[INFO] Plotting combined marker heatmap for all clusters")
    sc.pl.heatmap(
        adata,
        var_names=top_genes_all,
        groupby=cluster_key,
        show=False,
        cmap="viridis",
    )
    plt.savefig(
        os.path.join(marker_dir, f"markers_heatmap_{cluster_key}_all_clusters.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # --- Individual heatmaps per cluster ---
    for g in groups:
        genes_g = top_genes_per_cluster[g]
        print(f"[INFO] Plotting marker heatmap for cluster {g}")
        # Subset to this cluster only
        adata_g = adata[adata.obs[cluster_key] == g].copy()
        sc.pl.heatmap(
            adata_g,
            var_names=genes_g,
            groupby=cluster_key,
            show=False,
            cmap="viridis",
        )
        plt.savefig(
            os.path.join(marker_dir, f"markers_heatmap_{cluster_key}_{g}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def _build_celltypist_input_adata(adata: ad.AnnData) -> ad.AnnData:
    """
    Build an AnnData suitable for CellTypist:

    Preference order (following the tutorial expectations):
      1) adata.layers['counts']  (raw counts, best for CellTypist)
      2) adata.raw               (often raw/log1p counts)
      3) adata.X                 (fallback; not recommended unless it's log1p CPM)

    This function only chooses the source matrix; normalization/log1p are done later.
    """
    if "counts" in adata.layers:
        print("[INFO] Using adata.layers['counts'] as input for CellTypist (raw counts).")
        X = adata.layers["counts"]
        adata_ct = ad.AnnData(
            X=X.copy(),
            obs=adata.obs.copy(),
            var=adata.var.copy(),
        )
    elif adata.raw is not None:
        print("[INFO] Using adata.raw as input for CellTypist.")
        adata_ct = adata.raw.to_adata()
        # Ensure obs matches main AnnData for consistent indexing
        adata_ct.obs = adata.obs.copy()
    else:
        print("[INFO] Using adata.X as input for CellTypist (fallback).")
        adata_ct = adata.copy()

    return adata_ct


def annotate_cell_types_with_celltypist(
    adata: ad.AnnData,
    output_dir: str,
    model_name: Optional[str] = None,
    custom_model_path: Optional[str] = None,
    majority_voting: bool = True,
) -> ad.AnnData:
    """
    Annotate cell types using CellTypist and add:
        - adata.obs['cell_type']
        - adata.obs['celltypist_conf_score']

    Based on the official CellTypist tutorial:
      - The input expression matrix for CellTypist should be
        log1p-normalised to 10,000 counts per cell.

    Here we:
      1) Build a CellTypist input AnnData from layers['counts'] (preferred).
      2) Normalize to 1e4 and log1p if using counts.
      3) If not counts, heuristically avoid double-log-transform.
      4) Clean NaN/Inf before calling the underlying sklearn LogisticRegression.
    """
    if (model_name is None and custom_model_path is None) or (
        model_name is not None and custom_model_path is not None
    ):
        raise ValueError("Provide exactly one of `model_name` or `custom_model_path`.")

    plot_dir = os.path.join(output_dir, "celltypist")
    os.makedirs(plot_dir, exist_ok=True)

    # --- Build input AnnData for CellTypist ---
    adata_ct = _build_celltypist_input_adata(adata)

    # --- Normalize/log for CellTypist ---
    using_counts = "counts" in adata.layers
    if using_counts:
        # This matches the tutorial: normalize_total(target_sum=1e4) then log1p
        print("[INFO] Normalizing and log-transforming counts for CellTypist (1e4 + log1p).")
        sc.pp.normalize_total(adata_ct, target_sum=1e4)
        sc.pp.log1p(adata_ct)
    else:
        # For non-counts input, avoid double log-transform on already log1p data
        print("[INFO] Input for CellTypist is not raw counts; checking if extra normalize/log is needed.")
        X_sample = adata_ct.X
        if not isinstance(X_sample, np.ndarray):
            X_sample = X_sample[:100].toarray()
        else:
            X_sample = X_sample[:100]

        # Very simple heuristic: values > 20 are unlikely to be log1p counts
        if (X_sample > 20).sum() > 0:
            print("[INFO] Detected large values, performing normalize_total + log1p for CellTypist.")
            sc.pp.normalize_total(adata_ct, target_sum=1e4)
            sc.pp.log1p(adata_ct)
        else:
            print("[INFO] Data already appears log-transformed; skipping extra normalize/log for CellTypist.")

    # --- Ensure dense matrix for CellTypist / sklearn ---
    if not isinstance(adata_ct.X, np.ndarray):
        adata_ct.X = adata_ct.X.toarray()

    # --- NaN/Inf cleanup before sklearn LogisticRegression ---
    n_total = adata_ct.X.size
    n_nans = int(np.isnan(adata_ct.X).sum())
    n_posinf = int(np.isposinf(adata_ct.X).sum())
    n_neginf = int(np.isneginf(adata_ct.X).sum())
    if n_nans or n_posinf or n_neginf:
        print(
            f"[WARN] Found problematic values in adata_ct.X "
            f"(NaN: {n_nans}, +Inf: {n_posinf}, -Inf: {n_neginf} "
            f"out of {n_total} entries). Replacing with 0."
        )
        adata_ct.X = np.nan_to_num(adata_ct.X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        print("[INFO] No NaN/Inf detected in adata_ct.X before CellTypist.")

    # --- Load CellTypist model ---
    if model_name is not None:
        print(f"[INFO] Preparing CellTypist model: {model_name}")
        # Try to download/update index; if no internet, just warn and continue
        try:
            models.download_models(force_update=False, model=[model_name])
        except Exception as e:
            print(
                "[WARN] Could not contact CellTypist server to download models.\n"
                f"       Error: {e}\n"
                "       Will try to load the model from local cache only."
            )
        # Now try to load from local cache
        try:
            model = models.Model.load(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Could not load CellTypist model '{model_name}' from local cache.\n"
                "You likely need to:\n"
                "  1) Run celltypist.models.download_models(model=['"
                + model_name
                + "']) on a node with internet, OR\n"
                "  2) Download the .pkl manually and set custom_model_path to that file."
            ) from e
    else:
        if not os.path.exists(custom_model_path):
            raise FileNotFoundError(f"Custom model not found: {custom_model_path}")
        print(f"[INFO] Loading custom CellTypist model: {custom_model_path}")
        model = models.Model.load(custom_model_path)

    # --- Run CellTypist ---
    print(f"[INFO] Running CellTypist with model: {model_name or custom_model_path}")
    predictions = celltypist.annotate(
        adata_ct,
        model=model,
        majority_voting=majority_voting,
    )
    pred_adata = predictions.to_adata()

    # --- Align and transfer labels back to original AnnData ---
    common_idx = adata.obs.index.intersection(pred_adata.obs.index)
    if len(common_idx) == 0:
        raise RuntimeError(
            "No overlapping cell indices between original AnnData and CellTypist predictions. "
            "Check that obs indices have not been altered."
        )

    adata = adata[common_idx].copy()
    pred_adata = pred_adata[common_idx]

    adata.obs["cell_type"] = pred_adata.obs["majority_voting"]
    adata.obs["celltypist_conf_score"] = pred_adata.obs["conf_score"]

    # --- Cell type bar plot ---
    plt.figure(figsize=(10, 6))
    adata.obs["cell_type"].value_counts().sort_values(ascending=False).plot(kind="bar")
    plt.title("Predicted Cell Types (CellTypist)")
    plt.ylabel("Number of Cells")
    plt.xlabel("Cell Type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, "celltypist_cell_label_distribution.png"),
        dpi=300,
    )
    plt.close()

    # --- UMAP plot colored by cell type ---
    if "X_umap" not in adata.obsm:
        print("[INFO] No UMAP found, computing one for visualization.")
        if "X_pca_harmony" in adata.obsm:
            sc.pp.neighbors(adata, use_rep="X_pca_harmony")
        elif "X_pca" in adata.obsm:
            sc.pp.neighbors(adata, use_rep="X_pca")
        else:
            sc.pp.pca(adata, svd_solver="arpack")
            sc.pp.neighbors(adata, n_pcs=20)
        sc.tl.umap(adata)

    sc.pl.umap(
        adata,
        color="cell_type",
        title="CellTypist Annotated Cell Types",
        show=False,
    )
    plt.savefig(
        os.path.join(plot_dir, "celltypist_umap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return adata


def run_leiden_and_celltypist(
    h5ad_path: str,
    output_dir: str,
    leiden_resolution: float = 0.5,
    sample_column: str = "sample",
    cell_type_column: str = "cell_type",
    celltypist_model_name: Optional[str] = None,
    custom_model_path: Optional[str] = None,
) -> None:
    """
    Load AnnData, ensure Leiden clusters exist, find marker genes and heatmaps,
    then run CellTypist annotation and write everything back to the SAME .h5ad file.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading AnnData from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"[INFO] Shape: {adata.n_obs} cells × {adata.n_vars} genes")

    # --- Ensure neighbors, Leiden, and UMAP exist ---
    if "neighbors" not in adata.uns:
        print("[INFO] Computing neighbors...")
        if "X_pca_harmony" in adata.obsm:
            sc.pp.neighbors(adata, use_rep="X_pca_harmony")
        elif "X_pca" in adata.obsm:
            sc.pp.neighbors(adata, use_rep="X_pca")
        else:
            # Fallback path if no PCA/harmony present; note this acts on adata.X,
            # which in your inspected file is already scaled, so this should rarely run.
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.pca(adata, svd_solver="arpack")
            sc.pp.neighbors(adata, n_pcs=20)

    if "leiden" not in adata.obs.columns:
        print(f"[INFO] Running Leiden clustering (resolution={leiden_resolution})...")
        sc.tl.leiden(adata, resolution=leiden_resolution, key_added="leiden")
    else:
        print("[INFO] Using existing Leiden clustering in adata.obs['leiden'].")

    if "X_umap" not in adata.obsm:
        print("[INFO] Computing UMAP embedding...")
        sc.tl.umap(adata)
    else:
        print("[INFO] Using existing UMAP embedding in adata.obsm['X_umap'].")

    # --- Marker genes & heatmaps BEFORE annotation ---
    print("[INFO] Finding marker genes and generating heatmaps before annotation.")
    find_markers_and_heatmaps(
        adata=adata,
        output_dir=output_dir,
        cluster_key="leiden",
        n_top=10,
    )

    # --- Run CellTypist annotation ---
    adata = annotate_cell_types_with_celltypist(
        adata=adata,
        output_dir=output_dir,
        model_name=celltypist_model_name,
        custom_model_path=custom_model_path,
        majority_voting=True,
    )

    # --- Map Leiden clusters to majority cell type ---
    if "leiden" in adata.obs.columns and cell_type_column in adata.obs.columns:
        cluster_map = (
            adata.obs.groupby("leiden")[cell_type_column]
            .agg(lambda x: x.value_counts().idxmax())
        )
        adata.uns["leiden_to_cell_type"] = cluster_map.to_dict()
        print("[INFO] Stored Leiden → cell_type mapping in adata.uns['leiden_to_cell_type'].")

    # --- Write back to same file ---
    print(f"[INFO] Writing updated AnnData back to: {h5ad_path}")
    adata.write_h5ad(h5ad_path, compression="gzip")
    print("[INFO] Annotation and marker analysis complete.")


if __name__ == "__main__":
    H5AD_PATH = "/dcs07/hongkai/data/harry/result/long_covid/QC/long_covid_qc_harmony_umap.h5ad"
    OUTPUT_DIR = "/dcs07/hongkai/data/harry/result/long_covid/annotation"

    run_leiden_and_celltypist(
        h5ad_path=H5AD_PATH,
        output_dir=OUTPUT_DIR,
        leiden_resolution=0.5,
        sample_column="sample",
        cell_type_column="cell_type",
        # COVID immunology: use PaediatricAdult_COVID19_PBMC by default
        custom_model_path="/users/hjiang/GenoDistance/long_covid/PaediatricAdult_COVID19_PBMC.pkl",
    )
