#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import scanpy.external as sce
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for HPC
matplotlib.use("Agg")


def _qc_plots(
    adata: ad.AnnData,
    output_dir: str,
    prefix: str = "qc",
    sample_column: Optional[str] = None,
) -> None:
    """
    Make basic QC plots and save them as PNGs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Violin plots for basic QC metrics
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
        show=False,
    )
    plt.savefig(
        os.path.join(output_dir, f"{prefix}_violin.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # 2) Scatter: total_counts vs pct_counts_mt
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="pct_counts_mt",
        show=False,
    )
    plt.savefig(
        os.path.join(output_dir, f"{prefix}_scatter_counts_vs_mito.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # 3) Scatter: total_counts vs n_genes_by_counts
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        show=False,
    )
    plt.savefig(
        os.path.join(output_dir, f"{prefix}_scatter_counts_vs_genes.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # 4) Per-sample cell counts barplot (if sample_column exists)
    if sample_column is not None and sample_column in adata.obs.columns:
        # Avoid pandas FutureWarning by explicitly setting observed=False
        cell_counts = (
            adata.obs.groupby(sample_column, observed=False).size().sort_values()
        )
        plt.figure(figsize=(10, 4))
        cell_counts.plot(kind="bar")
        plt.ylabel("Number of cells")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}_cells_per_sample.png"),
            dpi=300,
        )
        plt.close()


def _umap_plots(
    adata: ad.AnnData,
    output_dir: str,
    prefix: str = "umap",
    sample_column: str = "sample",
    cell_type_column: str = "cell_type",
) -> None:
    """
    Make UMAP plots colored by sample and cell type (if present).
    """
    os.makedirs(output_dir, exist_ok=True)

    # UMAP colored by sample
    if sample_column in adata.obs.columns:
        sc.pl.umap(
            adata,
            color=[sample_column],
            wspace=0.4,
            show=False,
        )
        plt.savefig(
            os.path.join(output_dir, f"{prefix}_by_{sample_column}.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    # UMAP colored by cell type
    if cell_type_column in adata.obs.columns:
        sc.pl.umap(
            adata,
            color=[cell_type_column],
            wspace=0.4,
            show=False,
        )
        plt.savefig(
            os.path.join(output_dir, f"{prefix}_by_{cell_type_column}.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


def long_covid_qc_and_dr(
    h5ad_path: str,
    output_dir: str,
    # QC params from your list:
    min_cells_per_gene: int = 500,
    min_genes_per_cell: int = 500,
    max_mito_fraction: float = 0.20,  # 20%
    # DR / integration params:
    n_hvgs: int = 2000,
    n_pcs: int = 20,
    harmony_max_iter: int = 30,
    sample_column: str = "sample",
    cell_type_column: str = "cell_type",
    min_cells_per_sample: Optional[int] = None,  # set e.g. 500 if you want
    run_doublet_detection: bool = False,  # placeholder; see note in code
    save_filtered: bool = True,
) -> ad.AnnData:
    """
    End-to-end pipeline for:
      - QC
      - Normalization (10k counts/cell, log1p)
      - HVG selection (Seurat v3, 2000 genes, on raw counts)
      - PCA (20 PCs)
      - Harmony integration on 'sample'
      - Neighbors + UMAP + Leiden clustering

    Assumes the merged .h5ad is raw counts from Cell Ranger.
    Keeps any existing 'cell_type' annotations in .obs.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Loading merged AnnData from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"Raw shape: {adata.n_obs} cells × {adata.n_vars} genes")

    # --- QC metrics ---
    print("=== Computing QC metrics ===")
    # Mito genes (human: MT-)
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

    # Optional ribosomal flags (not used for filtering, but useful to inspect)
    adata.var["ribo"] = adata.var_names.str.upper().str.startswith(("RPS", "RPL"))

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # --- Pre-filter QC plots ---
    print("=== QC plots: pre-filtering ===")
    _qc_plots(
        adata,
        output_dir=plot_dir,
        prefix="qc_prefilter",
        sample_column=sample_column,
    )

    # --- Basic gene/cell filtering ---
    print("=== Basic gene/cell filtering ===")
    # 1) Filter genes by min_cells_per_gene
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    print(
        f"After gene filter (min_cells={min_cells_per_gene}): "
        f"{adata.n_obs} cells × {adata.n_vars} genes"
    )

    # 2) Filter cells by number of detected genes
    cell_mask = adata.obs["n_genes_by_counts"] >= min_genes_per_cell
    print(
        f"Filtering cells with n_genes_by_counts < {min_genes_per_cell}: "
        f"keeping {cell_mask.sum()} / {adata.n_obs}"
    )
    adata = adata[cell_mask].copy()

    # 3) Filter cells by mitochondrial fraction
    mito_mask = adata.obs["pct_counts_mt"] <= (max_mito_fraction * 100.0)
    print(
        f"Filtering cells with pct_counts_mt > {max_mito_fraction * 100:.1f}%: "
        f"keeping {mito_mask.sum()} / {adata.n_obs}"
    )
    adata = adata[mito_mask].copy()

    print(
        f"After basic cell filters: {adata.n_obs} cells × {adata.n_vars} genes"
    )

    # --- Optional per-sample filtering ---
    if sample_column in adata.obs.columns and min_cells_per_sample is not None:
        print(
            f"=== Per-sample filtering: min_cells_per_sample = {min_cells_per_sample} ==="
        )
        # Avoid FutureWarning with observed=False
        cell_counts = (
            adata.obs.groupby(sample_column, observed=False).size()
        )
        keep_samples = cell_counts[cell_counts >= min_cells_per_sample].index
        print(
            f"Keeping {len(keep_samples)} / {len(cell_counts)} samples "
            f"with ≥ {min_cells_per_sample} cells"
        )
        adata = adata[adata.obs[sample_column].isin(keep_samples)].copy()
        print(
            f"After per-sample filter: {adata.n_obs} cells × {adata.n_vars} genes"
        )

    # --- Recompute QC metrics after filtering ---
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # --- QC plots after filtering ---
    print("=== QC plots: post-filtering ===")
    _qc_plots(
        adata,
        output_dir=plot_dir,
        prefix="qc_postfilter",
        sample_column=sample_column,
    )

    # --- (Optional) Doublet detection placeholder ---
    # NOTE: You said "doublet detection enabled (though commented out in code)".
    # Here we keep it optional; you can plug in scrublet or sc.pp.scrublet if you like.
    if run_doublet_detection:
        print("=== Doublet detection is requested but not implemented in this stub ===")
        # Example placeholder:
        # import scrublet as scr
        # (implement scrublet here)
        # For now, we just print a message.

    # --- Store raw counts in a layer before normalization ---
    if "counts" not in adata.layers:
        print("Storing raw counts in adata.layers['counts']")
        adata.layers["counts"] = adata.X.copy()

    # --- HVG selection (Seurat v3, 2000 genes, on raw counts) ---
    print("=== HVG selection (Seurat v3, n_hvgs={}) ===".format(n_hvgs))
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=n_hvgs,
        layer="counts",  # use raw counts to avoid non-integer warning
        subset=True,
        inplace=True,
    )
    print(f"After HVG selection: {adata.n_obs} cells × {adata.n_vars} HVGs")

    # --- Normalization & log1p ---
    print("=== Normalization (10k counts/cell) and log1p ===")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Keep a copy of normalized, log1p data as .raw for plotting later
    adata.raw = adata

    # --- PCA ---
    print(f"=== PCA (n_pcs={n_pcs}) ===")
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    print("Stored PCA in adata.obsm['X_pca']")

    # --- Harmony integration on sample_column ---
    print(
        f"=== Harmony integration on '{sample_column}' (max_iter={harmony_max_iter}) ==="
    )
    if sample_column not in adata.obs.columns:
        raise KeyError(
            f"Expected '{sample_column}' in adata.obs for Harmony batch correction."
        )

    sce.pp.harmony_integrate(
        adata,
        key=sample_column,
        basis="X_pca",
        max_iter_harmony=harmony_max_iter,
    )
    # Harmony result is typically stored as 'X_pca_harmony'
    if "X_pca_harmony" not in adata.obsm:
        raise KeyError(
            "Harmony did not produce 'X_pca_harmony' in adata.obsm. "
            "Check your scanpy.external.harmony version."
        )

    # --- Neighbors, UMAP, Leiden ---
    print("=== Neighbors, UMAP, Leiden clustering ===")
    sc.pp.neighbors(
        adata,
        use_rep="X_pca_harmony",
        n_pcs=None,  # using the representation directly
    )
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5, key_added="leiden")

    print("Stored UMAP in adata.obsm['X_umap'] and Leiden labels in adata.obs['leiden']")

    # --- UMAP plots (by sample and cell_type) ---
    print("=== UMAP plots ===")
    _umap_plots(
        adata,
        output_dir=plot_dir,
        prefix="umap",
        sample_column=sample_column,
        cell_type_column=cell_type_column,
    )

    # --- Save processed AnnData ---
    if save_filtered:
        base = Path(h5ad_path).with_suffix("")
        out_path = os.path.join(
            output_dir, base.name + "_qc_harmony_umap.h5ad"
        )
        print(f"Saving processed AnnData to: {out_path}")
        adata.write_h5ad(out_path)

    print("Long COVID QC + DR pipeline complete.")
    return adata


if __name__ == "__main__":
    # User-modifiable section
    MERGED_H5AD = "/dcl01/hongkai/data/data/hjiang/Data/long_covid/long_covid.h5ad"
    OUTPUT_DIR = "/dcs07/hongkai/data/harry/result/long_covid/QC"

    long_covid_qc_and_dr(
        h5ad_path=MERGED_H5AD,
        output_dir=OUTPUT_DIR,
        min_cells_per_gene=500,      # QC parameter
        min_genes_per_cell=500,      # QC parameter
        max_mito_fraction=0.20,      # 20% mito cutoff
        n_hvgs=2000,                 # HVG parameter
        n_pcs=20,                    # PCA components
        harmony_max_iter=30,         # Harmony iterations
        sample_column="sample",      # batch correction at sample level
        cell_type_column="cell_type",
        min_cells_per_sample=None,   # set to 500 if you want per-sample filtering
        run_doublet_detection=False, # placeholder; see note above
        save_filtered=True,
    )
