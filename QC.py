#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for HPC
matplotlib.use("Agg")


def _get_sample_stats(
    adata: ad.AnnData,
    sample_column: str = "sample",
) -> pd.DataFrame:
    """
    Compute per-sample statistics: cell counts and median genes/UMIs per cell.
    """
    if sample_column not in adata.obs.columns:
        return pd.DataFrame()

    stats = adata.obs.groupby(sample_column, observed=False).agg(
        n_cells=("n_genes_by_counts", "size"),
        median_genes=("n_genes_by_counts", "median"),
        median_counts=("total_counts", "median"),
        median_pct_mt=("pct_counts_mt", "median"),
    )
    return stats.sort_index()


def _compare_sample_stats(
    pre_stats: pd.DataFrame,
    post_stats: pd.DataFrame,
    output_dir: str,
    prefix: str = "sample_comparison",
) -> pd.DataFrame:
    """
    Compare pre- and post-filtering sample statistics.
    Returns a merged DataFrame and saves it as CSV.
    """
    if pre_stats.empty or post_stats.empty:
        return pd.DataFrame()

    # Rename columns for clarity
    pre_stats = pre_stats.add_suffix("_pre")
    post_stats = post_stats.add_suffix("_post")

    # Merge on sample index
    comparison = pre_stats.join(post_stats, how="outer")

    # Calculate cells lost
    comparison["cells_lost"] = (
        comparison["n_cells_pre"] - comparison["n_cells_post"].fillna(0)
    )
    comparison["pct_cells_retained"] = (
        comparison["n_cells_post"].fillna(0) / comparison["n_cells_pre"] * 100
    ).round(2)

    # Sort by percentage retained (ascending) to highlight problematic samples
    comparison = comparison.sort_values("pct_cells_retained", ascending=True)

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{prefix}.csv")
    comparison.to_csv(csv_path)
    print(f"Sample comparison saved to: {csv_path}")

    return comparison


def _plot_sample_comparison(
    comparison: pd.DataFrame,
    output_dir: str,
    prefix: str = "sample_comparison",
) -> None:
    """
    Plot before/after cell counts per sample and retention percentage.
    """
    if comparison.empty:
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1) Side-by-side barplot: pre vs post cell counts
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sort by pre-filter cell count for this plot
    plot_df = comparison.sort_values("n_cells_pre", ascending=True)

    x = np.arange(len(plot_df))
    width = 0.35

    axes[0].barh(x - width / 2, plot_df["n_cells_pre"], width, label="Pre-QC", alpha=0.8)
    axes[0].barh(x + width / 2, plot_df["n_cells_post"].fillna(0), width, label="Post-QC", alpha=0.8)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(plot_df.index, fontsize=8)
    axes[0].set_xlabel("Number of Cells")
    axes[0].set_title("Cells per Sample: Pre vs Post QC")
    axes[0].legend()

    # 2) Retention percentage barplot (sorted by retention)
    retention_sorted = comparison.sort_values("pct_cells_retained", ascending=True)
    colors = ["red" if v < 50 else "orange" if v < 70 else "green" for v in retention_sorted["pct_cells_retained"]]

    axes[1].barh(range(len(retention_sorted)), retention_sorted["pct_cells_retained"], color=colors)
    axes[1].set_yticks(range(len(retention_sorted)))
    axes[1].set_yticklabels(retention_sorted.index, fontsize=8)
    axes[1].set_xlabel("% Cells Retained")
    axes[1].set_title("Cell Retention by Sample (red<50%, orange<70%, green≥70%)")
    axes[1].axvline(x=50, color="red", linestyle="--", alpha=0.5)
    axes[1].axvline(x=70, color="orange", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{prefix}_barplot.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # 3) Scatter: pre vs post cell counts
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        comparison["n_cells_pre"],
        comparison["n_cells_post"].fillna(0),
        alpha=0.6,
        edgecolors="k",
        linewidth=0.5,
    )
    # Diagonal line (y=x)
    max_val = max(comparison["n_cells_pre"].max(), comparison["n_cells_post"].max())
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="y=x (no loss)")
    ax.set_xlabel("Pre-QC Cell Count")
    ax.set_ylabel("Post-QC Cell Count")
    ax.set_title("Per-Sample Cell Counts: Pre vs Post QC")
    ax.legend()

    # Annotate samples with low retention
    low_retention = comparison[comparison["pct_cells_retained"] < 50]
    for sample_name, row in low_retention.iterrows():
        ax.annotate(
            sample_name,
            (row["n_cells_pre"], row["n_cells_post"] if pd.notna(row["n_cells_post"]) else 0),
            fontsize=7,
            alpha=0.7,
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{prefix}_scatter.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


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


def _print_sample_summary(
    comparison: pd.DataFrame,
    sample_column: str = "sample",
) -> None:
    """
    Print a text summary of sample-level QC impact.
    """
    if comparison.empty:
        return

    print("\n" + "=" * 60)
    print(f"SAMPLE-LEVEL QC SUMMARY ('{sample_column}' column)")
    print("=" * 60)

    n_samples_pre = (comparison["n_cells_pre"] > 0).sum()
    n_samples_post = (comparison["n_cells_post"].fillna(0) > 0).sum()
    samples_lost = n_samples_pre - n_samples_post

    total_cells_pre = comparison["n_cells_pre"].sum()
    total_cells_post = comparison["n_cells_post"].fillna(0).sum()
    total_pct_retained = (total_cells_post / total_cells_pre * 100) if total_cells_pre > 0 else 0

    print(f"Samples: {n_samples_pre} -> {n_samples_post} ({samples_lost} removed entirely)")
    print(f"Total cells: {int(total_cells_pre):,} -> {int(total_cells_post):,} ({total_pct_retained:.1f}% retained)")

    # Breakdown by retention
    low_retention = comparison[comparison["pct_cells_retained"] < 50]
    med_retention = comparison[(comparison["pct_cells_retained"] >= 50) & (comparison["pct_cells_retained"] < 70)]
    high_retention = comparison[comparison["pct_cells_retained"] >= 70]

    print(f"\nRetention breakdown:")
    print(f"  <50% retained:  {len(low_retention)} samples (FLAGGED)")
    print(f"  50-70% retained: {len(med_retention)} samples")
    print(f"  ≥70% retained:  {len(high_retention)} samples")

    if len(low_retention) > 0:
        print(f"\n⚠️  Low-retention samples (<50%):")
        for sample_name, row in low_retention.iterrows():
            print(
                f"    {sample_name}: {int(row['n_cells_pre'])} -> "
                f"{int(row['n_cells_post']) if pd.notna(row['n_cells_post']) else 0} "
                f"({row['pct_cells_retained']:.1f}%)"
            )

    print("=" * 60 + "\n")


def long_covid_qc_and_dr(
    h5ad_path: str,
    output_dir: str,
    # QC params:
    min_cells_per_gene: int = 500,
    min_genes_per_cell: int = 500,
    max_mito_fraction: float = 0.20,
    # DR / integration params:
    n_hvgs: int = 2000,
    n_pcs: int = 20,
    harmony_max_iter: int = 30,
    sample_column: str = "sample",
    cell_type_column: str = "cell_type",
    min_cells_per_sample: Optional[int] = None,
    run_doublet_detection: bool = False,
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
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.upper().str.startswith(("RPS", "RPL"))

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # --- Capture pre-filter sample stats ---
    print("=== Capturing pre-filter sample statistics ===")
    pre_filter_stats = _get_sample_stats(adata, sample_column=sample_column)
    if not pre_filter_stats.empty:
        print(f"Pre-filter: {len(pre_filter_stats)} samples detected")

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
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    print(
        f"After gene filter (min_cells={min_cells_per_gene}): "
        f"{adata.n_obs} cells × {adata.n_vars} genes"
    )

    cell_mask = adata.obs["n_genes_by_counts"] >= min_genes_per_cell
    print(
        f"Filtering cells with n_genes_by_counts < {min_genes_per_cell}: "
        f"keeping {cell_mask.sum()} / {adata.n_obs}"
    )
    adata = adata[cell_mask].copy()

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
        cell_counts = adata.obs.groupby(sample_column, observed=False).size()
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

    # --- Capture post-filter sample stats and compare ---
    print("=== Capturing post-filter sample statistics ===")
    post_filter_stats = _get_sample_stats(adata, sample_column=sample_column)

    sample_comparison = _compare_sample_stats(
        pre_filter_stats,
        post_filter_stats,
        output_dir=output_dir,
        prefix="sample_qc_comparison",
    )

    _print_sample_summary(sample_comparison, sample_column=sample_column)

    _plot_sample_comparison(
        sample_comparison,
        output_dir=plot_dir,
        prefix="sample_qc_comparison",
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
    if run_doublet_detection:
        print("=== Doublet detection is requested but not implemented in this stub ===")

    # --- Store raw counts in a layer before normalization ---
    if "counts" not in adata.layers:
        print("Storing raw counts in adata.layers['counts']")
        adata.layers["counts"] = adata.X.copy()

    # === HVG selection on raw counts, but do NOT subset genes ===
    print(f"=== HVG selection (Seurat v3, n_hvgs={n_hvgs}) ===")
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=n_hvgs,
        layer="counts",
        subset=False,
        inplace=True,
    )
    print("Number of HVGs:", adata.var["highly_variable"].sum())

    # === Normalization on ALL genes ===
    adata.X = adata.layers["counts"].copy()

    print("=== Normalization (10k counts/cell) and log1p on ALL genes ===")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata.raw = adata.copy()

    # === PCA using only HVGs ===
    print(f"=== PCA (n_pcs={n_pcs}) on HVGs only ===")
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(
        adata,
        n_comps=n_pcs,
        svd_solver="arpack",
        use_highly_variable=True,
    )
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
        n_pcs=None,
    )
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5, key_added="leiden")

    print("Stored UMAP in adata.obsm['X_umap'] and Leiden labels in adata.obs['leiden']")

    # --- UMAP plots ---
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
    MERGED_H5AD = "/dcl01/hongkai/data/data/hjiang/Data/long_covid/long_covid_test.h5ad"
    OUTPUT_DIR = "/dcs07/hongkai/data/harry/result/long_covid/QC"

    long_covid_qc_and_dr(
        h5ad_path=MERGED_H5AD,
        output_dir=OUTPUT_DIR,
        min_cells_per_gene=500,
        min_genes_per_cell=500,
        max_mito_fraction=0.20,
        n_hvgs=2000,
        n_pcs=20,
        harmony_max_iter=30,
        sample_column="sample",
        cell_type_column="cell_type",
        min_cells_per_sample=None,
        run_doublet_detection=False,
        save_filtered=True,
    )