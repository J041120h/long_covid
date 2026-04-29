#!/usr/bin/env python3
"""
Differential cell-type-proportion test (LC vs Recovered) for T / B cell
subset h5ad files.

Mirrors differential_celltype_proportion.py but:
  - Uses the single ``cell_type`` column (no resolution loop).
  - Reads B_clean_subclusterclean.h5ad / T_clean_subclusterclean.h5ad.
  - Writes results under
        /dcs07/hongkai/data/harry/result/long_covid/subset/{B,T}_differential_cell_type_proportion/

Output per subset:
    proportions_per_sample.csv
    ttest_results.csv
    differential_proportions.png
"""

import os
import sys
import math
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUBSET_CONFIGS = {
    "B": {
        "adata_path": "/dcs07/hongkai/data/harry/result/long_covid/subset/B_clean_subclusterclean.h5ad",
        "out_dir":    "/dcs07/hongkai/data/harry/result/long_covid/subset/B_differential_cell_type_proportion",
    },
    "T": {
        "adata_path": "/dcs07/hongkai/data/harry/result/long_covid/subset/T_clean_subclusterclean.h5ad",
        "out_dir":    "/dcs07/hongkai/data/harry/result/long_covid/subset/T_differential_cell_type_proportion",
    },
}

SAMPLE_COL   = "sample"
LC_COL       = "LC/Recovered"
CELLTYPE_COL = "manual_cell_type"
GROUPS       = ("LC", "Recovered")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def compute_sample_proportions(adata, sample_col, celltype_col):
    counts = pd.crosstab(adata.obs[sample_col], adata.obs[celltype_col])
    proportions = counts.div(counts.sum(axis=1), axis=0)
    return proportions


def sample_lc_labels(adata, sample_col, lc_col):
    return (
        adata.obs[[sample_col, lc_col]]
        .drop_duplicates(subset=[sample_col])
        .set_index(sample_col)[lc_col]
    )


def run_ttests(proportions, sample_group):
    rows = []
    lc_samples  = sample_group[sample_group == GROUPS[0]].index
    rec_samples = sample_group[sample_group == GROUPS[1]].index

    lc_samples  = proportions.index.intersection(lc_samples)
    rec_samples = proportions.index.intersection(rec_samples)

    for ct in proportions.columns:
        lc_vals  = proportions.loc[lc_samples,  ct].to_numpy()
        rec_vals = proportions.loc[rec_samples, ct].to_numpy()

        if len(lc_vals) < 2 or len(rec_vals) < 2:
            tstat, pval = np.nan, np.nan
        elif (np.all(lc_vals == lc_vals[0]) and np.all(rec_vals == rec_vals[0])
              and lc_vals[0] == rec_vals[0]):
            tstat, pval = 0.0, 1.0
        else:
            tstat, pval = stats.ttest_ind(lc_vals, rec_vals,
                                          equal_var=False, nan_policy="omit")

        rows.append({
            "cell_type":       ct,
            "n_LC":            len(lc_vals),
            "n_Recovered":     len(rec_vals),
            "mean_LC":         float(np.mean(lc_vals)) if len(lc_vals) else np.nan,
            "mean_Recovered":  float(np.mean(rec_vals)) if len(rec_vals) else np.nan,
            "std_LC":          float(np.std(lc_vals, ddof=1)) if len(lc_vals) > 1 else np.nan,
            "std_Recovered":   float(np.std(rec_vals, ddof=1)) if len(rec_vals) > 1 else np.nan,
            "mean_diff_LC_minus_Recovered":
                (float(np.mean(lc_vals)) - float(np.mean(rec_vals)))
                if (len(lc_vals) and len(rec_vals)) else np.nan,
            "t_stat":  float(tstat) if tstat is not None and not np.isnan(tstat) else np.nan,
            "p_value": float(pval)  if pval  is not None and not np.isnan(pval)  else np.nan,
        })

    results = pd.DataFrame(rows)
    results = results.sort_values("p_value", na_position="last").reset_index(drop=True)
    return results


def plot_proportions(proportions, sample_group, results, subset_label, out_path):
    ordered_cts = results["cell_type"].tolist()
    n = len(ordered_cts)
    ncols = min(4, n) if n > 0 else 1
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.6 * ncols, 3.2 * nrows),
                             squeeze=False)

    group_order = list(GROUPS)
    palette = {GROUPS[0]: "#d62728", GROUPS[1]: "#1f77b4"}

    df_long = proportions.join(sample_group.rename("group"), how="inner")

    for i, ct in enumerate(ordered_cts):
        ax = axes[i // ncols][i % ncols]
        plot_df = pd.DataFrame({
            "group":      df_long["group"].values,
            "proportion": df_long[ct].values,
        })
        plot_df = plot_df[plot_df["group"].isin(group_order)]

        sns.boxplot(data=plot_df, x="group", y="proportion",
                    order=group_order, palette=palette,
                    width=0.5, fliersize=0, ax=ax)
        sns.stripplot(data=plot_df, x="group", y="proportion",
                      order=group_order, color="black",
                      size=2.5, alpha=0.6, jitter=0.18, ax=ax)

        row = results.loc[results["cell_type"] == ct].iloc[0]
        pval = row["p_value"]
        pval_str = f"p = {pval:.2e}" if pd.notna(pval) else "p = NA"

        ax.set_title(f"{ct}\n{pval_str}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("proportion")
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Cell-type proportion: LC vs Recovered  ({subset_label})",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-subset run
# ---------------------------------------------------------------------------
def run_for_subset(subset_name):
    cfg = SUBSET_CONFIGS[subset_name]
    adata_path = cfg["adata_path"]
    out_dir    = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"Subset    : {subset_name}")
    print(f"AnnData in: {adata_path}")
    print(f"Output dir: {out_dir}")
    print("=" * 60)

    adata = sc.read_h5ad(adata_path)
    print(f"  {adata.n_obs:,} cells  x  {adata.n_vars:,} genes")

    # Only metadata is needed — no need to touch X / counts layer.
    adata.obs[CELLTYPE_COL] = adata.obs[CELLTYPE_COL].astype(str)

    sample_group = sample_lc_labels(adata, SAMPLE_COL, LC_COL).astype(str)
    sample_group = sample_group[sample_group.isin(GROUPS)]
    print(f"  Samples: total={len(sample_group)}, "
          f"LC={(sample_group == GROUPS[0]).sum()}, "
          f"Recovered={(sample_group == GROUPS[1]).sum()}")

    proportions = compute_sample_proportions(adata, SAMPLE_COL, CELLTYPE_COL)
    proportions = proportions.loc[proportions.index.intersection(sample_group.index)]
    print(f"  proportions matrix: {proportions.shape[0]} samples x "
          f"{proportions.shape[1]} cell types")

    prop_out = proportions.copy()
    prop_out.insert(0, "group", sample_group.reindex(prop_out.index).values)
    prop_out.to_csv(os.path.join(out_dir, "proportions_per_sample.csv"))

    results = run_ttests(proportions, sample_group)
    results.to_csv(os.path.join(out_dir, "ttest_results.csv"), index=False)
    print(f"  t-test results (top 5 by p-value):")
    print(results.head(5).to_string(index=False))

    fig_path = os.path.join(out_dir, "differential_proportions.png")
    plot_proportions(proportions, sample_group, results,
                     subset_label=f"{subset_name} subset", out_path=fig_path)
    print(f"  Saved figure: {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=["B", "T", "both"], default="both")
    args = parser.parse_args()

    subsets = ["B", "T"] if args.subset == "both" else [args.subset]
    for s in subsets:
        run_for_subset(s)
        print()


if __name__ == "__main__":
    main()
