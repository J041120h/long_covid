"""
Diagnose whether 228-M3 and 228-M6 failure is sequence-data-side or pipeline-side.

For each sample, read the RAW (pre-cell-call) feature_bc_matrix and characterize:
  - total non-zero barcodes
  - UMI distribution across barcodes (full distribution, not just top-N)
  - barcode-rank curve (knee plot)
  - top-1k, top-10k, top-100k barcodes — UMI sums and gene counts
  - mapped-read accounting against metrics_summary

If the raw matrix has a clean knee with thousands of high-UMI barcodes,
the data is fine and Cell Ranger's cell calling failed.

If the raw matrix has no knee (UMIs evenly spread or extremely sparse), the
library itself is broken — sequencing depth went into duplicates / ambient /
off-target reads, not real cells.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CR_ROOT = Path("/dcs07/antar/data/cellranger")
OUT_DIR = Path("/users/hjiang/GenoDistance/long_covid/claude/preprocess/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "1--104-M1":  "healthy reference",
    "19--228-M1": "same donor, healthy timepoint",
    "20--228-M3": "broken (suspected)",
    "21--228-M6": "broken (suspected)",
}


def load_raw_h5(sample: str):
    """Load CellRanger raw_feature_bc_matrix.h5 -> (csc matrix, feature_types)."""
    p = CR_ROOT / sample / "outs" / "multi" / "count" / "raw_feature_bc_matrix.h5"
    with h5py.File(p, "r") as f:
        m = f["matrix"]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = tuple(m["shape"][:])  # (n_features, n_barcodes)
        feat_types = m["features"]["feature_type"][:].astype(str)
    X = sp.csc_matrix((data, indices, indptr), shape=shape)
    return X, feat_types


def summarize_modality(X: sp.csc_matrix, feat_mask: np.ndarray, label: str):
    """Per-barcode UMI totals and detected-gene totals for one feature type."""
    sub = X[feat_mask, :]
    umi = np.asarray(sub.sum(axis=0)).ravel()
    genes = np.asarray((sub > 0).sum(axis=0)).ravel()
    nz = umi > 0
    return {
        "label": label,
        "n_features": int(feat_mask.sum()),
        "n_barcodes_total": int(X.shape[1]),
        "n_barcodes_nonzero": int(nz.sum()),
        "umi_sum_total": int(umi.sum()),
        "umi_sum_top10k": int(np.sort(umi)[-10000:].sum()) if nz.sum() else 0,
        "umi_sum_top1k":  int(np.sort(umi)[-1000:].sum())  if nz.sum() else 0,
        "median_umi_top1k":  float(np.median(np.sort(umi)[-1000:]))  if nz.sum() else 0,
        "median_umi_top10k": float(np.median(np.sort(umi)[-10000:])) if nz.sum() else 0,
        "median_genes_top1k":  float(np.median(np.sort(genes)[-1000:]))  if nz.sum() else 0,
        "median_genes_top10k": float(np.median(np.sort(genes)[-10000:])) if nz.sum() else 0,
        "umi_p99":  float(np.percentile(umi[nz], 99))  if nz.sum() else 0,
        "umi_p999": float(np.percentile(umi[nz], 99.9)) if nz.sum() else 0,
        "umi_max":  int(umi.max()) if nz.sum() else 0,
        "_umi_sorted": np.sort(umi)[::-1],  # for plotting
        "_genes_sorted_by_umi": genes[np.argsort(umi)[::-1]],
    }


def parse_metrics_summary(sample: str) -> dict:
    """Extract key metrics from cellranger metrics_summary.csv."""
    p = CR_ROOT / sample / "outs" / "per_sample_outs" / sample / "metrics_summary.csv"
    df = pd.read_csv(p)
    keep = {}
    for _, row in df.iterrows():
        cat = row.get("Category", "")
        lib = row.get("Library Type", "")
        gb = row.get("Grouped By", "")
        name = row.get("Metric Name", "")
        val = row.get("Metric Value", "")
        if cat == "Cells" and not isinstance(gb, str):
            keep[f"{lib} :: {name}"] = val
        elif cat == "Library" and gb == "Physical library ID" and lib == "Gene Expression":
            keep[f"GEX lib :: {name}"] = val
    return keep


def main():
    summary_rows = []
    raw_data_blob = {}

    for sample, role in SAMPLES.items():
        print(f"\n=== {sample}  ({role}) ===")
        X, ftypes = load_raw_h5(sample)
        gex_mask = ftypes == "Gene Expression"
        adt_mask = ftypes == "Antibody Capture"
        print(f"  raw matrix shape (features x barcodes): {X.shape}")
        print(f"  features by type: {dict(zip(*np.unique(ftypes, return_counts=True)))}")

        gex = summarize_modality(X, gex_mask, "GEX")
        adt = summarize_modality(X, adt_mask, "ADT")

        for s in (gex, adt):
            print(f"  [{s['label']}] nz_barcodes={s['n_barcodes_nonzero']:>9}  "
                  f"umi_sum={s['umi_sum_total']:>12,}  "
                  f"top1k_med_umi={s['median_umi_top1k']:.0f}  "
                  f"top10k_med_umi={s['median_umi_top10k']:.0f}  "
                  f"max_umi={s['umi_max']}")

        metrics = parse_metrics_summary(sample)

        summary_rows.append({
            "sample": sample,
            "role": role,
            "raw_n_barcodes_nonzero_GEX": gex["n_barcodes_nonzero"],
            "raw_n_barcodes_nonzero_ADT": adt["n_barcodes_nonzero"],
            "raw_GEX_total_UMI": gex["umi_sum_total"],
            "raw_GEX_top1k_med_UMI": gex["median_umi_top1k"],
            "raw_GEX_top10k_med_UMI": gex["median_umi_top10k"],
            "raw_GEX_top1k_med_genes": gex["median_genes_top1k"],
            "raw_GEX_top10k_med_genes": gex["median_genes_top10k"],
            "raw_GEX_max_UMI": gex["umi_max"],
            "raw_GEX_p99_UMI": gex["umi_p99"],
            "raw_GEX_p999_UMI": gex["umi_p999"],
            "metric_GEX_reads_in_library": metrics.get("GEX lib :: Number of reads", ""),
            "metric_GEX_valid_barcodes": metrics.get("GEX lib :: Valid barcodes", ""),
            "metric_GEX_cells_called": metrics.get("Gene Expression :: Cells", ""),
            "metric_GEX_median_UMI_per_cell": metrics.get("Gene Expression :: Median UMI counts per cell", ""),
            "metric_GEX_mean_reads_per_cell": metrics.get("Gene Expression :: Mean reads per cell", ""),
            "metric_GEX_confident_mapped_in_cells": metrics.get("Gene Expression :: Confidently mapped reads in cells", ""),
        })

        raw_data_blob[sample] = {
            "umi_sorted": gex["_umi_sorted"][:200000],  # top 200k for plotting
            "genes_sorted_by_umi": gex["_genes_sorted_by_umi"][:200000],
        }

    # write summary table
    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT_DIR / "raw_matrix_summary.tsv", sep="\t", index=False)
    print(f"\nwrote {OUT_DIR / 'raw_matrix_summary.tsv'}")
    print(df.to_string(index=False))

    # barcode-rank (knee) plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"1--104-M1": "C0", "19--228-M1": "C2", "20--228-M3": "C3", "21--228-M6": "C1"}
    for sample, blob in raw_data_blob.items():
        umi = blob["umi_sorted"]
        rank = np.arange(1, len(umi) + 1)
        nz = umi > 0
        axes[0].loglog(rank[nz], umi[nz], label=f"{sample} ({SAMPLES[sample]})", color=colors[sample], lw=1.2)
    axes[0].set_xlabel("Barcode rank (sorted by UMI desc)")
    axes[0].set_ylabel("Total UMI per barcode (GEX)")
    axes[0].set_title("Barcode-rank (knee) plot — GEX, raw matrix")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, which="both", alpha=0.3)

    # zoom on top 50k barcodes
    for sample, blob in raw_data_blob.items():
        umi = blob["umi_sorted"][:50000]
        axes[1].semilogy(np.arange(1, len(umi) + 1), np.maximum(umi, 1),
                         label=sample, color=colors[sample], lw=1.2)
    axes[1].set_xlabel("Barcode rank (top 50k)")
    axes[1].set_ylabel("Total UMI per barcode (GEX)")
    axes[1].set_title("Top 50,000 barcodes — does a knee exist?")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "barcode_rank.png", dpi=150)
    print(f"wrote {OUT_DIR / 'barcode_rank.png'}")

    # save raw arrays for further inspection
    np.savez_compressed(
        OUT_DIR / "barcode_rank_data.npz",
        **{f"{s}_umi": d["umi_sorted"] for s, d in raw_data_blob.items()},
        **{f"{s}_genes": d["genes_sorted_by_umi"] for s, d in raw_data_blob.items()},
    )
    print(f"wrote {OUT_DIR / 'barcode_rank_data.npz'}")


if __name__ == "__main__":
    main()
