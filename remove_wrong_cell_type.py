from pathlib import Path

import anndata as ad
import pandas as pd


ADATA_PATH = Path("/dcs07/hongkai/data/harry/result/long_covid/analysis/preprocess/adata_sample.h5ad")
CSV_PATH = Path("/dcs07/hongkai/data/harry/result/long_covid/analysis/clustering/AI_celltype_annotations.csv")
FILTERED_OUT = Path("/dcs07/hongkai/data/harry/result/long_covid/analysis/preprocess/adata_filtered.h5ad")

LEIDEN_COL = "leiden_0.25"
CELLTYPE_COL = "cell_type"
TARGET_RESOLUTION = 0.25
REMOVE_CELLTYPE = "Platelet"   # matches the new CSV


def normalize_cluster_label(x) -> str:
    """Convert cluster labels like 1, 1.0, '1' into canonical string form '1'."""
    if pd.isna(x):
        return None
    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return str(x).strip()


def main():
    print(f"Loading AnnData: {ADATA_PATH}")
    adata = ad.read_h5ad(ADATA_PATH)

    print(f"Loading annotations CSV: {CSV_PATH}")
    anno_df = pd.read_csv(CSV_PATH)

    required_cols = {"Resolution", "Cluster", "Identified Cell Type"}
    missing = required_cols - set(anno_df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # keep only resolution 0.25
    anno_df = anno_df.loc[anno_df["Resolution"] == TARGET_RESOLUTION].copy()
    if anno_df.empty:
        raise ValueError(f"No rows found in CSV for Resolution == {TARGET_RESOLUTION}")

    # build mapping: cluster number -> identified cell type
    anno_df["Cluster_norm"] = anno_df["Cluster"].map(normalize_cluster_label)
    cluster_to_celltype = dict(
        zip(anno_df["Cluster_norm"], anno_df["Identified Cell Type"].astype(str).str.strip())
    )

    print("\nResolution 0.25 mapping:")
    for k in sorted(cluster_to_celltype, key=lambda x: int(x)):
        print(f"  {k} -> {cluster_to_celltype[k]}")

    if LEIDEN_COL not in adata.obs.columns:
        raise KeyError(f"{LEIDEN_COL!r} not found in adata.obs")
    if CELLTYPE_COL not in adata.obs.columns:
        raise KeyError(f"{CELLTYPE_COL!r} not found in adata.obs")

    leiden_labels = adata.obs[LEIDEN_COL].map(normalize_cluster_label)

    unmapped = sorted(set(leiden_labels.dropna().unique()) - set(cluster_to_celltype.keys()), key=lambda x: int(x))
    if unmapped:
        raise ValueError(
            "Some leiden_0.25 cluster labels in AnnData are not present in the CSV mapping: "
            f"{unmapped}"
        )

    # overwrite cell_type using mapped leiden_0.25 labels
    new_celltypes = leiden_labels.map(cluster_to_celltype)
    adata.obs[CELLTYPE_COL] = pd.Categorical(new_celltypes)

    print("\nUpdated cell_type counts:")
    print(adata.obs[CELLTYPE_COL].value_counts(dropna=False))

    # overwrite original file in place
    print(f"\nOverwriting original AnnData in place: {ADATA_PATH}")
    adata.write_h5ad(ADATA_PATH)

    # remove Platelet cells and save filtered result
    keep_mask = adata.obs[CELLTYPE_COL].astype(str) != REMOVE_CELLTYPE
    adata_filtered = adata[keep_mask].copy()

    print(f"\nSaving filtered AnnData without {REMOVE_CELLTYPE!r}: {FILTERED_OUT}")
    adata_filtered.write_h5ad(FILTERED_OUT)

    print("\nDone.")
    print(f"Original overwritten: {ADATA_PATH}")
    print(f"Filtered saved to:   {FILTERED_OUT}")


if __name__ == "__main__":
    main()