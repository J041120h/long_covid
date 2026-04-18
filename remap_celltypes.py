import os
import scanpy as sc
import pandas as pd
from collections import defaultdict

ADATA_PATH = "/dcs07/hongkai/data/harry/result/long_covid/analysis/preprocess/adata_sample.h5ad"
CSV_PATH = "/dcs07/hongkai/data/harry/result/long_covid/analysis/clustering/AI_celltype_annotations.csv"


def build_mapping(df, resolution):
    sub = df[df["Resolution"].astype(float) == float(resolution)].copy()
    total = defaultdict(int)
    for ct in sub["Identified Cell Type"]:
        total[ct] += 1
    running = defaultdict(int)
    mapping = {}
    for _, row in sub.iterrows():
        cluster = str(int(row["Cluster"]))
        ct = row["Identified Cell Type"]
        if total[ct] > 1:
            running[ct] += 1
            mapping[cluster] = f"{ct} ({running[ct]})"
        else:
            mapping[cluster] = ct
    return mapping


df = pd.read_csv(CSV_PATH)
map_025 = build_mapping(df, 0.25)
map_1 = build_mapping(df, 1)

print("Resolution 0.25 mapping:")
for k, v in sorted(map_025.items(), key=lambda x: int(x[0])):
    print(f"  {k} -> {v}")
print("Resolution 1 mapping:")
for k, v in sorted(map_1.items(), key=lambda x: int(x[0])):
    print(f"  {k} -> {v}")

print(f"\nLoading AnnData from: {ADATA_PATH}")
adata = sc.read_h5ad(ADATA_PATH)
print(f"Loaded: {adata.shape}")

leiden_025 = adata.obs["leiden_0.25"].astype(str)
leiden_1 = adata.obs["leiden_1"].astype(str)

missing_025 = set(leiden_025.unique()) - set(map_025.keys())
missing_1 = set(leiden_1.unique()) - set(map_1.keys())
if missing_025:
    raise ValueError(f"Unmapped leiden_0.25 clusters: {missing_025}")
if missing_1:
    raise ValueError(f"Unmapped leiden_1 clusters: {missing_1}")

new_025 = leiden_025.map(map_025)
new_1 = leiden_1.map(map_1)

adata.obs["cell_type_0.25"] = pd.Categorical(new_025.values)
adata.obs["cell_type_1"] = pd.Categorical(new_1.values)

print("\nNew cell_type_0.25 value counts:")
print(adata.obs["cell_type_0.25"].value_counts())
print("\nNew cell_type_1 value counts:")
print(adata.obs["cell_type_1"].value_counts())

tmp_path = ADATA_PATH + ".tmp"
print(f"\nWriting to temp: {tmp_path}")
adata.write_h5ad(tmp_path)
print(f"Atomic replace -> {ADATA_PATH}")
os.replace(tmp_path, ADATA_PATH)
print("Done.")
