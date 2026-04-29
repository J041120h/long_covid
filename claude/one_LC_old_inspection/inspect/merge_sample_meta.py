#!/usr/bin/env python3
"""
Merge sample_meta.csv into the cleaned T and B anndata files.

Only rewrites the /obs group via anndata.experimental.{read_elem,write_elem},
so X/layers/obsm/obsp/raw are never loaded into memory. Safe for the 3.8 GB T
file that previously OOM-killed a full ad.read_h5ad.

Join key: obs['sample'] with the batch prefix stripped.
    '1--104-M1' -> '104-M1' matches sample_meta 'sample' column.
"""

import os
import re
import sys

import anndata as ad
import h5py
import pandas as pd
from anndata.experimental import read_elem, write_elem

SAMPLE_META = "/dcl01/hongkai/data/data/hjiang/Data/long_covid/sample_meta.csv"
FILES = {
    "T": "/dcs07/hongkai/data/harry/result/long_covid/subset/T_clean_subclusterclean.h5ad",
    "B": "/dcs07/hongkai/data/harry/result/long_covid/subset/B_clean_subclusterclean.h5ad",
}

BATCH_PREFIX_RE = re.compile(r"^\d+--(.+)$")


def strip_batch_prefix(s: str) -> str:
    m = BATCH_PREFIX_RE.match(str(s))
    return m.group(1) if m else str(s)


def merge_one(lineage: str, path: str, meta: pd.DataFrame) -> None:
    print(f"\n========== {lineage}: {path} ==========", flush=True)

    # Read only /obs
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
    print(f"obs shape: {obs.shape}", flush=True)
    print(f"obs columns before: {list(obs.columns)}", flush=True)

    if "sample" not in obs.columns:
        sys.exit(f"ERROR: obs has no 'sample' column; got {list(obs.columns)}")

    obs_sample_raw = obs["sample"].astype(str)
    join_key = obs_sample_raw.map(strip_batch_prefix)
    uniq_raw = sorted(obs_sample_raw.unique())
    uniq_key = sorted(join_key.unique())
    print(f"unique obs['sample']: {len(uniq_raw)} (e.g. {uniq_raw[:3]})",
          flush=True)
    print(f"unique join keys    : {len(uniq_key)} (e.g. {uniq_key[:3]})",
          flush=True)

    meta_keys = set(meta.index)
    missing = sorted(set(uniq_key) - meta_keys)
    extra = sorted(meta_keys - set(uniq_key))
    if missing:
        print(f"WARNING: {len(missing)} samples in obs not in sample_meta: "
              f"{missing}", flush=True)
    if extra:
        print(f"INFO: {len(extra)} samples in sample_meta not in lineage: "
              f"{extra}", flush=True)

    for col in meta.columns:
        if col in obs.columns:
            print(f"  overwriting existing obs[{col!r}]", flush=True)
            del obs[col]

    merged = meta.reindex(join_key.values)
    merged.index = obs.index
    for col in meta.columns:
        obs[col] = merged[col].values

    n_missing = sum(int(k not in meta_keys) for k in join_key)
    print(f"\nRows with no metadata match: {n_missing} / {len(obs)}",
          flush=True)
    print(f"obs columns after:  {list(obs.columns)}", flush=True)
    print("\nSanity check (first 3 rows of obs, new columns only):",
          flush=True)
    print(obs[list(meta.columns)].head(3).to_string(), flush=True)

    # Write /obs back in place. write_elem overwrites cleanly.
    print(f"\nWriting /obs back into {path} ...", flush=True)
    with h5py.File(path, "r+") as f:
        if "obs" in f:
            del f["obs"]
        write_elem(f, "obs", obs)
    print(f"Done.", flush=True)


def main() -> None:
    meta = pd.read_csv(SAMPLE_META)
    print(f"sample_meta: {meta.shape}, cols: {list(meta.columns)}", flush=True)
    if "sample" not in meta.columns:
        sys.exit("ERROR: sample_meta.csv has no 'sample' column")
    if meta["sample"].duplicated().any():
        sys.exit("ERROR: sample_meta.csv has duplicate 'sample' keys")
    meta = meta.set_index("sample")

    for lineage, path in FILES.items():
        merge_one(lineage, path, meta)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
