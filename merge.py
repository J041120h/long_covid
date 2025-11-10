#!/usr/bin/env python3

import os
from pathlib import Path
from typing import List

import anndata as ad
import scanpy as sc


def _parse_sample_name(h5_file: Path) -> str:
    """
    Extract the sample name from a Cell Ranger HDF5 filename.

    Expected pattern (example):
        1--104-M1_sample_filtered_feature_bc_matrix.h5

    This function will return:
        "1--104-M1"
    """
    fname = h5_file.name

    # Most specific pattern first
    marker = "_sample_filtered_feature_bc_matrix"
    if marker in fname:
        return fname.split(marker)[0]

    # Fallback: strip extension
    return h5_file.stem


def merge_cellranger_h5_to_h5ad(input_dir: str, output_path: str) -> None:
    """
    Merge all Cell Ranger .h5 files in a directory into a single AnnData object and save as .h5ad.

    Each .h5 file is assumed to be a 10x Genomics / Cell Ranger matrix file
    (e.g. filtered_feature_bc_matrix.h5), not an .h5ad.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing .h5 files.
    output_path : str
        Path to the output .h5ad file (including filename).
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"{input_dir} is not a valid directory")

    # Collect all .h5 files in the directory (non-recursive)
    h5_files: List[Path] = sorted(input_path.glob("*.h5"))

    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in directory: {input_dir}")

    print(f"Found {len(h5_files)} .h5 file(s) in {input_dir}:")
    for f in h5_files:
        print(f"  - {f.name}")

    adatas: List[ad.AnnData] = []
    sample_names: List[str] = []

    for f in h5_files:
        sample_name = _parse_sample_name(f)  # e.g. "1--104-M1"
        sample_names.append(sample_name)

        print(f"\nReading Cell Ranger HDF5 file: {f}")
        # Use read_10x_h5 for Cell Ranger HDF5
        adata = sc.read_10x_h5(str(f))

        # Make gene/feature names unique
        adata.var_names_make_unique()

        # Annotate each cell with its sample information and source file
        adata.obs["sample"] = sample_name
        adata.obs["source_file"] = f.name

        print(f"  Loaded shape: {adata.n_obs} cells x {adata.n_vars} features "
              f"(sample = {sample_name})")

        adatas.append(adata)

    print("\nConcatenating AnnData objects across samples...")
    merged: ad.AnnData = ad.concat(
        adatas,
        join="outer",
        label="sample",
        keys=sample_names,   # one key per sample name
        index_unique="-",    # cell IDs become "<sample>-<barcode>"
    )

    print(f"Merged AnnData shape: {merged.n_obs} cells x {merged.n_vars} features")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting merged AnnData to: {output_path}")
    merged.write_h5ad(str(output_path))
    print("Done.")


if __name__ == "__main__":
    # >>> User-modifiable section <<<

    # Directory with files like:
    #   1--104-M1_sample_filtered_feature_bc_matrix.h5
    #   2--104-M3_sample_filtered_feature_bc_matrix.h5
    #   ...
    INPUT_DIR = "/dcs07/antar/data/cellranger/feature_matrices"

    # Output merged file
    OUTPUT_PATH = "/dcl01/hongkai/data/data/hjiang/Data/long_covid/long_covid.h5ad"

    merge_cellranger_h5_to_h5ad(INPUT_DIR, OUTPUT_PATH)
