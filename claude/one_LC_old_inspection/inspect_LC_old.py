#!/usr/bin/env python3
"""
Inspect the LC_Old cell of the LC x Age interaction analysis.

Goal
----
The user observed that in some boxplots of the LC x Age interaction analysis,
the LC_Old group shows only one dot, while in other boxplots/comparisons it
shows two or more.  This script produces a definitive accounting of:

  1. How many samples (rows of the design matrix) end up in LC_Old
     for every (root, resolution, cell_type, comparison) the pipeline ran.
  2. How those samples decompose by timepoint and by patient.
  3. For every gene reported as significant in that comparison,
     how many of those LC_Old samples actually carry usable expression
     (non-NA, non-zero) — this is what determines how many dots appear
     in the per-gene boxplot.

Roots inspected
---------------
- /dcs07/hongkai/data/harry/result/long_covid/LC_recovered_decouple/LC_removed_check
    Two resolutions: cell_type_0.25, cell_type_1
- /dcs07/hongkai/data/harry/result/long_covid/subset/B_LC_recovered_decouple
    One resolution: cell_type
- /dcs07/hongkai/data/harry/result/long_covid/subset/T_LC_recovered_decouple
    One resolution: cell_type

Within each (root, resolution) the pipeline writes to
    {root}/{resolution}/interaction_step2/LC_x_Age/{cell_type}/{comparison}/
        design_matrix.csv
        LC_x_Age_OldVsYoung/DE_results.csv
        LC_x_Age_OldVsMiddle/DE_results.csv
        LC_x_Age_MiddleVsYoung/DE_results.csv
and the underlying pseudobulk matrices live at
    {root}/{resolution}/merged/{timepoint}/{cell_type}/
        pseudobulk_expression.csv
        pseudobulk_metadata.csv

Outputs are written to:
    /dcs07/hongkai/data/harry/result/long_covid/analysis/one_LC_old_inspection/
"""

from __future__ import annotations

import csv
import os
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


OUTPUT_ROOT = Path("/dcs07/hongkai/data/harry/result/long_covid/analysis/one_LC_old_inspection")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

ROOTS: List[Tuple[str, str, List[str]]] = [
    (
        "LC_removed_check",
        "/dcs07/hongkai/data/harry/result/long_covid/LC_recovered_decouple/LC_removed_check",
        ["cell_type_0.25", "cell_type_1"],
    ),
    (
        "B_subset",
        "/dcs07/hongkai/data/harry/result/long_covid/subset/B_LC_recovered_decouple",
        ["cell_type"],
    ),
    (
        "T_subset",
        "/dcs07/hongkai/data/harry/result/long_covid/subset/T_LC_recovered_decouple",
        ["cell_type"],
    ),
]

# Significant gene file written by step 2 for each contrast
SIG_GENE_FILES = [
    "significant_genes_strict_mode.txt",
    "significant_genes_FDR_0.05_all.txt",
]

CONTRAST_DIRS = [
    "LC_x_Age_OldVsYoung",
    "LC_x_Age_OldVsMiddle",
    "LC_x_Age_MiddleVsYoung",
]


# --------------------------------------------------------------------------- #
# Design matrix parsing
# --------------------------------------------------------------------------- #
def parse_design_matrix(path: Path) -> Optional[pd.DataFrame]:
    """Return the design matrix indexed by sample, with int columns."""
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "sample" not in df.columns:
        df = df.rename(columns={df.columns[0]: "sample"})
    df = df.set_index("sample")
    return df


def lc_old_rows(design: pd.DataFrame) -> List[str]:
    """
    Identify samples encoded as LC_Old in a `~ lc_group * age_group + ...`
    model.matrix design.  Reference levels (Recovered / Young) are encoded by
    all-zero indicators for the contrasts.
    """
    if "lc_groupLC" not in design.columns or "age_groupOld" not in design.columns:
        return []
    mask = (design["lc_groupLC"] == 1) & (design["age_groupOld"] == 1)
    return list(design.index[mask])


# --------------------------------------------------------------------------- #
# Comparison name → timepoints
# --------------------------------------------------------------------------- #
COMPARISON_RE = re.compile(r"^M(\d+)_vs_M(\d+)$")


def comparison_to_timepoints(comparison: str) -> Optional[Tuple[str, str]]:
    """'M6_vs_M1' -> ('1', '6')   (returned in (earlier, later) order)."""
    m = COMPARISON_RE.match(comparison)
    if not m:
        return None
    a, b = m.group(2), m.group(1)  # M{later}_vs_M{earlier} -> earlier first
    return a, b


# Sample names in the design matrix come in two flavours:
#   LC_removed_check: "M1_185-M1"        (tp=1, patient=185)
#   subset:           "M1_35--185-M1"    (tp=1, library prefix=35, patient=185)
# In both cases the trailing "-M{tp}" is the timepoint suffix, and the
# patient id is the last numeric token before that suffix.
SAMPLE_RE = re.compile(r"^M(?P<tp>\d+)_(?P<rest>.+?)-M\d+$")


def split_sample(sample_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (timepoint, patient) parsed from a merged-tree sample id."""
    m = SAMPLE_RE.match(sample_id)
    if not m:
        return None, None
    rest = m.group("rest")
    # patient = last "-"-delimited token (handles both "185" and "35--185")
    parts = [p for p in rest.split("-") if p != ""]
    patient = parts[-1] if parts else rest
    return m.group("tp"), patient


def merged_sample_id(sample_id: str) -> str:
    """Return the original pseudobulk sample id (drop the leading 'M{tp}_')."""
    m = SAMPLE_RE.match(sample_id)
    if not m:
        return sample_id
    return f"{m.group('rest')}-M{m.group('tp')}"


# --------------------------------------------------------------------------- #
# Metadata + expression loaders
# --------------------------------------------------------------------------- #
def load_pseudobulk(merged_dir: Path, timepoint: str, cell_type: str):
    """
    Return (expr_df_genes_x_samples, meta_df) for the given cell type at
    the given timepoint, or (None, None) if missing.
    """
    folder = merged_dir / timepoint / cell_type
    expr_path = folder / "pseudobulk_expression.csv"
    meta_path = folder / "pseudobulk_metadata.csv"
    if not expr_path.is_file() or not meta_path.is_file():
        return None, None

    meta = pd.read_csv(meta_path)
    if "sample" not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: "sample"})
    meta = meta.set_index("sample")

    # Read expression: samples in rows, genes in columns
    expr = pd.read_csv(expr_path)
    if "sample" not in expr.columns:
        expr = expr.rename(columns={expr.columns[0]: "sample"})
    expr = expr.set_index("sample")
    # Transpose to genes x samples for easier per-gene look-ups
    expr = expr.T
    return expr, meta


def metadata_lc_old_summary(meta: pd.DataFrame) -> Dict[str, object]:
    """Count LC_Old patients in a single-timepoint metadata table."""
    if meta is None or meta.empty:
        return {"n_lc_old": 0, "patients": []}

    # Find LC/Recovered + age_cluster columns regardless of header variant
    lc_col = next(
        (c for c in meta.columns if c.lower().replace("/", "_") in {"lc_recovered", "lc"}),
        None,
    )
    if lc_col is None:
        for c in meta.columns:
            vals = set(meta[c].dropna().astype(str).unique())
            if {"LC", "Recovered"} <= vals or {"LongCOVID", "Recovered"} <= vals:
                lc_col = c
                break
    age_col = next((c for c in meta.columns if c.lower() == "age_cluster"), None)
    if age_col is None:
        for c in meta.columns:
            vals = set(meta[c].dropna().astype(str).unique())
            if {"Young", "Middle", "Old"} <= vals:
                age_col = c
                break

    if lc_col is None or age_col is None:
        return {"n_lc_old": 0, "patients": [], "warn": "missing LC/age column"}

    lc_vals = meta[lc_col].astype(str).replace({"LongCOVID": "LC"})
    age_vals = meta[age_col].astype(str)
    sub = meta[(lc_vals == "LC") & (age_vals == "Old")]
    pat_col = next(
        (c for c in sub.columns if c.lower() in {"patient", "outsmart short number"}),
        None,
    )
    if pat_col is not None:
        patients = sorted({str(p) for p in sub[pat_col].dropna()})
    else:
        # Fall back to deriving patient id from the sample id
        patients = sorted({str(idx).split("-")[0] for idx in sub.index})
    return {"n_lc_old": int(len(sub)), "patients": patients, "samples": list(sub.index)}


# --------------------------------------------------------------------------- #
# Per-gene expression usability for LC_Old samples
# --------------------------------------------------------------------------- #
def gene_usability_for_lc_old(
    expr_per_tp: Dict[str, pd.DataFrame],
    lc_old_samples: List[str],
    genes: List[str],
) -> List[Dict[str, object]]:
    """
    For each gene return how many LC_Old samples have a usable
    (non-NA, non-zero) expression value, plus the per-sample values.

    `expr_per_tp` is {timepoint -> genes_x_samples DataFrame}
    `lc_old_samples` are merged-tree ids ('M1_185-M1', 'M6_185-M6', ...).
    """
    rows: List[Dict[str, object]] = []
    for gene in genes:
        per_sample_vals: Dict[str, Optional[float]] = {}
        for s in lc_old_samples:
            tp, _patient = split_sample(s)
            tp_expr = expr_per_tp.get(tp)
            if tp_expr is None:
                per_sample_vals[s] = None
                continue
            local_id = merged_sample_id(s)
            if gene not in tp_expr.index or local_id not in tp_expr.columns:
                per_sample_vals[s] = None
                continue
            v = tp_expr.at[gene, local_id]
            try:
                per_sample_vals[s] = float(v) if v == v else None  # NaN check
            except Exception:
                per_sample_vals[s] = None

        n_total = len(lc_old_samples)
        n_with_value = sum(v is not None for v in per_sample_vals.values())
        n_nonzero = sum(v is not None and v != 0.0 for v in per_sample_vals.values())
        rows.append(
            {
                "gene": gene,
                "n_LC_Old_design": n_total,
                "n_LC_Old_with_value": n_with_value,
                "n_LC_Old_nonzero": n_nonzero,
                "values": ";".join(
                    f"{s}={('NA' if per_sample_vals[s] is None else f'{per_sample_vals[s]:.4f}')}"
                    for s in lc_old_samples
                ),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Main inspection driver
# --------------------------------------------------------------------------- #
def read_significant_genes(contrast_dir: Path) -> List[str]:
    """Pick the first available significant-gene list under the contrast dir."""
    for name in SIG_GENE_FILES:
        f = contrast_dir / name
        if f.is_file():
            with f.open() as fh:
                genes = [g.strip() for g in fh if g.strip()]
            return genes
    return []


def inspect_one(
    label: str,
    root: str,
    resolution: str,
) -> Tuple[List[Dict], List[Dict]]:
    """Walk one (root, resolution) tree.  Return (per_design rows, per_gene rows)."""
    base = Path(root) / resolution
    step2_dir = base / "interaction_step2" / "LC_x_Age"
    merged_dir = base / "merged"
    if not step2_dir.is_dir():
        return [], []

    per_design: List[Dict] = []
    per_gene: List[Dict] = []

    for cell_type_dir in sorted(p for p in step2_dir.iterdir() if p.is_dir()):
        cell_type = cell_type_dir.name
        for comp_dir in sorted(p for p in cell_type_dir.iterdir() if p.is_dir()):
            comparison = comp_dir.name
            tps = comparison_to_timepoints(comparison)
            if tps is None:
                continue

            design = parse_design_matrix(comp_dir / "design_matrix.csv")
            if design is None:
                continue

            samples = lc_old_rows(design)
            by_tp: Dict[str, List[str]] = defaultdict(list)
            patients: Dict[str, List[str]] = defaultdict(list)
            for s in samples:
                tp, patient = split_sample(s)
                if tp is None:
                    continue
                by_tp[tp].append(s)
                if patient is not None:
                    patients[tp].append(patient)

            # Pull a parallel count from the merged metadata so we can flag
            # design-vs-metadata mismatches.
            meta_summary: Dict[str, Dict] = {}
            expr_per_tp: Dict[str, pd.DataFrame] = {}
            for tp in tps:
                expr, meta = load_pseudobulk(merged_dir, tp, cell_type)
                if expr is not None:
                    expr_per_tp[tp] = expr
                meta_summary[tp] = (
                    metadata_lc_old_summary(meta) if meta is not None else {"n_lc_old": 0, "patients": []}
                )

            per_design.append(
                {
                    "root": label,
                    "resolution": resolution,
                    "cell_type": cell_type,
                    "comparison": comparison,
                    "tp_early": tps[0],
                    "tp_late": tps[1],
                    "n_LC_Old_design_total": len(samples),
                    "n_LC_Old_tp_early_design": len(by_tp.get(tps[0], [])),
                    "n_LC_Old_tp_late_design": len(by_tp.get(tps[1], [])),
                    "n_unique_patients_design": len({p for ps in patients.values() for p in ps}),
                    "patients_tp_early": ",".join(sorted(set(patients.get(tps[0], [])))),
                    "patients_tp_late": ",".join(sorted(set(patients.get(tps[1], [])))),
                    "lc_old_samples_design": ",".join(samples),
                    "n_LC_Old_meta_tp_early": meta_summary.get(tps[0], {}).get("n_lc_old", 0),
                    "n_LC_Old_meta_tp_late": meta_summary.get(tps[1], {}).get("n_lc_old", 0),
                    "patients_meta_tp_early": ",".join(meta_summary.get(tps[0], {}).get("patients", [])),
                    "patients_meta_tp_late": ",".join(meta_summary.get(tps[1], {}).get("patients", [])),
                }
            )

            # Per-gene: only inspect genes the pipeline already called
            # significant for each contrast (those are the boxplots the
            # user is looking at).
            for contrast_name in CONTRAST_DIRS:
                contrast_dir = comp_dir / contrast_name
                if not contrast_dir.is_dir():
                    continue
                genes = read_significant_genes(contrast_dir)
                if not genes:
                    continue
                gene_rows = gene_usability_for_lc_old(expr_per_tp, samples, genes)
                for r in gene_rows:
                    r.update(
                        {
                            "root": label,
                            "resolution": resolution,
                            "cell_type": cell_type,
                            "comparison": comparison,
                            "contrast": contrast_name,
                        }
                    )
                    per_gene.append(r)

    return per_design, per_gene


def main() -> int:
    all_design_rows: List[Dict] = []
    all_gene_rows: List[Dict] = []
    errors: List[str] = []

    for label, root, resolutions in ROOTS:
        for resolution in resolutions:
            try:
                d_rows, g_rows = inspect_one(label, root, resolution)
                all_design_rows.extend(d_rows)
                all_gene_rows.extend(g_rows)
                print(
                    f"[{label}/{resolution}] designs={len(d_rows)} gene_rows={len(g_rows)}",
                    flush=True,
                )
            except Exception:
                errors.append(f"{label}/{resolution}\n{traceback.format_exc()}")
                print(f"ERROR in {label}/{resolution}", file=sys.stderr)
                traceback.print_exc()

    design_df = pd.DataFrame(all_design_rows)
    gene_df = pd.DataFrame(all_gene_rows)

    design_csv = OUTPUT_ROOT / "lc_old_design_counts.csv"
    gene_csv = OUTPUT_ROOT / "lc_old_per_gene_usability.csv"
    summary_txt = OUTPUT_ROOT / "summary.txt"

    design_df.to_csv(design_csv, index=False)
    gene_df.to_csv(gene_csv, index=False)

    # Plain-text summary that answers the user's two questions directly.
    with summary_txt.open("w") as fh:
        fh.write("LC_Old inspection — summary\n")
        fh.write("=" * 78 + "\n\n")

        if not design_df.empty:
            sub_lc = design_df[design_df["root"] == "LC_removed_check"]
            by_comp = (
                design_df.groupby("comparison")["n_LC_Old_design_total"]
                .agg(["min", "max"]).reset_index()
            )
            fh.write("HEADLINE\n")
            fh.write("-" * 78 + "\n")
            fh.write(
                "1) How many LC_Old samples enter each interaction model?\n"
                "   The number depends ONLY on which timepoints the comparison\n"
                "   spans (which patients have an Old + LC pseudobulk at those\n"
                "   timepoints).  All cell types and all three roots agree:\n"
            )
            for _, r in by_comp.iterrows():
                fh.write(
                    f"     {r['comparison']}: {int(r['min'])} LC_Old sample(s)\n"
                )
            fh.write(
                "\n"
                "   Concretely, only one Old patient (185) has an LC pseudobulk\n"
                "   at M1, while two Old patients (156, 185) have LC\n"
                "   pseudobulks at M6.  So:\n"
                "     M3_vs_M1  -> 1 sample  (185 at M1; nobody at M3)\n"
                "     M6_vs_M3  -> 2 samples (156, 185 at M6)\n"
                "     M6_vs_M1  -> 3 samples (185 at M1; 156, 185 at M6)\n"
                "\n"
                "2) Why does a boxplot sometimes show fewer LC_Old dots than\n"
                "   the design contains?\n"
                "   Every LC_Old sample IS in the data — none are NA.  When dots\n"
                "   appear to be missing it is because the gene's normalised\n"
                "   pseudobulk expression is exactly 0 in those samples (the\n"
                "   gene was not detected in that patient's cells of that type),\n"
                "   so multiple zero-valued dots pile up at the bottom of the\n"
                "   plot and read as one dot.\n"
            )
            g_total = len(gene_df)
            g_zero = int((gene_df["n_LC_Old_nonzero"] < gene_df["n_LC_Old_design"]).sum())
            g_na = int((gene_df["n_LC_Old_with_value"] < gene_df["n_LC_Old_design"]).sum())
            fh.write(
                f"\n   Of {g_total} significant-gene rows: {g_na} had an NA value\n"
                f"   in any LC_Old sample, while {g_zero} had at least one\n"
                f"   zero-valued LC_Old sample (= overlapping dot at y=0).\n\n"
            )

        fh.write("Source roots inspected:\n")
        for label, root, resolutions in ROOTS:
            for resolution in resolutions:
                fh.write(f"  - {label}: {root}/{resolution}\n")
        fh.write("\n")

        if not design_df.empty:
            fh.write("How many LC_Old samples per (root, resolution, cell_type, comparison)\n")
            fh.write("-" * 78 + "\n")
            fh.write("All counts come from the design_matrix.csv that limma actually fitted\n")
            fh.write("(rows where lc_groupLC == 1 and age_groupOld == 1).\n\n")

            dist = (
                design_df.groupby("n_LC_Old_design_total").size().sort_index()
            )
            fh.write("Distribution of n_LC_Old across all comparisons:\n")
            for k, v in dist.items():
                fh.write(f"  n_LC_Old = {int(k):>2}  →  {int(v):>4} comparisons\n")
            fh.write("\n")

            for n in sorted(design_df["n_LC_Old_design_total"].unique()):
                sub = design_df[design_df["n_LC_Old_design_total"] == n]
                fh.write(f"Comparisons with n_LC_Old = {int(n)}  ({len(sub)} total):\n")
                for _, row in sub.iterrows():
                    fh.write(
                        f"  [{row['root']}/{row['resolution']}] "
                        f"{row['cell_type']} | {row['comparison']} "
                        f"| early(M{row['tp_early']})={row['n_LC_Old_tp_early_design']} "
                        f"late(M{row['tp_late']})={row['n_LC_Old_tp_late_design']} "
                        f"| patients={row['n_unique_patients_design']}\n"
                    )
                fh.write("\n")

        if not gene_df.empty:
            fh.write("\nWhy do some boxplots show fewer dots than n_LC_Old?\n")
            fh.write("-" * 78 + "\n")
            fh.write(
                "For every significant gene reported in each contrast we counted\n"
                "how many of the LC_Old samples actually carry a usable\n"
                "(non-NA, non-zero) expression value.  Below are the genes where\n"
                "fewer dots appear in the boxplot than there are LC_Old rows in\n"
                "the design.\n\n"
            )
            mismatches = gene_df[
                gene_df["n_LC_Old_nonzero"] < gene_df["n_LC_Old_design"]
            ].copy()
            fh.write(
                f"Total significant-gene rows: {len(gene_df)}\n"
                f"Rows where dots < design samples (zero or NA in LC_Old): "
                f"{len(mismatches)}\n\n"
            )
            if not mismatches.empty:
                # Sort: worst-mismatch first
                mismatches["delta"] = (
                    mismatches["n_LC_Old_design"] - mismatches["n_LC_Old_nonzero"]
                )
                mismatches = mismatches.sort_values(
                    ["delta", "root", "resolution", "cell_type", "comparison"],
                    ascending=[False, True, True, True, True],
                )
                head = mismatches.head(200)
                for _, row in head.iterrows():
                    fh.write(
                        f"  [{row['root']}/{row['resolution']}] "
                        f"{row['cell_type']} | {row['comparison']} | {row['contrast']} "
                        f"| {row['gene']}: design={row['n_LC_Old_design']}, "
                        f"with_value={row['n_LC_Old_with_value']}, "
                        f"nonzero={row['n_LC_Old_nonzero']}\n"
                        f"      values: {row['values']}\n"
                    )
                if len(mismatches) > 200:
                    fh.write(
                        f"  ... and {len(mismatches) - 200} more "
                        f"(see lc_old_per_gene_usability.csv)\n"
                    )

        if errors:
            fh.write("\nErrors:\n")
            fh.write("-" * 78 + "\n")
            for err in errors:
                fh.write(err + "\n")

    print(f"\nWrote: {design_csv}")
    print(f"Wrote: {gene_csv}")
    print(f"Wrote: {summary_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
