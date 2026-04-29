# T / B cell subset differential-analysis pipeline

Adapted from the pipeline in `/users/hjiang/GenoDistance/long_covid/` and
`/users/hjiang/r/` to run on the sub-clustered T and B cell AnnData files:

- `/dcs07/hongkai/data/harry/result/long_covid/subset/B_clean_subclusterclean.h5ad`
- `/dcs07/hongkai/data/harry/result/long_covid/subset/T_clean_subclusterclean.h5ad`

Raw counts are read from `adata.layers['counts']`; cell type labels are in
`adata.obs['cell_type']` (single resolution, no `cell_type_0.25` / `cell_type_1`
split).

## Stages

1. **Pseudobulk** (`run_pseudobulk_TB_subset.py`) — per (LC/Recovered × month ×
   cell_type) pseudobulk aggregation; builds the `LC/`, `Recovered/` and
   `merged/` trees + cell-type proportion CSVs.
2. **Differential cell-type proportion test**
   (`differential_celltype_proportion_TB_subset.py`) — Welch's t-test LC vs
   Recovered on per-sample proportions, plus a boxplot figure per subset.
3. **Step 1 interaction** (`run_step1_interaction_TB_subset.R`) — LC × Age
   interaction, per timepoint, on gene expression AND on cell-type proportions.
4. **Step 2 interaction** (`run_step2_interaction_TB_subset.R`) — LC × Month
   and LC × Age pooled across timepoint pairs, on gene expression AND
   cell-type proportions.

## Output locations (none overlap with the existing `sample_pseudobulk_differential_gene/` tree)

```
/dcs07/hongkai/data/harry/result/long_covid/subset/
  B_LC_recovered_decouple/cell_type/
    LC/{1,3,6}/<celltype>/pseudobulk{.h5ad, _expression.csv, _metadata.csv}
    Recovered/{1,3,6}/<celltype>/...
    merged/
      {1,3,6}/<celltype>/pseudobulk_expression.csv  (LC + Recovered row-bound)
      cell_type_proportions_all.csv
      cell_type_proportions_summary.csv
    interaction_step1/   # LC x Age per timepoint
    interaction_step2/   # LC x Month + LC x Age pooled
  B_differential_cell_type_proportion/
    ttest_results.csv, proportions_per_sample.csv, differential_proportions.png
  T_LC_recovered_decouple/... (same structure)
  T_differential_cell_type_proportion/...
```

## How to run

```bash
# Submit as a batch job (recommended)
sbatch /users/hjiang/GenoDistance/long_covid/claude/run_all.sbatch

# or interactively, after grabbing a compute node:
srun --pty --mem=100G --cpus-per-task=8 --time=24:00:00 bash
bash /users/hjiang/GenoDistance/long_covid/claude/run_all.sh
```

Logs:
- Slurm: `logs/slurm_<jobid>.{out,err}`
- Per-stage: `logs/{pseudobulk_B,pseudobulk_T,differential_proportion,step1_interaction,step2_interaction}.log`

The driver is idempotent: each stage checks for a sentinel output file and is
skipped if already complete. To force a re-run of everything, run with
`FORCE=1` in the environment:

```bash
FORCE=1 bash /users/hjiang/GenoDistance/long_covid/claude/run_all.sh
```

## Environment notes

- Python: `/users/hjiang/.conda/envs/hongkai/bin/python` is hard-coded.
- R: `conda_R/4.4` on JHPCE. The driver prepends the R bin dir directly on
  PATH because `module load` isn't reliable inside `sbatch`.
- R library path: the wrappers override `setup_pipeline_env()` to set
  `.libPaths()` to `c("~/R/4.4", "~/R_envs/differential_gene")` — the default
  env in the original code (`~/R_envs/differential_gene`) is missing a
  compatible `gtable` / `pheatmap` / `RColorBrewer` / `ggrepel` / `tidyr`,
  which the user-level `~/R/4.4` library supplies.
- GPU: disabled (`--no-gpu`). The `rapids_singlecell` stack hits
  `cudaErrorInsufficientDriver` on the compute partitions we have access to;
  the wrapper short-circuits its import so `pseudobulk.py` falls back cleanly
  to the CPU path. CPU pseudobulk + normalization of ~25k cells is ~seconds.

## File manifest

```
run_pseudobulk_TB_subset.py              — Stage 1 (B + T)
differential_celltype_proportion_TB_subset.py — Stage 2 (B + T)
run_step1_interaction_TB_subset.R        — Stage 3 (B + T)
run_step2_interaction_TB_subset.R        — Stage 4 (B + T)
run_all.sh                                — Sequential driver
run_all.sbatch                            — Slurm wrapper
README.md                                 — this file
```
