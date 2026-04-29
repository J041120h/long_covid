#!/bin/bash
# =============================================================================
# Sequential driver for the T / B subset differential-analysis pipeline.
#
# Steps (run in order):
#   1. Pseudobulk per (LC_group x month x cell_type) for each subset
#   2. Differential cell-type proportion test (LC vs Recovered) for each subset
#   3. Step 1 interaction analysis (LC x Age per timepoint)
#   4. Step 2 interaction analysis (LC x Month + LC x Age pooled)
#
# Writes everything under:
#   /dcs07/hongkai/data/harry/result/long_covid/subset/{B,T}_LC_recovered_decouple/
#   /dcs07/hongkai/data/harry/result/long_covid/subset/{B,T}_differential_cell_type_proportion/
#
# The existing sample_pseudobulk_differential_gene/ analysis is NOT touched.
#
# Each stage is skipped if its output already exists. To force a re-run of a
# stage, delete its output dir or set FORCE=1 in the environment.
#
# Run via sbatch:
#   sbatch /users/hjiang/GenoDistance/long_covid/claude/run_all.sbatch
# =============================================================================

set -euo pipefail

CLAUDE_DIR="/users/hjiang/GenoDistance/long_covid/claude"
PY="/users/hjiang/.conda/envs/hongkai/bin/python"
LOG_DIR="${CLAUDE_DIR}/logs"
mkdir -p "${LOG_DIR}"

SUBSET_ROOT="/dcs07/hongkai/data/harry/result/long_covid/subset"

FORCE="${FORCE:-0}"

# R (4.4) is provided by the conda_R module on JHPCE, but `module load`
# doesn't always update PATH in non-interactive sbatch contexts. Prepend the
# conda_R bin directories directly so we don't depend on lmod eval tricks.
if ! command -v Rscript >/dev/null 2>&1; then
  export PATH="/jhpce/shared/community/core/conda_R/4.4/R/bin:/jhpce/shared/community/core/conda_R/4.4/texlive/bin/x86_64-linux:${PATH}"
  export LD_LIBRARY_PATH="/jhpce/shared/community/core/conda_R/4.4/lib:${LD_LIBRARY_PATH:-}"
fi
command -v Rscript >/dev/null 2>&1 || {
  echo "ERROR: Rscript not found on PATH even after adding conda_R bin dir." >&2
  exit 1
}
echo "Using: $(command -v Rscript)"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
banner() {
  echo
  echo "############################################################"
  echo "# $(ts)  $*"
  echo "############################################################"
}

# Returns 0 (true) if pseudobulk for subset is already complete
pseudobulk_done() {
  local subset=$1
  local root="${SUBSET_ROOT}/${subset}_LC_recovered_decouple/cell_type"
  [[ -f "${root}/merged/cell_type_proportions_all.csv" ]]
}

proportion_test_done() {
  local subset=$1
  [[ -f "${SUBSET_ROOT}/${subset}_differential_cell_type_proportion/ttest_results.csv" ]]
}

step1_done() {
  local subset=$1
  [[ -f "${SUBSET_ROOT}/${subset}_LC_recovered_decouple/cell_type/interaction_step1/summary/complete_summary.txt" ]]
}

step2_done() {
  local subset=$1
  [[ -f "${SUBSET_ROOT}/${subset}_LC_recovered_decouple/cell_type/interaction_step2/summary/complete_summary.txt" ]]
}

# =============================================================================
# Step 1: Pseudobulk per subset
# =============================================================================
for subset in B T; do
  if [[ "${FORCE}" != "1" ]] && pseudobulk_done "${subset}"; then
    banner "Step 1 / 4 - Pseudobulk (${subset} subset) - SKIP (already done)"
  else
    banner "Step 1 / 4 - Pseudobulk (${subset} subset)"
    "${PY}" -u "${CLAUDE_DIR}/run_pseudobulk_TB_subset.py" --subset "${subset}" \
        2>&1 | tee "${LOG_DIR}/pseudobulk_${subset}.log"
  fi
done

# =============================================================================
# Step 2: Differential cell-type proportion test
# =============================================================================
if [[ "${FORCE}" != "1" ]] && proportion_test_done B && proportion_test_done T; then
  banner "Step 2 / 4 - Differential cell-type proportion - SKIP (already done)"
else
  banner "Step 2 / 4 - Differential cell-type proportion (B + T)"
  "${PY}" -u "${CLAUDE_DIR}/differential_celltype_proportion_TB_subset.py" --subset both \
      2>&1 | tee "${LOG_DIR}/differential_proportion.log"
fi

# =============================================================================
# Step 3: Step 1 interaction analysis (LC x Age per timepoint)
# =============================================================================
step1_subsets=()
for subset in B T; do
  if [[ "${FORCE}" != "1" ]] && step1_done "${subset}"; then
    banner "Step 3 / 4 - Step 1 interaction (${subset} subset) - SKIP (already done)"
  else
    step1_subsets+=("${subset}")
  fi
done
if [[ ${#step1_subsets[@]} -gt 0 ]]; then
  banner "Step 3 / 4 - Step 1 interaction (LC x Age per timepoint) - subsets: ${step1_subsets[*]}"
  Rscript "${CLAUDE_DIR}/run_step1_interaction_TB_subset.R" "${step1_subsets[@]}" \
      2>&1 | tee "${LOG_DIR}/step1_interaction.log"
fi

# =============================================================================
# Step 4: Step 2 interaction analysis (LC x Month + LC x Age pooled)
# =============================================================================
step2_subsets=()
for subset in B T; do
  if [[ "${FORCE}" != "1" ]] && step2_done "${subset}"; then
    banner "Step 4 / 4 - Step 2 interaction (${subset} subset) - SKIP (already done)"
  else
    step2_subsets+=("${subset}")
  fi
done
if [[ ${#step2_subsets[@]} -gt 0 ]]; then
  banner "Step 4 / 4 - Step 2 interaction (LC x Month pooled) - subsets: ${step2_subsets[*]}"
  Rscript "${CLAUDE_DIR}/run_step2_interaction_TB_subset.R" "${step2_subsets[@]}" \
      2>&1 | tee "${LOG_DIR}/step2_interaction.log"
fi

banner "DONE. Logs under ${LOG_DIR}"
