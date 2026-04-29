#!/bin/bash
#SBATCH --job-name=TB_subset_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --output=/users/hjiang/GenoDistance/long_covid/claude/logs/slurm_%j.out
#SBATCH --error=/users/hjiang/GenoDistance/long_covid/claude/logs/slurm_%j.err
# -----------------------------------------------------------------------------
# Submit: sbatch /users/hjiang/GenoDistance/long_covid/claude/run_all.sbatch
# -----------------------------------------------------------------------------

set -euo pipefail
mkdir -p /users/hjiang/GenoDistance/long_covid/claude/logs

bash /users/hjiang/GenoDistance/long_covid/claude/run_all.sh
