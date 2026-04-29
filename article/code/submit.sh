#!/bin/bash
#SBATCH --job-name=lc_article
#SBATCH --output=/users/hjiang/GenoDistance/long_covid/article/logs/run_%j.out
#SBATCH --error=/users/hjiang/GenoDistance/long_covid/article/logs/run_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --partition=shared

# Pass --cohort=female (default), --cohort=all, or --cohort=male via:
#   sbatch submit.sh --cohort=all
COHORT_ARG="${1:---cohort=female}"

mkdir -p /users/hjiang/GenoDistance/long_covid/article/logs

module load conda_R/4.4.x

echo "=== Long COVID Article Analysis ==="
echo "Cohort flag: $COHORT_ARG"
echo "Start: $(date)"
Rscript /users/hjiang/GenoDistance/long_covid/article/code/run_analysis.R "$COHORT_ARG"
echo "End:   $(date)"
