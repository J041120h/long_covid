#!/bin/bash
#SBATCH --job-name=lc_article
#SBATCH --output=/users/hjiang/GenoDistance/long_covid/article/logs/run_%j.out
#SBATCH --error=/users/hjiang/GenoDistance/long_covid/article/logs/run_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --partition=shared

mkdir -p /users/hjiang/GenoDistance/long_covid/article/logs

module load conda_R/4.4.x

echo "=== Long COVID Article Analysis ==="
echo "Start: $(date)"
Rscript /users/hjiang/GenoDistance/long_covid/article/code/run_analysis.R
echo "End:   $(date)"
