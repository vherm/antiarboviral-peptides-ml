#!/bin/bash
#SBATCH --job-name=antiarb-mem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
mkdir -p logs resultados
source ~/miniforge3/etc/profile.d/conda.sh
conda activate antiarboviral
srun python experiment_orchestrator.py \
  --matrix Matriz_Experimental_Maestra_AntiArboviral.xlsx \
  --manifest datasets_manifest.csv \
  --output resultados \
  --phase "F1 - Baseline factorial" \
  --priority Alta \
  --max-experiments 20 \
  --n-splits 5 \
  --n-repeats 5 \
  --search-mode none \
  --n-jobs 8 \
  --resume
