#!/bin/bash

#SBATCH --job-name=test01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
source ~/envs/ml_antipeptides/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[INFO] OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[INFO] OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "[INFO] MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "[INFO] BLIS_NUM_THREADS=$BLIS_NUM_THREADS"
echo "[INFO] NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"

# General data
SPLIT_DIR="./Splits"
SPLIT_METHOD="lorder"
RESULTS_DIR="./Results"
DATASET="Dataset_Random_2"

# Experiments
# 1 - ESM-2 + SVM + none
REP_NAME="ESM-2"
DATASET_NAME="${DATASET}_${REP_NAME}"
TRAIN_FILE="${SPLIT_DIR}/${DATASET_NAME}_train_${SPLIT_METHOD}.csv"
TEST_FILE="${SPLIT_DIR}/${DATASET_NAME}_test_${SPLIT_METHOD}.csv"

echo "[INFO] Train file : $TRAIN_FILE"
echo "[INFO] Test file  : $TEST_FILE"
echo "[INFO] Split mode : $SPLIT_METHOD"

echo "[INFO] Running ML pipeline (1)..."
srun python run_experiments.py \
  --train_file "$TRAIN_FILE" \
  --test_file "$TEST_FILE" \
  --representation "$REP_NAME" \
  --algorithms SVM \
  --feature_selection_methods none \
  --search_mode grid \
  --cv 5 \
  --interpretability permutation \
  --top_features 20 \
  --permutation_repeats 20 \
  --permutation_scoring average_precision \
  --results_dir "$RESULTS_DIR"

