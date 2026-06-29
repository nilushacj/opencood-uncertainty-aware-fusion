#!/bin/bash
#SBATCH --job-name=feaco_eval_parallel
#SBATCH --time=60:00:00
#SBATCH --partition=gpu-v100-32g,gpu-a100-80g
#SBATCH --gpus=1 
#SBATCH --gres=min-vram:32g
#SBATCH --mem=350G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

mkdir -p slurm_logs
mkdir -p parallel_inference_logs

module load mamba
source activate opencood

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

MANIFEST="opencood/pretrained/feaco/generated_manifest_fusion_test_split.csv"

# python opencood/tools/inference_manifest_parallel.py \
#     --manifest "${MANIFEST}" \
#     --workers 4 \
#     --fusion_method intermediate \
#     --log_dir parallel_inference_logs \
#     --skip_existing

python opencood/tools/inference_manifest_parallel.py \
    --manifest "${MANIFEST}" \
    --workers 4 \
    --seed "${SEED}" \
    --fusion_method intermediate \
    --log_dir "parallel_inference_logs_seed${SEED}" \
    --skip_existing

# TO RUN: sbatch --export=SEED=42 run_manifest_parallel.slurm