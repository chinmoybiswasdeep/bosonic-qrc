#!/bin/bash
#SBATCH --job-name=cv-ipc
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
set -euo pipefail
bosonic-qrc-cv-ipc configs/production/ipc.yaml \
  --output "results/shards/${SLURM_ARRAY_TASK_ID}" \
  --chunk-index "${SLURM_ARRAY_TASK_ID}" --chunk-count 10

