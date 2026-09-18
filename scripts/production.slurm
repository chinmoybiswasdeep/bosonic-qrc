#!/bin/bash
#SBATCH --job-name=dv-capacity
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
set -euo pipefail
bosonic-qrc-dv-static-capacity configs/production/static_ipc.yaml \
  --output "results/shards/${SLURM_ARRAY_TASK_ID}" \
  --chunk-index "${SLURM_ARRAY_TASK_ID}" --chunk-count 10

