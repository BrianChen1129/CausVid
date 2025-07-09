#!/bin/bash

#SBATCH --job-name=dmd_8n
#SBATCH --partition=main
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=dmd_8n/dmd_8n_%j.out
#SBATCH --error=dmd_8n/dmd_8n_%j.err
#SBATCH --exclusive
#SBATCH --time=72:00:00

set -e -x

# Set distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export RDZV_ID=$SLURM_JOB_ID

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RDZV_ID: $RDZV_ID"

export WANDB_MODE=online
srun torchrun --nnodes 2 --nproc_per_node 8 --node_rank=$((SLURM_PROCID)) --rdzv_id $RDZV_ID --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT causvid/train_distillation.py --config_path local_scripts/wan_bidirectional_dmd_from_scratch.yaml 