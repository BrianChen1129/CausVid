export WANDB_MODE=offline
torchrun --nnodes 1 --nproc_per_node=1 --master-port 29501 causvid/train_distillation.py \
    --config_path  local_scripts/wan_bidirectional_dmd_from_scratch_debug.yaml 