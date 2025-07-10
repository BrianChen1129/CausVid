# download and extract video from the Mixkit dataset 
python distillation_data/download_mixkit.py  --local_dir /mnt/sharefs/users/hao.zhang/mixkit 

# convert the video to 480x832x81 
python distillation_data/process_mixkit.py --input_dir /mnt/sharefs/users/hao.zhang/mixkit  --output_dir /mnt/sharefs/users/hao.zhang/mixkit_480x832x81 --width 832   --height 480  --fps 16 

# precompute the vae latent 
torchrun --nproc_per_node 8 distillation_data/compute_vae_latent.py --input_video_folder /mnt/sharefs/users/hao.zhang/mixkit_480x832x81  --output_latent_folder /mnt/sharefs/users/hao.zhang/mixkit_480x832x81_vae_latent   --info_path sample_dataset/video_mixkit_6484_caption.json

# combined everything into a lmdb dataset 
python causvid/ode_data/create_lmdb_iterative.py   --data_path /mnt/sharefs/users/hao.zhang/mixkit_480x832x81_vae_latent  --lmdb_path /mnt/sharefs/users/hao.zhang/mixkit_480x832x81_vae_latent_lmdb

torchrun --nnodes 8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR causvid/train_distillation.py \
    --config_path  local_scripts/wan_bidirectional_dmd_from_scratch.yaml 

export WANDB_MODE=offline
torchrun --nnodes 1 --nproc_per_node=1 --master-port 29501 causvid/train_distillation.py \
    --config_path  local_scripts/wan_bidirectional_dmd_from_scratch_debug.yaml 

python minimal_inference/bidirectional_inference.py --config_path configs/wan_bidirectional_dmd_from_scratch.yaml --checkpoint_folder /mnt/sharefs/users/hao.zhang/DMD/wan_bidirectional_dmd_from_scratch/2025-06-20-08-17-06.607828_seed1024/checkpoint_model_004800  --output_folder outputs_video_dmd_64   --prompt_file_path prompts.txt 