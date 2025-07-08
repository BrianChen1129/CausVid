#!/usr/bin/env python3
import subprocess
from pathlib import Path

# --------- fixed arguments ----------
CONFIG_PATH = "configs/wan_bidirectional_dmd_from_scratch.yaml"
PROMPT_FILE = "prompt.txt"
INFER_SCRIPT = "minimal_inference/bidirectional_inference.py"

# --------- paths you may want to adjust ----------
BASE_CHECKPOINT_DIR = Path(
    #"/mnt/sharefs/users/hao.zhang/DMD/wan_bidirectional_dmd_from_scratch/2025-06-20-08-17-06.607828_seed1024"
    "/mnt/sharefs/users/hao.zhang/DMD/wan_bidirectional_dmd_from_scratch/2025-07-08-05-28-30.814175_seed1024"
)
BASE_OUTPUT_DIR = Path("output_dmd/outputs_video_dmd_0-200step")

# --------- loop over checkpoints ----------
for step in range(0, 201, 20):          # 0, 200, …, 4800
    ckpt_name = f"checkpoint_model_{step:06d}"
    ckpt_path = BASE_CHECKPOINT_DIR / ckpt_name

    # make sure checkpoint exists
    if not ckpt_path.exists():
        print(f"Skip {ckpt_name}: not found")
        continue

    # output folder for this step
    out_dir = BASE_OUTPUT_DIR / str(step)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        INFER_SCRIPT,
        "--config_path", CONFIG_PATH,
        "--checkpoint_folder", str(ckpt_path),
        "--output_folder", str(out_dir),
        "--prompt_file_path", PROMPT_FILE,
    ]

    print(f"\n>>> Running step {step} <<<")
    print(" ".join(cmd))

    # run the inference
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[ERROR] Step {step} failed with exit code {result.returncode}")
        break  # or continue, depending on whether you want to stop on failures
