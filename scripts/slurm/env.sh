# Shared environment for LeWM Slurm jobs. Sourced by the .sbatch scripts.
REPO=/scratch/user/tahaismail424/klab/c2_wm_proj/adaptive-wm

module purge
module load uv/0.11.1
module load WebProxy/0000   # internet on compute nodes (W&B, HF)
source "$REPO/.venv/bin/activate"

export STABLEWM_HOME=/scratch/user/tahaismail424/.stable_wm
# stable-pretraining run dirs (metrics.csv, last.ckpt for requeue); default is ~/.cache which eats home quota
export SPT_CACHE_DIR=$STABLEWM_HOME/spt_runs
export HF_HOME=/scratch/user/tahaismail424/.cache/huggingface
export WANDB_DIR=$STABLEWM_HOME/wandb
mkdir -p "$WANDB_DIR"
# W&B: entity defaults to your personal account; override with WANDB_ENTITY=<team> at submit time
export WANDB_PROJECT=${WANDB_PROJECT:-lewm}
WANDB_ARGS="wandb.enabled=True wandb.config.entity=${WANDB_ENTITY:-null} wandb.config.project=$WANDB_PROJECT"
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q api.wandb.ai ~/.netrc 2>/dev/null; then
    # (not WANDB_MODE=offline: stable-pretraining's Manager crashes on a fresh offline run)
    echo "WARNING: not logged in to W&B (run 'wandb login' on a login node) -> W&B disabled, metrics.csv only"
    WANDB_ARGS="wandb.enabled=False"
fi
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

cd "$REPO"
echo "host=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES cpus=$SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
