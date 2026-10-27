#!/bin/bash
# Full LeWM training run on one A100 (paper setup: ~15M params, single GPU).
#
#   sbatch --job-name=lewm-shapeI --export=DATA=pusht_shapeI scripts/slurm/train.sbatch
#   sbatch --job-name=lewm-night  --export=DATA=pusht_night  scripts/slurm/train.sbatch
#   sbatch --job-name=lewm-heavy  --export=DATA=pusht_heavy  scripts/slurm/train.sbatch
#
# Checkpoints: $STABLEWM_HOME/checkpoints/lewm_<DATA>/weights_epoch_<k>.pt (override NAME=...)
#
# To use an A40 instead (often shorter queue): add
#   --partition=gpu-a40 --gres=gpu:a40:1
# on the sbatch command line (CLI flags override the #SBATCH lines below).
#
# Extra Hydra overrides can be appended after the script name, e.g.
#   sbatch --export=DATA=pusht_shapeI scripts/slurm/train.sbatch trainer.max_epochs=20
#
#SBATCH --job-name=lewm-train-pusht_shapeI
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --signal=SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=experiments/slurm_logs/%x-%j.out

source scripts/slurm/env.sh   # before set -u: Lmod's module() trips on unbound vars
set -euo pipefail

DATA=${DATA:-pusht_shapeI}
NAME=${NAME:-lewm_${DATA}}

# fixed subdir so a requeued/resubmitted job finds the same run dir
python train.py data=$DATA $WANDB_ARGS \
    output_model_name=$NAME subdir=$NAME \
    trainer.max_epochs=10 \
    num_workers=$((SLURM_CPUS_PER_TASK - 2)) \
    "$@"
