#!/bin/bash
# Planning evaluation (CEM, 50 episodes) on a small GPU.
# POLICY is relative to $STABLEWM_HOME/checkpoints/, a HF repo id, or "random".
#
#   sbatch --export=CONFIG=tworoom,POLICY=lewm_tworoom/weights_epoch_10.pt scripts/slurm/eval.sbatch
#   sbatch --export=CONFIG=cube,POLICY=lewm_ogb/weights_epoch_10.pt       scripts/slurm/eval.sbatch
#   sbatch --export=CONFIG=tworoom,POLICY=quentinll/lewm-tworooms          scripts/slurm/eval.sbatch   # released ckpt
#   sbatch --export=CONFIG=tworoom,POLICY=random                           scripts/slurm/eval.sbatch   # baseline
#
# Results (metrics txt + videos) land next to the policy: $STABLEWM_HOME/<policy dir>/
#
#SBATCH --job-name=lewm-eval-pusht_shapeI-val-random
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=012:00:00
#SBATCH --output=experiments/slurm_logs/%x-%j.out

source scripts/slurm/env.sh   # before set -u: Lmod's module() trips on unbound vars
set -euo pipefail

CONFIG=${CONFIG:-pusht_shapeI_val}
POLICY=${POLICY:-random}

python eval.py --config-name=$CONFIG.yaml policy=$POLICY "$@"
