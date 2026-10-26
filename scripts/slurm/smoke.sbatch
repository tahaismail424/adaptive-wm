#!/bin/bash
# Cheap end-to-end check on a T4/RTX node: data loading, a few hundred train
# steps, validation, and checkpoint saving. Also reports throughput (it/s)
# so you can extrapolate full-run wall time.
#
#   sbatch --export=DATA=pusht_shapeI scripts/slurm/smoke.sbatch
#
#SBATCH --job-name=lewm-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/slurm_logs/%x-%j.out

source scripts/slurm/env.sh   # before set -u: Lmod's module() trips on unbound vars
set -euo pipefail

DATA=${DATA:-pusht_shapeI}
NAME=smoke_${DATA}

# T4/RTX (Turing) have no native bf16 -> fp16 mixed precision
python train.py data=$DATA $WANDB_ARGS \
    output_model_name=$NAME subdir=${NAME}_$SLURM_JOB_ID \
    trainer.precision=16-mixed \
    trainer.max_epochs=2 \
    +trainer.limit_train_batches=200 \
    +trainer.limit_val_batches=20 \
    log_every_n_steps=10 \
    loader.batch_size=64 \
    num_workers=14 \
    "$@"

echo "--- checkpoints:"; ls -la $STABLEWM_HOME/checkpoints/$NAME
echo "--- latest metrics.csv:"; tail -5 "$(ls -t $(find $SPT_CACHE_DIR/runs -name metrics.csv) | head -1)"
