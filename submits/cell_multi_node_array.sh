#!/bin/bash -l
#SBATCH --job-name=cf-pretrain
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=4
#SBATCH --array [0-14]%1



# Run job step
LOG_INTERVAL=16
MAX_LENGTH=1200
per_proc_batch_size=64
LAYERS=6
EMBSIZE=256
JOB_NAME="debug"
SAVE_DIR="./save/cell_no_dat"
export GPUS_PER_NODE=4

CURRENT_EPOCH=$SLURM_ARRAY_TASK_ID

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

if [ $CURRENT_EPOCH -eq 0 ]; then
  echo "Running first epoch (epoch $CURRENT_EPOCH)"

srun --environment=bionemo accelerate launch \
    --num_processes $GPUS_PER_NODE \
    --mixed_precision bf16 \
    ./pretrain.py \
    --save-dir $SAVE_DIR \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size)) \
    --nlayers $LAYERS \
    --nheads 8 \
    --embsize $EMBSIZE \
    --d-hi 512 \
    --epochs 15 \
    --num-epochs 1 \
    --lr 0.0001 \
    --warmup-ratio-or-step 10000 \
    --log-interval $LOG_INTERVAL \
    --trunc-by-sample \
    --loss "mse" \
    --train-path "./pretraining_cells" \
    --zero-percentages 0.2 0.4 0.6 \
    --balance-primary "tissue" \
    --balance-secondary "technology" \
    --conditions "technology" \
    --wandb "fulldata"

else
PREV_EPOCH=$((CURRENT_EPOCH - 1))
CHECKPOINT_PATH="$SAVE_DIR/epoch_$PREV_EPOCH"

srun --environment=bionemo accelerate launch \
    --num_processes $GPUS_PER_NODE \
    --mixed_precision bf16 \
    ./pretrain.py \
    --resume-from-checkpoint $CHECKPOINT_PATH \
    --num-epochs 1

fi