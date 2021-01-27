#!/bin/sh

# If you want to resume wandb run, use the command below.
# $ WANDB_RUN_ID=wandb_run_id bash alfa+maml.sh run_name 0 [1]

export WANDB_RUN_NAME=$1
export GPU_ID=$2

echo $GPU_ID

cd ..
export DATASET_DIR="../datasets/"
export CUDA_VISIBLE_DEVICES=$2,$3
# Activate the relevant virtual environment:
python train_maml_system.py \
	--name_of_args_json_file experiment_config/alfa+maml.json \
	--gpu_to_use $GPU_ID \
	--wandb_run_name $WANDB_RUN_NAME
