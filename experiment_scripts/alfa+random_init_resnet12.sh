#!/bin/sh

export $WANDB_ID=$1
export GPU_ID=$2

echo $GPU_ID

cd ..
export DATASET_DIR="datasets/"
export CUDA_VISIBLE_DEVICES=$2,$3,$4,$5
# Activate the relevant virtual environment:
python train_maml_system.py --name_of_args_json_file experiment_config/alfa+random_init_resnet12.json --gpu_to_use $GPU_ID --wandb_id $WANDB_ID
