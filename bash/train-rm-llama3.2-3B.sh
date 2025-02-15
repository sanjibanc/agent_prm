#!/bin/bash

DATA_DIR=iter0-10k
MODEL=meta-llama/Llama-3.2-3B-Instruct

TRAIN_SPLITS=train
TEST_SPLITS=test
TRAIN_EPOCHS=1

current_date=$(date +"%y%m%d_%H%M%S")
echo $current_date

DATASET=data/alfworld/prm/${DATA_DIR}
SAVE_DIR=save/rm/${current_date}/${DATA_DIR}_${MODEL//\//-}/
EVAL_DIR=eval/rm/${current_date}/${DATA_DIR}_${MODEL//\//-}/

accelerate launch  --num-processes 2 \
    --config_file configs/ds_configs/deepspeed_zero3.yaml scripts/train/rm.py \
    --dataset_train_splits ${TRAIN_SPLITS} \
    --dataset_eval_splits ${TEST_SPLITS} \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --learning_rate 5e-5 \
    --use_peft False \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --num_train_epochs ${TRAIN_EPOCHS} \
    --num_evals 10 \
    --save_freq 500 \
    --output_dir ${SAVE_DIR} \
    --eval_dir ${EVAL_DIR} \
    --gradient_checkpointing \
    --with_tracking \
    --seed 2 \
