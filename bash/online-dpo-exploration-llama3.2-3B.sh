#!/bin/bash
DATA_DIR=iter0-70k
TRAIN_SPLITS=train
TEST_SPLITS=test
TRAIN_EPOCHS=1

POLICY_MODEL=rl-llm-agent/Llama-3.2-3B-Instruct-sft-alfworld-iter0
REWARD_MODEL=rl-llm-agent/Llama-3.2-3B-Instruct-reward-alfworld-iter0

# ## small model debugging ##
# POLICY_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct
# REWARD_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct

exp_name=exploration
current_date=$(date +"%y%m%d_%H%M%S")
echo "Saving to subdir: ${current_date}_${exp_name}"

DATASET=data/alfworld/prm/${DATA_DIR}
# DATASET=data/alfworld/shaped_rewards/${DATA_DIR}
SAVE_DIR=save/online_dpo/${current_date}_${exp_name}/${POLICY_MODEL//\//-}_${DATA_DIR}/
# SAVE_DIR=save/online_dpo/${current_date}_${exp_name}/${POLICY_MODEL//\//-}_${DATA_DIR}/
# EVAL_DIR=eval/online_dpo/${DATA_DIR}

accelerate launch  --num-processes 5 \
    --config_file configs/ds_configs/deepspeed_zero2.yaml scripts/train/online_dpo_vllm_exploration_thread.py \
    --dataset_mixer "{\"${DATASET}\": 1.0}" \
    --dataset_train_splits ${TRAIN_SPLITS} \
    --dataset_eval_mixer "{\"${DATASET}\": 1.0}" \
    --dataset_eval_splits ${TEST_SPLITS} \
    --model_name_or_path ${POLICY_MODEL} \
    --reward_model_path ${REWARD_MODEL} \
    --non_stop_penalty \
    --stop_token eos \
    --learning_rate 8e-7 \
    --total_episodes 50000 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 6 \
    --gradient_checkpointing True \
    --max_prompt_token_length 2000 \
    --response_length 64 \
    --num_train_epochs ${TRAIN_EPOCHS} \
    --beta 0.03 \
    --temperature 0.7 \
    --num_generation_per_prompt 2 \
    --sanity_check_max_samples 128 \
    --output_dir ${SAVE_DIR} \
    --checkpoint_output_dir tmp/chkpts/ \
    --save_freq 20 \
    --vllm_device cuda:5 \
    --vllm_gpu_memory_utilization 0.9 \
    --hf_metadata_dataset "" \
    --no_try_launch_beaker_eval_jobs \
    --gradient_checkpointing \
    --with_tracking \
    --wandb_project_name "LLM_RM" \
    --exploration_prob 0.5
