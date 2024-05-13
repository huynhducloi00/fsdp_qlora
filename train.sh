python train.py \
--train_type bnb_dora \
--model_name meta-llama/Meta-Llama-3-8B \
--dataset orca_math \
--dataset_samples 10000 \
--batch_size 1 \
--context_length 2048 \
--gradient_accumulation_steps 2 \
--sharding_strategy full_shard \
--use_gradient_checkpointing true \
--reentrant_checkpointing true \
--lora_rank 8 \
--use_cpu_offload false \
--use_activation_cpu_offload false \
--project_name "fsdp-quantized-ft-exps" \
--save_model true \
--output_dir models/Llama-3-8b-orca-math-10k-bnb-QDoRA \
--log_to wandb

# Compare LORA and QLORA on Alpaca dataset with same effective batch size ~32, lr sched, and lr.
# Reference for some hyperparams: https://arxiv.org/abs/2305.14314
# LORA (pure bf16)
# https://wandb.ai/answerdotai/fsdp/runs/gb34o6p4?workspace=user-k-answer-ai
# NOTE: Loss curve is flat - 1) use lower lr ? 2) start immediate annealing get_cosine_one_cycle_scheduler(..., min_lr_fraction=0.0)
# python train.py \
# --model_name meta-llama/meta-Llama-3-8b \
# --gradient_accumulation_steps 2 \
# --batch_size 8 \
# --context_length 512 \
# --num_epochs 1 \
# --train_type lora \
# --use_gradient_checkpointing False \
# --use_cpu_offload False \
# --log_to wandb \
# --dataset alpaca \
# --verbose false \
# --save_model true \
# --output_dir ~/models/lora_alpaca

# # QLORA (pure bf16)
# python train.py \
# --model_name meta-llama/Llama-2-7b-hf \
# --gradient_accumulation_steps 2 \
# --batch_size 8 \
# --context_length 512 \
# --num_epochs 1 \
# --train_type qlora \
# --use_gradient_checkpointing False \
# --use_cpu_offload False \
# --log_to wandb \
# --dataset alpaca \
# --verbose false \
# --save_model true \
# --output_dir ~/models/qlora_alpaca

# # QLORA (autocast bf16)
# python train.py \
# --model_name meta-llama/Llama-2-7b-hf \
# --precision bf16_buffers_autocast \
# --gradient_accumulation_steps 2 \
# --batch_size 8 \
# --context_length 512 \
# --num_epochs 1 \
# --train_type qlora \
# --use_gradient_checkpointing False \
# --use_cpu_offload False \
# --log_to wandb \
# --dataset alpaca \
# --verbose false \
# --save_model true \
# --output_dir ~/models/qlora_alpaca_autocast_buffers_bf16

# # stop instance
# # requires: az login --use-device-code
# az vm deallocate -g resource-group-us-east -n a100-duo

# export CUDA_VISIBLE_DEVICES=3,4
# python train.py \
# --world_size 2 \
# --model_name meta-llama/Llama-2-7b-hf \
# --gradient_accumulation_steps 2 \
# --batch_size 1 \
# --context_length 512 \
# --num_epochs 1 \
# --sharding_strategy full_shard \
# --precision bf16 \
# --train_type hqq_lora \
# --use_gradient_checkpointing true \
# --use_cpu_offload false \
# --log_to stdout \
# --dataset alpaca \
# --verbose true

# export CUDA_VISIBLE_DEVICES=4,5
# python train.py \
# --world_size 2 \
# --master_port 12356 \
# --model_name meta-llama/Llama-2-7b-hf \
# --gradient_accumulation_steps 2 \
# --batch_size 1 \
# --context_length 512 \
# --num_epochs 1 \
# --sharding_strategy full_shard \
# --precision bf16 \
# --train_type hqq_lora \
# --use_gradient_checkpointing true \
# --use_cpu_offload false \
# --log_to stdout \
# --dataset dummy \
# --verbose true

# export CUDA_VISIBLE_DEVICES=3,4
# python train.py \
# --world_size 3 \
# --model_name meta-llama/Llama-2-70b-hf \
# --gradient_accumulation_steps 2 \
# --batch_size 1 \
# --context_length 4096 \
# --num_epochs 1 \
# --sharding_strategy full_shard \
# --precision bf16 \
# --train_type hqq_dora \
# --use_gradient_checkpointing true \
# --use_cpu_offload false \
# --log_to wandb \
# --dataset dummy \
# --verbose true      