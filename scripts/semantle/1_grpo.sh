#!/bin/bash

current_time=$(date "+%Y-%m-%d_%H-%M-%S")
targets=("airbase" "birthstone" "cement" "computer" "filament" "machetes" "meatloaf" "mob" "polyethylene" "skillet")
model_name="meta-llama/Llama-3.2-3B-Instruct"

for target in "${targets[@]}"; do
  # Vanilla GRPO
  accelerate launch train_grpo_arc.py --task "semantle" --date "$current_time" --model "$model_name" --target "$target" --trl_seed 2 --warmstart_seed 43 --strategy "grpo" --num_train_epochs 1 --wandb_prefix "semantle_grpo_vanilla" --wandb_tags "grpo,semantle_vanilla" --min_training_size 100 --max_training_size 100 --inf_batch_size 4 --num_guesses 10 --batch_size 5 --num_generations 5 --online_max_completion_length 64 --inf_num_samples 0 --learning_rate 5e-5 --lora_alpha 16 --lora_rank 64 --num_iterations 2 --migrate_gamma 0 --migrate_alpha 10 --migrate_beta 0
done
