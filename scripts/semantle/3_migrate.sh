#!/bin/bash

current_time=$(date "+%Y-%m-%d_%H-%M-%S")
targets=("airbase" "birthstone" "cement" "computer" "filament" "machetes" "meatloaf" "mob" "polyethylene" "skillet")
model_name="meta-llama/Llama-3.2-3B-Instruct"

for target in "${targets[@]}"; do
  # Migrate
  accelerate launch train.py --task "semantle" --date "$current_time" --model "$model_name" --target "$target" --trl_seed 2 --warmstart_seed 43 --strategy "migrate_top3" --num_train_epochs 1 --wandb_prefix "semantle_migrate" --wandb_tags "grpo,semantle,migrate,top3" --min_training_size 100 --max_training_size 100 --inf_batch_size 4 --num_guesses 10 --batch_size 5 --num_generations 5 --online_max_completion_length 64 --inf_num_samples 0 --learning_rate 5e-5 --lora_alpha 16 --lora_rank 64 --num_iterations 2 --migrate_sampling_strategy "random_topk" --greedy_topk 3 --migrate_gamma 10 --migrate_alpha 0 --migrate_beta 0
done
