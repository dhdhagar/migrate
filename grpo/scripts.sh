#!/bin/bash

targets=("airbase" "birthstone" "cement" "computer" "filament" "machetes" "meatloaf" "mob" "polyethylene" "skillet")
model_name="meta-llama/Llama-3.2-1B-Instruct"

# Oracle_Single
for target in "${targets[@]}"; do
  accelerate launch train_grpo.py --model "$model_name" --target "$target" --n_reps 1 --batch_size 10 --strategy "Oracle_Single"
done

# Online_Single
for target in "${targets[@]}"; do
  accelerate launch train_grpo.py --model "$model_name" --target "$target" --n_reps 1 --batch_size 10 --strategy "Online_Single"
done

# Online_Mean
for target in "${targets[@]}"; do
  accelerate launch train_grpo.py --model "$model_name" --target "$target" --n_reps 5 --batch_size 2 --strategy "Online_Mean"
done

# Online_Max
for target in "${targets[@]}"; do
  accelerate launch train_grpo.py --model "$model_name" --target "$target" --n_reps 5 --batch_size 2 --strategy "Online_Max"
done
