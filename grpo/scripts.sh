#!/bin/bash

current_time=$(date "+%Y-%m-%d_%H-%M-%S")
targets=("airbase" "birthstone" "cement" "computer" "filament" "machetes" "meatloaf" "mob" "polyethylene" "skillet")
model_name="meta-llama/Llama-3.2-1B-Instruct"

# Oracle_Single
for target in "${targets[@]}"; do
  accelerate launch train_grpo.py --date "$current_time" --model "$model_name" --target "$target" --n_reps 1 --num_guesses 10 --strategy "Oracle_Single"
done
python generate_plots.py --date ${current_time} --model "$model_name" --steps 100 --strategy "Oracle_Single"

# Online_Single
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
for target in "${targets[@]}"; do
  accelerate launch train_grpo.py --date "$current_time" --model "$model_name" --target "$target" --n_reps 1 --num_guesses 10 --strategy "Online_Single"
done
python generate_plots.py --date ${current_time} --model "$model_name" --steps 100 --strategy "Online_Single"

# Online_Mean
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
for target in "${targets[@]}"; do
  accelerate launch train_grpo.py --date "$current_time" --model "$model_name" --target "$target" --n_reps 5 --num_guesses 2 --strategy "Online_Mean"
done
python generate_plots.py --date ${current_time} --model "$model_name" --steps 100 --strategy "Online_Mean"

# Online_Max
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
for target in "${targets[@]}"; do
  accelerate launch train_grpo.py --date "$current_time" --model "$model_name" --target "$target" --n_reps 5 --num_guesses 2 --strategy "Online_Max"
done
python generate_plots.py --date ${current_time} --model "$model_name" --steps 100 --strategy "Online_Max"

# Online_Batch_Mean
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
for target in "${targets[@]}"; do
  accelerate launch train_grpo.py --date "$current_time" --model "$model_name" --target "$target" --n_reps 1 --num_guesses 10 --strategy "Online_Batch_Mean"
done
python generate_plots.py --date ${current_time} --model "$model_name" --steps 100 --strategy "Online_Batch_Mean"

Mean# Online_Batch_Max
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
for target in "${targets[@]}"; do
  accelerate launch train_grpo.py --date "$current_time" --model "$model_name" --target "$target" --n_reps 1 --num_guesses 10 --strategy "Online_Batch_Max"
done
python generate_plots.py --date ${current_time} --model "$model_name" --steps 100 --strategy "Online_Batch_Max"
