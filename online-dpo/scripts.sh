#!/bin/bash

targets=("airbase" "birthstone" "cement" "computer" "filament" "machetes" "meatloaf" "mob" "polyethylene" "skillet")
warmstarts=("1" "2" "3")

model_name="meta-llama/Llama-3.2-1B-Instruct"
# model_name="meta-llama/Llama-3.2-3B-Instruct"
# model_name="meta-llama/Llama-3.1-8B-Instruct"

# Oracle
strategy="oracle"
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
for warmstart in "${warmstarts[@]}"; do
  for target in "${targets[@]}"; do
    accelerate launch train_online_dpo.py --model "$model_name" --date "$current_time" --target "$target" -s "$strategy" --warmstart "$warmstart"
  done
done

# Random
strategy="random"
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
for warmstart in "${warmstarts[@]}"; do
  for target in "${targets[@]}"; do
    accelerate launch train_online_dpo.py --model "$model_name" --date "$current_time" --target "$target" -s "$strategy" --warmstart "$warmstart"
  done
done

# Greedy
strategy="greedy"
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
for warmstart in "${warmstarts[@]}"; do
  for target in "${targets[@]}"; do
    accelerate launch train_online_dpo.py --model "$model_name" --date "$current_time" --target "$target" -s "$strategy" --warmstart "$warmstart"
  done
done

# Top_delta
strategy="top_delta"
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
for warmstart in "${warmstarts[@]}"; do
  for target in "${targets[@]}"; do
    accelerate launch train_online_dpo.py --model "$model_name" --date "$current_time" --target "$target" -s "$strategy" --warmstart "$warmstart"
  done
done

# Hard
strategy="hard"
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
for warmstart in "${warmstarts[@]}"; do
  for target in "${targets[@]}"; do
    accelerate launch train_online_dpo.py --model "$model_name" --date "$current_time" --target "$target" -s "$strategy" --warmstart "$warmstart"
  done
done

# Greedy-Related
strategy="greedy"
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
for warmstart in "${warmstarts[@]}"; do
  for target in "${targets[@]}"; do
    accelerate launch train_online_dpo.py --model "$model_name" --date "$current_time" --target "$target" -s "$strategy" --warmstart "$warmstart" --related
  done
done

# Top_delta-Related
strategy="top_delta"
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
for warmstart in "${warmstarts[@]}"; do
  for target in "${targets[@]}"; do
    accelerate launch train_online_dpo.py --model "$model_name" --date "$current_time" --target "$target" -s "$strategy" --warmstart "$warmstart" --related
  done
done

# Hard-Related
strategy="hard"
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
for warmstart in "${warmstarts[@]}"; do
  for target in "${targets[@]}"; do
    accelerate launch train_online_dpo.py --model "$model_name" --date "$current_time" --target "$target" -s "$strategy" --warmstart "$warmstart" --related
  done
done
