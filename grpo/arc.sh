#!/bin/bash

current_time=$(date "+%Y-%m-%d_%H-%M-%S")
# targets=("00576224" "0934a4d8" "0a1d4ef5" "0bb8deee" "0c786b71" "0c9aba6e" "12997ef3" "17cae0c1" "195ba7dc" "1990f7a8")
# targets=('17cae0c1' '31d5ba1a' '4cd1b7b2' '66e6c45b' '6ea4a07e' 'be03b35f' 'ca8de6ea' 'e133d23d' 'e345f17b' 'e633a9e5')
targets=('17cae0c1' '31d5ba1a' '4cd1b7b2' '66e6c45b' '6ea4a07e' 'be03b35f')

# targets=("00576224")
# targets=("0bb8deee")
# targets=("0a1d4ef5" "0bb8deee" "0c786b71" "0c9aba6e" "12997ef3" "17cae0c1" "195ba7dc" "1990f7a8")
model_name="ekinakyurek/marc-8B-finetuned-llama3"
# model_name="meta-llama/Llama-3.2-1B-Instruct"

for target in "${targets[@]}"; do
  accelerate launch train_grpo.py --date "$current_time" --model "$model_name" --target "$target" --n_reps 5 --num_guesses 1 --strategy "Greedy_Single" --task "arc" --steps 2 --num_train_epochs 1
done

# Other arguments
# learning_rate
# num_train_epochs
# online_temperature
# online_max_completion_length
# beta
# lora_rank
# lora_alpha
# lora_dropout
# target_modules
# arc_dataset_file
