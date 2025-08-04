#!/bin/bash

current_time=$(date "+%Y-%m-%d_%H-%M-%S")

targets=("00576224" "0c786b71" "0c9aba6e" "17cae0c1" "195ba7dc" "2072aba6" "27a77e38" "281123b4" "31d5ba1a" "32e9702f" "34b99a2b" "3b4c2228" "3d31c5b3" "48131b3c" "4852f2fa" "4cd1b7b2" "506d28a5" "5783df64" "59341089" "5d2a5c43" "60c09cac" "626c0bcc" "62b74c02" "66e6c45b" "66f2d22f" "68b67ca3" "6a11f6da" "6ad5bdfd" "6ea4a07e" "7953d61e" "833dafe3" "8597cfd7" "8ba14f53" "9110e3c5" "a8610ef7" "aa18de87" "af24b4cc" "b0722778" "b1fc8b8e" "bbb1b8b6" "be03b35f" "c074846d" "c8b7cc0f" "ca8de6ea" "d017b73f" "d19f7514" "e133d23d" "e345f17b" "e633a9e5" "e6de6e8f" "e99362f0" "ed74f2f2" "ed98d772" "fc754716")

model_name="barc0/Llama-3.1-ARC-Potpourri-Induction-8B"

for target in "${targets[@]}"; do
  # Migrate
  accelerate launch train_grpo_arc.py --task "arc" --date "$current_time" --model "$model_name" --target "$target" --trl_seed 2 --strategy "migrate_top1" --num_train_epochs 1 --wandb_prefix "arc_small_migrate" --wandb_tags "grpo,arc,migrate,top1" --min_training_size 64 --max_training_size 64 --inf_batch_size 16 --batch_size 16 --num_generations 16 --online_max_completion_length 1024 --inf_num_samples 0 --learning_rate 1e-5 --lora_alpha 32 --lora_rank 128 --num_iterations 1 --migrate_sampling_strategy "random_topk" --greedy_topk 1 --migrate_gamma 4 --migrate_alpha 11 --migrate_beta 1 --use_induction
done
