#!/bin/bash

current_time=$(date "+%Y-%m-%d_%H-%M-%S")
targets=('IGF1R' 'JAK2' 'KIT' 'LCK' 'MAPK14' 'MAPKAPK2' 'MET' 'PTK2' 'PTPN1' 'SRC' 'ABL1' 'AKT1' 'AKT2' 'CDK2' 'CSF1R' 'EGFR' 'KDR' 'MAPK1' 'FGFR1' 'ROCK1' 'MAP2K1' 'PLK1' 'HSD11B1' 'PARP1' 'PDE5A' 'PTGS2' 'ACHE' 'MAOB' 'CA2' 'GBA' 'HMGCR' 'NOS1' 'REN' 'DHFR' 'ESR1' 'ESR2' 'NR3C1' 'PGR' 'PPARA' 'PPARD' 'PPARG' 'AR' 'THRB' 'ADAM17' 'F10' 'F2' 'BACE1' 'CASP3' 'MMP13' 'DPP4' 'ADRB1' 'ADRB2' 'DRD2' 'DRD3' 'ADORA2A' 'CYP2C9' 'CYP3A4' 'HSP90AA1')
model_name="meta-llama/Llama-3.2-3B-Instruct"

for target in "${targets[@]}"; do
  # Migrate (OPRO)
  accelerate launch train.py --task "molecule" --date "$current_time" --model "$model_name" --target "$target" --trl_seed 2 --strategy "migrate_opro" --num_train_epochs 1 --wandb_prefix "dockstring_migrate_opro" --wandb_tags "grpo,dockstring,migrate,opro" --min_training_size 40 --max_training_size 40 --inf_batch_size 10 --num_guesses 1 --batch_size 5 --num_generations 5 --online_max_completion_length 256 --inf_num_samples 0 --learning_rate 5e-5 --lora_alpha 16 --lora_rank 64 --num_iterations 2 --migrate_sampling_strategy "opro" --greedy_topk 5 --migrate_gamma 2 --migrate_alpha 2 --migrate_beta 1
done
