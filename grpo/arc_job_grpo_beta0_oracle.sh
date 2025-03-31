#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=50000
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --constraint=[a40|l40s|a100]
#SBATCH --time 8:00:00
#SBATCH -A pi_mccallum_umass_edu
#SBATCH -o logs/slurm-Eval--%j.out
#SBATCH -e logs/slurm-Eval--%j.err

export HF_HOME="/work/pi_mccallum_umass_edu/pkphan_umass_edu/pkphan/hf_cache"
export HF_HUB_CACHE="/work/pi_mccallum_umass_edu/pkphan_umass_edu/pkphan/hf_cache/hub"

module load conda/latest
module load cuda/12.6
conda activate /work/pi_mccallum_umass_edu/pkphan_umass_edu/pkphan/.conda/dpo

current_time=$(date "+%Y-%m-%d_%H-%M-%S")

# Full 54 tasks
default_targets=('00576224' '0c786b71' '0c9aba6e' '17cae0c1' '195ba7dc' '2072aba6' '27a77e38' '281123b4' '31d5ba1a' '32e9702f' '34b99a2b' '3b4c2228' '3d31c5b3' '48131b3c' '4852f2fa' '4cd1b7b2' '506d28a5' '5783df64' '59341089' '5d2a5c43' '60c09cac' '626c0bcc' '62b74c02' '66e6c45b' '66f2d22f' '68b67ca3' '6a11f6da' '6ad5bdfd' '6ea4a07e' '7953d61e' '833dafe3' '8597cfd7' '8ba14f53' '9110e3c5' 'a8610ef7' 'aa18de87' 'af24b4cc' 'b0722778' 'b1fc8b8e' 'bbb1b8b6' 'be03b35f' 'c074846d' 'c8b7cc0f' 'ca8de6ea' 'd017b73f' 'd19f7514' 'e133d23d' 'e345f17b' 'e633a9e5' 'e6de6e8f' 'e99362f0' 'ed74f2f2' 'ed98d772' 'fc754716')

# Accept targets passed in as a comma-separated string
if [ -z "$1" ]; then
  targets=("${default_targets[@]}")
else
  IFS=',' read -ra targets <<< "$1"
fi

# TTT pretrained model
model_name="ekinakyurek/marc-8B-finetuned-llama3"

for target in "${targets[@]}"; do
  printf "Launching job for target %s\n" "$target"

  accelerate launch train_grpo.py --date $(date "+%Y-%m-%d_%H-%M-%S") --model "ekinakyurek/marc-8B-finetuned-llama3" --strategy "Greedy_Single" --task "arc" --num_train_epochs 1 --batch_size 10 --num_generations 10 --use_permutations --learning_rate 3e-4 --online_max_completion_length 256 --target "$target" --nll_weight=0 --grpo_weight=1 --wandb_prefix "grpo_beta0_oracle" --wandb_tags "grpo,beta0,oracle" --beta=0
done
