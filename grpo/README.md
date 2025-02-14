# Test-Time Search with GRPO

## Instructions for running experiments
All experiments are in `scripts.sh` and can be ran with the following command:
```
sh scripts.sh
```

Example of running an individual experiment
```
accelerate launch train_grpo.py --model "meta-llama/Llama-3.2-1B-Instruct" --target "computer" --n_reps 1 --batch_size 10 --strategy "Online_Single"
```

## Currently Implemented Variations
- `Oracle_Single`
- `Online_Single`
- `Online_Mean`
- `Online_Max`

## Note
- Llama 3.2 1B Instruct is unlikely to generate multiple valid completions so experiments with high `n_reps` will do poorly with this model
