# Test-Time Search with GRPO

## Instructions for running (semantle) experiments
All experiments are in `scripts.sh` and can be ran with the following command:
```
sh scripts.sh
```

Example of running an individual experiment
```
accelerate launch train_grpo.py --model "meta-llama/Llama-3.2-1B-Instruct" --target "computer" --n_reps 1 --batch_size 10 --strategy "Online_Single"
```

## Instructions for running (ARC) experiments
```
sh arc.sh
```

### Arguments

- `--model`
- `--target` -- ARC task ID
- `--n_reps` -- Number of online samples per training step
- `--strategy` -- One of the strategies below (use `Greedy_Single` for ARC for now)
- `--date` -- For logging
- `--related` -- Do neighborhood sampling
- `--task` -- `["semantle"|"arc"]`
- `--learning_rate` -- Default: `5e-5`
- `--num_train_epochs` -- Default: 15
- `--online_temperature` -- Temperature of online samples, Default: 0.9
- `--online_max_completion_length` -- Max token of online samples, Default: 512
- `--beta` -- KL coefficient, Default: 0.0
- `--lora_rank` -- Default: 128
- `--lora_alpha` -- Default: 32
- `--lora_dropout` -- Default: 0.0
- `--target_modules` -- Default: `["q_proj", "v_proj"]`
- `arc_dataset_file` -- File path to `arc-agi_evaluation_challenges.json`



## Currently Implemented Variations
- `Oracle_Single`
- `Online_Single`
- `Online_Mean`
- `Online_Max`
- `Online_Batch_Mean`
- `Online_Batch_Max`
- `Greedy_Single`
- `Greedy_Batch_Mean`
- `Greedy_Batch_Max`
- `TopDelta_Batch_Mean`
- `TopDelta_Batch_Max`
- `Greedy_Batch_Mean_Related`
- `Greedy_Batch_Max_Related`
