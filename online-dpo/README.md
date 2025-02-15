# test-time-search

## Description
These are experiments for playing Semantle with test-time fine-tuning using [Online DPO](https://huggingface.co/docs/trl/en/online_dpo_trainer). The custom judge for Semantle is based on comparing the cosine similarities of the embeddings of generated responses and the target. The embedder used is `princeton-nlp/sup-simcse-roberta-large`.

## Instructions for running experiments
All experiments are in `scripts.sh` and can be ran with the following command:
```
sh scripts.sh
```

Example of running an individual experiment
```
accelerate launch train_online_dpo.py --model "meta-llama/Llama-3.2-1B-Instruct" --date "2025-01-1_00-00-00" --target "computer" -s "oracle" --warmstart "1"
```

## Currently Implemented Preference Constructions
In each iteration, `g` preference pairs are created according to the following strategies:
- `Oracle`
  - The chosen completion is replaced with the `target`
- `Random`
  - `g` pairs are randomly chosen from all possible pairing
- `Greedy`
  - The chosen completion is replaced with the best completion found so far
- `Top_delta`
  - The top `g` pairs with the largest delta in black-box scores
  - The chosen completion can be any completion found so far
  - The rejected completion must be a completion from the online sample of the active model
- `Hard`
  - The top `g` pairs with the highest completion scores and smallest delta in scores
  - Pairs are sorted by highest chosen's score and then by smallest delta 
  - The chosen completion can be any completion found so far
  - The rejected completion must be a completion from the online sample of the active model
- `Greedy-Related`
  - Similar to `Greedy` but substitutes the chosen completion with a related word
- `Top_delta-Related`
  - Similar to `Top_delta` but substitutes the chosen completion with a related word
- `Hard-Related`
  - Similar to `Hard-Related` but substitutes the chosen completion with a related word
