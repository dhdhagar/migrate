# test-time-search# test-time-search

## Description
These are experiments for playing Semantle with test-time fine-tuning using [Online DPO](https://huggingface.co/docs/trl/en/online_dpo_trainer). There are three approaches towards test-time fine-tuning denoted as `random`, `greedy`, and `cheat`. The custom judge for Semantle is based on comparing the cosine similarities of the embeddings of generated responses and the target. The embedder used is `princeton-nlp/sup-simcse-roberta-large`.

### Strategies
- `random`
  - Akin to vanilla online DPO
  - `N` responses (guesses) are sampled and `N` pairs of responses are randomly chosen to be judged
- `greedy`
  - Keeps track of the top `N` responses
  - In each training iteration, `N` responses are sampled and each is randomly judged against one of the top `N` previous responses
- `cheat`
  - Directly compares each of the `N` sampled responses to the target response


## Arguments
Every model is experimented with the following LoRA configuration:
```python
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```
- Llama 3.2 1b (8bit)
  - batch_size: 10
  - steps: 100
  - lr: 1e-4
  - beta: 0.1
- Llama 3.2 3b (8bit)
  - batch_size: 10
  - steps: 100
  - lr: 1e-4
  - beta: 0.1
- Llama 3.1 8b (8bit)
  - batch_size: 5
  - steps: 100
  - lr: 1e-4
  - beta: 0.1

