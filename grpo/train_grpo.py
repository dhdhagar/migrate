import os
from datetime import datetime
import argparse
import json
import torch
from trl import GRPOConfig
from SemantleGRPOTrainer import SemantleGRPOTrainer
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

num_guesses = 10
dataset = [
    {
        "prompt": [
            {
                "content": 'You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no \
                additional conversation. All your responses should be in JSON format, i.e. {key: value}, where the key is always \
                "response" and the value can be a string, int, list, or dict, depending on the context.',
                "role": "system",
            },
            {
                "content": f'Your task is to guess a hidden word from the English dictionary. Stick to proper, single-word English words. Now, guess exactly n={num_guesses} new word(s) that could be the hidden word. Be creative! (Note: give only a list of word(s) in the provided JSON format, e.g. {{"response": ["word1", "word2",...]}})',
                "role": "user",
            },
        ]
    }
    for _ in range(100)
]


# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    print(completions)
    return [-abs(20 - len(completion)) for completion in completions]


peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
quant_config = BitsAndBytesConfig(
    load_in_8bit=True
    # load_in_4bit=True,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16,
)


model_name = "Qwen/Qwen2-0.5B-Instruct"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", quantization_config=quant_config, torch_dtype=torch.bfloat16
)

training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO",
    logging_steps=10,
    fp16=False,
    bf16=True,
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    temperature=0.9,
    num_generations=2,
    max_completion_length=128,
)
trainer = SemantleGRPOTrainer(
    model=model,
    peft_config=peft_config,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
