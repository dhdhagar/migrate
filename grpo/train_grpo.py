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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--4bit", type=bool, default=False)
    parser.add_argument("--target", "-t", type=str, default="computer")
    parser.add_argument("--batch_size", "-b", type=int, default=10)
    parser.add_argument("--n_reps", "-n", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmstart", type=float, default=0)
    parser.add_argument("--strategy", type=str, default="Oracle_Single")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    params = vars(args)

    model_name = params["model"]
    target_word = params["target"]

    num_guesses = params["batch_size"]
    dataset = [
        {
            "prompt": [
                {
                    "content": 'You are a helpful chatbot with high attention to detail who is not talkative and responds only \
    with the answer and no additional conversation. All your responses should be in JSON format, i.e. {key: value}, where the \
    key is always "response" and the value can be a string, int, list, or dict, depending on the context.',
                    "role": "system",
                },
                {
                    "content": f'Your task is to guess a hidden word from the English dictionary. Stick to proper, single-word \
    English words. Now, guess exactly n={num_guesses} new word(s) that could be the hidden word. Be creative! (Note: give only \
    a list of word(s) in the provided JSON format, e.g. {{"response": ["word1", "word2",...]}})',
                    "role": "user",
                },
            ]
        }
        for _ in range(params["steps"])
    ]

    # Placeholder reward func. We process and compute our own rewards in the trainer
    def reward_len(completions, **kwargs):
        return 1.0

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    if params["4bit"]:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=quant_config, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = f'logs/{params["strategy"]}/{target_word}'
    os.makedirs(logdir, exist_ok=True)
    logfile = f'{logdir}/{timestamp}_{params["warmstart"]}.log'
    with open(logfile, "w") as file:
        json.dump({"Params": params, "Guesses": [], "Final_Sample": ""}, file, indent=4)

    training_args = GRPOConfig(
        output_dir="GRPO",
        logging_steps=5,
        fp16=False,
        bf16=True,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        temperature=0.9,
        num_generations=1,
        max_completion_length=128,
        beta=0.04,
    )
    trainer = SemantleGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
        logfile=logfile,
        target=params["target"],
        num_guesses=params["batch_size"],
        n_reps=params["n_reps"],
        strategy=params["strategy"],
    )
    trainer.train()
