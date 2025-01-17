import os
import argparse
import json
from datetime import datetime
from datasets import Dataset
from trl import OnlineDPOConfig
from SemantleOnlineDPOTrainer import SemantleOnlineDPOTrainer
from SimPairJudge import SimPairJudge
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
import torch

import pandas as pd
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", "-t", type=str, default="computer")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--strategy", "-s", type=str, default="random")
    parser.add_argument("--batch_size", "-b", type=int, default=10)
    parser.add_argument("--n_reps", "-n", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmstart", type=int, default=0)
    parser.add_argument("--g", type=int, default=5)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    params = vars(args)
    # model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    # model_name = "Qwen/Qwen2-0.5B-Instruct"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model_name = params["model"]
    target_word = params["target"]

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=quant_config, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    judge = SimPairJudge(target_word, "princeton-nlp/sup-simcse-roberta-large")
    num_guesses = params["batch_size"]
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
        for _ in range(params["steps"])
    ]
    train_dataset = Dataset.from_list(dataset)
    del dataset

    training_args = OnlineDPOConfig(
        output_dir=f"{model_name}-OnlineDPO",
        logging_steps=5,
        fp16=False,
        bf16=True,
        use_cpu=True,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        temperature=0.9,
        max_new_tokens=int(32 + (5 * num_guesses)),
        beta=[0.1],
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = f'logs/{target_word}/{params["strategy"]}'
    os.makedirs(logdir, exist_ok=True)
    logfile = f'{logdir}/{timestamp}_{params["warmstart"]}.log'
    trainer = SemantleOnlineDPOTrainer(
        model=model,
        judge=judge,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        target=target_word,
        num_guesses=num_guesses,
        logfile=logfile,
        strategy=params["strategy"],
        warmstart=params["warmstart"],
    )
    trainer.train()

    # Sample from final trained model and log
    sample_prompt = [
        {
            "content": 'You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no \
additional conversation. All your responses should be in JSON format, i.e. {key: value}, where the key is always \
"response" and the value can be a string, int, list, or dict, depending on the context.',
            "role": "system",
        },
        {
            "content": f'Your task is to guess a hidden word from the English dictionary. Stick to proper, single-word English words. Now, guess exactly n=20 new word(s) that could be the hidden word. Be creative! (Note: give only a list of word(s) in the provided JSON format, e.g. {{"response": ["word1", "word2",...]}})',
            "role": "user",
        },
    ]
    inputs = tokenizer.apply_chat_template(sample_prompt, tokenize=True, return_tensors="pt").to(DEVICE)
    output = model.generate(inputs, num_return_sequences=1, max_new_tokens=512, temperature=0.9)[0][len(inputs[0]) :]
    with open(logfile, "r") as file:
        data = json.load(file)
    data.append({"Final Sample": tokenizer.decode(output, skip_special_tokens=True)})
    with open(logfile, "w") as file:
        json.dump(data, file, indent=4)

    del inputs
    del output
