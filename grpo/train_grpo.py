from unsloth import FastLanguageModel  # Needs to be first import
import os
import time
from datetime import datetime
import argparse
import json
import torch
from trl import GRPOConfig

# from GRPOTrainer import GRPOTrainer
from trainers.ARCTrainer import GRPOTrainer
import prompts as prompts_getter
import numpy as np
from typing import List

# from peft import LoraConfig
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
# )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--4bit", action="store_true", default=False)
    parser.add_argument("--target", "-t", type=str, default="computer")
    parser.add_argument("--num_guesses", type=int, default=10)
    parser.add_argument("--n_reps", "-n", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmstart", type=float, default=0)
    parser.add_argument("--strategy", type=str, default="Oracle_Single")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--related", action="store_true", default=False)
    parser.add_argument("--task", type=str, default="semantle")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--online_temperature", type=float, default=0.9)
    parser.add_argument("--online_max_completion_length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=int, default=0)
    parser.add_argument("--target_modules", type=List[str], default=["q_proj", "v_proj"])
    parser.add_argument(
        "--arc_dataset_file", type=str, default="kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json"
    )
    args = parser.parse_args()
    return args


def create_dataset(params):
    if params["task"] != "arc":
        return [
            {"prompt": prompts_getter.get_prompt(params["task"], params["num_guesses"], params["target"])}
            for _ in range(params["steps"])
        ]
    else:
        return prompts_getter.get_arc_prompt(params["target"], params["arc_dataset_file"])


def setup_logging(params):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = f'logs/{params["strategy"]}/{params["date"]}/{params["target"]}'
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{timestamp}.log"
    with open(logfile, "w") as file:
        json.dump({"Params": params, "Guesses": [], "Chosen": [], "Validation": [], "Final_Sample": ""}, file, indent=4)
    return logfile


def setup_model(params):
    # peft_config = LoraConfig(
    # peft_config = FastLanguageModel.get_peft_model(
    #     r=128,
    #     lora_alpha=32,
    #     lora_dropout=0.0,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    #     target_modules=["q_proj", "v_proj"],
    #     use_gradient_checkpointing = "unsloth"
    # )
    # if params["4bit"]:
    #     quant_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #     )
    # else:
    #     quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        params["model"],
        device_map="auto",
        max_lora_rank=params["lora_rank"],
        gpu_memory_utilization=0.8,
        # quantization_config=quant_config,
        # torch_dtype=torch.bfloat16,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=params["lora_rank"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"],
        bias="none",
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        target_modules=["q_proj", "v_proj"],
        use_rslora=False,
        loftq_config=None,
        # use_gradient_checkpointing = "unsloth"
    )
    # tokenizer = AutoTokenizer.from_pretrained(params["model"]) # no need with unsloth
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer  # , peft_config


# Placeholder reward func. We process and compute our own rewards in the trainer
def reward_len(completions, **kwargs):
    return 1.0


def main():
    args = parse_arguments()
    params = vars(args)

    dataset = create_dataset(params)
    print(dataset)
    # dataset = [dataset[0].copy() for i in range(100)]
    if params["task"] == "arc":
        params["batch_size"] = len(dataset)

    # model, tokenizer, peft_config = setup_model(params)
    model, tokenizer = setup_model(params)

    logfile = setup_logging(params)

    training_args = GRPOConfig(
        output_dir="GRPO",
        logging_steps=5,
        fp16=False,
        bf16=True,
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=len(dataset),
        # gradient_accumulation_steps=1,
        num_train_epochs=params["num_train_epochs"],
        # num_train_epochs=1,
        temperature=params["online_temperature"],
        num_generations=1,
        max_completion_length=params["online_max_completion_length"],
        beta=params["beta"],
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        # peft_config=peft_config, # no need with unsloth
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
        logfile=logfile,
        target=params["target"],
        n_reps=params["n_reps"],
        strategy=params["strategy"],
        sample_related=params["related"],
        task=params["task"],
        arc_dataset_file=params["arc_dataset_file"],
    )
    start_time = time.time()

    trainer.train()

    # Log training time
    train_time = time.time() - start_time
    with open(logfile, "r") as file:
        data = json.load(file)
        data["Duration"] = train_time
    with open(logfile, "w") as file:
        json.dump(data, file, indent=4)

    if params["task"] == "arc":
        with open(params["arc_dataset_file"], "r", encoding="utf-8") as handle:
            data = json.load(handle)
        context_examples = data[params["target"]]["train"]
        context = ""
        for example in context_examples:
            input_arr = np.array(example["input"])
            input_str = np.array2string(input_arr, separator=" ", formatter={"all": lambda x: str(x)})
            output_arr = np.array(example["output"])
            output_str = np.array2string(output_arr, separator=" ", formatter={"all": lambda x: str(x)})
            context += f"{input_str} -> {output_str}#\n"

        prompt = [
            {
                "content": "Figure out the underlying transformation in the following examples and apply it to the test case. "
                "Here are some examples from this transformation, your answer must follow the format. Respond with only the output grid.\nThe input-output grids "
                "are provided as python arrays:\n",
                "role": "system",
            },
            {"content": "", "role": "user"},
        ]
        prompt[0]["content"] += context
        prompt[1]["content"] = f'{np.array(data[params["target"]]["test"][0]["input"])} -> '

        inputs = tokenizer.apply_chat_template([prompt], tokenize=True, return_tensors="pt").to(DEVICE)
        attention_mask = (inputs != tokenizer.pad_token_id).long()
        with torch.no_grad():
            output0 = model.generate(inputs, attention_mask=attention_mask, do_sample=False, max_new_tokens=1024)[
                :, len(inputs[0]) :
            ]
            output = model.generate(
                inputs, attention_mask=attention_mask, num_return_sequences=5, max_new_tokens=1024, temperature=0.9
            )[:, len(inputs[0]) :]
        with open(logfile, "r") as file:
            data = json.load(file)
        decoded_greedy = tokenizer.batch_decode(output0, skip_special_tokens=True)
        decoded_sample = tokenizer.batch_decode(output, skip_special_tokens=True)
        data["Final_Sample"] = decoded_greedy + decoded_sample
        with open(logfile, "w") as file:
            json.dump(data, file, indent=4)


if __name__ == "__main__":
    main()
