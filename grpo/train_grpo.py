from unsloth import FastLanguageModel  # Needs to be first import
import os
import time
from datetime import datetime
import argparse
import json
import torch
from trl import GRPOConfig

from arc_utils.utils import parse_response
# from GRPOTrainer import GRPOTrainer
from trainers.ARCTrainer import GRPOTrainer
import prompts as prompts_getter
import numpy as np
from typing import List
import wandb

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
    parser.add_argument("--4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target", "-t", type=str, default="computer")
    parser.add_argument("--num_guesses", type=int, default=10)
    parser.add_argument("--n_reps", "-n", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmstart", type=float, default=0)
    parser.add_argument("--strategy", type=str, default="Oracle_Single")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--related", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--task", type=str, default="semantle")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_acc_steps", type=int)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--online_temperature", type=float, default=1.0)
    parser.add_argument("--online_max_completion_length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=int, default=0)
    parser.add_argument("--target_modules", type=List[str], default=["q_proj", "v_proj"])
    parser.add_argument(
        "--arc_dataset_file", type=str, default="kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json"
    )
    parser.add_argument(
        "--arc_dataset_solutions_file", type=str,
        default="kaggle/input/arc-prize-2024/arc-agi_evaluation_solutions.json"
    )
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--all_combinations", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--readable_prompt", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    return args


def create_dataset(params):
    if params["task"] != "arc":
        return [
            {"prompt": prompts_getter.get_prompt(params["task"], params["num_guesses"], params["target"])}
            for _ in range(params["steps"])
        ]
    else:
        return prompts_getter.get_arc_prompt(params["target"], params["arc_dataset_file"],
                                             params["arc_dataset_solutions_file"],
                                             readable_prompt=params["readable_prompt"],
                                             all_combinations=params["all_combinations"])


def setup_logging(params):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = f'logs/{params["strategy"]}/{params["target"]}'
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{timestamp}.log"
    wandb_id = f'{params["strategy"]}-{params["target"]}-{timestamp}'
    with open(logfile, "w") as file:
        file.write(
            json.dumps({
                "params": params, "guesses": [], "chosen": [], "validation": [], "final_sample": ""
            }, indent=2))
    return logdir, logfile, wandb_id


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


def main(params):
    dataset, validation_example, test_example = create_dataset(params)
    # dataset = [dataset[0].copy() for i in range(100)]
    if params["task"] == "arc":
        params["batch_size"] = 1  # len(dataset)

    # model, tokenizer, peft_config = setup_model(params)
    model, tokenizer = setup_model(params)

    logdir, logfile, wandb_id = setup_logging(params)

    training_args = GRPOConfig(
        output_dir="GRPO",
        logging_steps=5,
        fp16=False,
        bf16=True,
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params["batch_size"],
        gradient_accumulation_steps=len(dataset) if params["grad_acc_steps"] is None else params["grad_acc_steps"],
        num_train_epochs=params["num_train_epochs"],
        temperature=params["online_temperature"],
        num_generations=params["num_generations"],
        max_completion_length=params["online_max_completion_length"],
        beta=params["beta"],
        report_to="wandb",
        run_name=wandb_id
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        # peft_config=peft_config, # no need with unsloth
        reward_funcs=reward_len,  # Placeholder reward func
        args=training_args,
        train_dataset=dataset,
        validation_example=validation_example,
        logfile=logfile,
        target=params["target"],
        n_reps=params["n_reps"],
        strategy=params["strategy"],
        sample_related=params["related"],
        task=params["task"],
        arc_dataset_file=params["arc_dataset_file"],
        generation_args={}
    )
    start_time = time.time()

    trainer.train()
    wandb.finish()

    # Log training time
    train_time = time.time() - start_time
    with open(logfile, "r") as file:
        data = json.load(file)
        data["Duration"] = train_time
    with open(logfile, "w") as file:
        json.dump(data, file, indent=2)

    if params["task"] == "arc":
        prompt = test_example["prompt"]
        solution = test_example["solution"]

        inputs = tokenizer.apply_chat_template([prompt], tokenize=True, return_tensors="pt").to(DEVICE)
        attention_mask = (inputs != tokenizer.pad_token_id).long()
        with torch.no_grad():
            output0 = model.generate(inputs,
                                     attention_mask=attention_mask,
                                     do_sample=False,
                                     max_new_tokens=1024)[:, len(inputs[0]):]
            output = model.generate(inputs,
                                    attention_mask=attention_mask,
                                    num_return_sequences=5,
                                    max_new_tokens=1024,
                                    temperature=0.9)[:, len(inputs[0]):]
        with open(logfile, "r") as file:
            data = json.load(file)
        decoded_greedy = tokenizer.batch_decode(output0, skip_special_tokens=True)
        decoded_sample = tokenizer.batch_decode(output, skip_special_tokens=True)
        final_samples = [parse_response(decoded_greedy + decoded_sample)]
        final_scores = [trainer.get_bb_score(sample, solution) for sample in final_samples]
        data["Final_Sample"] = decoded_greedy + decoded_sample
        data["Final_Score"] = final_scores
        with open(logfile, "w") as file:
            json.dump(data, file, indent=4)

    # Save model
    if params["save_model"]:
        model.save_pretrained_merged(logdir, tokenizer, save_method="lora")


if __name__ == "__main__":
    # Set up W&B
    os.environ["WANDB_PROJECT"] = "ttt-arc"
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    # Get arguments
    args = parse_arguments()
    params = vars(args)

    main(params)
