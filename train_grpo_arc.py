from unsloth import FastLanguageModel, PatchFastRL  # Needs to be first import

PatchFastRL("GRPO", FastLanguageModel)

import os
import time
from datetime import datetime
import argparse
import json
import torch

from trl import GRPOConfig

# from GRPOTrainer import GRPOTrainer
from trainers.TTT_GRPOTrainer import TTT_GRPOTrainer, CustomProgressCallback
import prompts as prompts_getter
from typing import List
import wandb
from inference import run_induction_inference, run_transduction_inference

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
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmstart", type=float, default=0)
    parser.add_argument("--strategy", type=str, default="gold")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--neighborhood_sampling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--task", type=str, default="semantle")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=10)
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
        "--arc_dataset_solutions_file",
        type=str,
        default="kaggle/input/arc-prize-2024/arc-agi_evaluation_solutions.json",
    )
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_permutations", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--readable_prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_training_size", type=int, default=50)
    parser.add_argument("--max_training_size", type=int, default=80)
    parser.add_argument("--max_validation_size", type=int, default=64)
    parser.add_argument("--max_test_size", type=int, default=64)
    parser.add_argument("--validation_interval", type=int, default=5)
    parser.add_argument("--use_vllm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grpo_weight", type=float, default=1.0)
    parser.add_argument("--nll_weight", type=float, default=0.0)
    parser.add_argument("--pro_loss_weight", type=float, default=0.0)
    parser.add_argument("--pro_loss_only_positive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_temperature", type=float, default=1.0)
    parser.add_argument("--use_train_temp_schedule", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb_prefix", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--save_datasets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inject_best_at_lowest_score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_early_stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_seq_len", type=int, default=2048)

    parser.add_argument("--use_barc_format", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_induction", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--only_inference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inf_batch_size", type=int, default=10)
    parser.add_argument("--inf_temperature", type=float, default=0.8)
    parser.add_argument("--inf_max_new_tokens", type=int, default=512)
    parser.add_argument("--inf_num_samples", type=int, default=64)

    args = parser.parse_args()
    return args


def create_dataset(params, tokenizer=None):
    if params["task"] != "arc":
        return [
            {"prompt": prompts_getter.get_prompt(params["task"], params["num_guesses"], params["target"])}
            for _ in range(params["steps"])
        ]
    else:
        return prompts_getter.get_arc_datasets(
            params["target"],
            tokenizer=tokenizer,
            **{k: v for k, v in params.items() if k in prompts_getter.get_arc_datasets.__code__.co_varnames},
        )


def setup_logging(params, validation_data, test_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = f'logs/{params["strategy"]}/{params["target"]}'
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{timestamp}.log"
    wandb_id = f'{params["strategy"]}-{params["target"]}-{timestamp}'
    if params["wandb_prefix"] is not None:
        wandb_id = f"{params['wandb_prefix']}-{wandb_id}"
    init_data = {
        "params": params,
        "task": {
            "validation": {"problem": str(validation_data["problem"]), "solution": str(validation_data["solution"])},
            "test": {"problem": str(test_data["problem"]), "solution": str(test_data["solution"])},
        },
        "guesses": [],
        "validation": [],
    }
    with open(logfile, "w") as file:
        file.write(json.dumps(init_data, indent=2))
    return init_data, logdir, logfile, wandb_id


def setup_model(params):
    model, tokenizer = FastLanguageModel.from_pretrained(
        params["model"],
        device_map="auto",
        max_lora_rank=params["lora_rank"],
        gpu_memory_utilization=0.8,
        fast_inference=params["use_vllm"],
        # quantization_config=quant_config,
        # torch_dtype=torch.bfloat16,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=params["lora_rank"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"],
        bias="none",
        target_modules=params["target_modules"],
        use_rslora=False,
        loftq_config=None,
        # use_gradient_checkpointing = "unsloth"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


# Placeholder reward func. We process and compute our own rewards in the trainer
def reward_len(completions, **kwargs):
    return 1.0


def main(params):
    model, tokenizer = setup_model(params)

    training_dataset, validation_dataset, test_dataset = create_dataset(params, tokenizer=tokenizer)

    data, logdir, logfile, wandb_id = setup_logging(params, validation_dataset, test_dataset)

    if params["save_datasets"]:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "training_dataset.json"), "w") as file:
            json.dump(training_dataset, file, indent=2)
        with open(os.path.join(logdir, "validation_dataset.json"), "w") as file:
            json.dump(validation_dataset, file, indent=2)
        with open(os.path.join(logdir, "test_dataset.json"), "w") as file:
            json.dump(test_dataset, file, indent=2)

    training_args = GRPOConfig(
        output_dir="GRPO",
        logging_steps=1,
        fp16=False,
        bf16=True,
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params["batch_size"],
        gradient_accumulation_steps=params["grad_acc_steps"],
        num_train_epochs=params["num_train_epochs"],
        temperature=params["online_temperature"],
        num_generations=params["num_generations"],
        max_completion_length=params["online_max_completion_length"],
        beta=params["beta"],
        report_to="wandb",
        run_name=wandb_id,
        use_vllm=params["use_vllm"],
        max_prompt_length=None,
        disable_tqdm=True,  # To avoid double tqdm bars because of CustomProgressCallback
    )
    trainer = TTT_GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        # peft_config=peft_config, # no need with unsloth
        reward_funcs=reward_len,  # Placeholder reward func
        args=training_args,
        train_dataset=training_dataset,
        validation_dataset=validation_dataset,
        validation_interval=params["validation_interval"],
        logfile=logfile,
        target=params["target"],
        strategy=params["strategy"],
        task=params["task"],
        arc_dataset_file=params["arc_dataset_file"],
        generation_args={},
        grpo_weight=params["grpo_weight"],
        nll_weight=params["nll_weight"],
        pro_loss_weight=params["pro_loss_weight"],
        pro_loss_only_positive=params["pro_loss_only_positive"],
        train_temperature=params["train_temperature"],
        use_train_temp_schedule=params["use_train_temp_schedule"],
        inject_best_at_lowest_score=params["inject_best_at_lowest_score"],
        inf_batch_size=params["inf_batch_size"],
        use_early_stopping=params["use_early_stopping"],
        neighborhood_sampling=params["neighborhood_sampling"],
        n_neighbors=params["n_neighbors"],
        callbacks=[CustomProgressCallback()],
        use_barc_format=params["use_barc_format"],
        use_induction=params["use_induction"],
    )

    start_time = time.time()
    if not params["only_inference"]:
        trainer.train()
        wandb.config.update(params)
        _model_for_inference = trainer.model

        # Log training time
        train_time = time.time() - start_time
        with open(logfile, "r") as file:
            data = json.load(file)
            data["duration"] = train_time
        with open(logfile, "w") as file:
            json.dump(data, file, indent=2)
    else:
        _model_for_inference = model
        wandb.init(project="ttt-arc", id=wandb_id)

    # Update wandb run with tags
    if params["wandb_tags"] is not None:
        wandb.run.tags = params["wandb_tags"].split(",")

    # Log training time
    train_time = time.time() - start_time
    with open(logfile, "r") as file:
        data = json.load(file)
        data["Duration"] = train_time
    with open(logfile, "w") as file:
        json.dump(data, file, indent=2)

    print("\n==================\nRUNNING ON TEST\n==================")
    if params["use_induction"]:
        best_program = sorted(trainer.past_guesses.items(), key=lambda x: x[1], reverse=True)[0][0]
        data.update(
            run_induction_inference(
                trainer, tokenizer, _model_for_inference, training_dataset, test_dataset, params, best_program
            )
        )
    else:
        data.update(
            run_transduction_inference(trainer, tokenizer, _model_for_inference, training_dataset, test_dataset, params)
        )

    # Save these logs to wandb
    wandb.run.summary["test/solved_majority"] = data["pass1"]["score"] == 1 if data["pass1"] is not None else 0
    wandb.run.summary["test/score_majority"] = data["pass1"]["score"] if data["pass1"] is not None else 0
    wandb.run.summary["test/solved_majority_pass2"] = data["pass2"]["score"] == 1 if data["pass2"] is not None else 0
    wandb.run.summary["test/score_majority_pass2"] = data["pass2"]["score"] if data["pass2"] is not None else 0
    wandb.run.summary["test/solved_oracle"] = data["oracle"]["score"] == 1
    wandb.run.summary["test/best_score"] = data["oracle"]["score"]
    wandb.run.summary["test/best_completion"] = data["oracle"]["output"]

    # print(f"TEST SOLVED @ pass1: {data['pass1']['score'] == 1}")
    # print(f"TEST SOLVED @ pass2: {data['pass2']['score'] == 1}")
    print(f"TEST SOLVED ORACLE: {data['oracle']['score'] == 1}")
    if data["oracle"]["score"] < 1:
        print(f"BEST COMPLETION: {data['oracle']['output']}")

    with open(logfile, "w") as file:
        json.dump(data, file, indent=4)

    wandb.run.summary["log_fpath"] = logfile
    wandb.finish()

    # Save model
    if params["save_model"]:
        model.save_pretrained_merged(logdir, tokenizer, save_method="lora")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up W&B
    os.environ["WANDB_PROJECT"] = "ttt-arc"
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    # Get arguments
    args = parse_arguments()
    params = vars(args)

    main(params)
