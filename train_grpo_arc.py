from unsloth import FastLanguageModel, PatchFastRL  # Needs to be first import

PatchFastRL("GRPO", FastLanguageModel)

import os
import time
from datetime import datetime
import argparse
import json
import torch
from trl import GRPOConfig, maybe_apply_chat_template

from arc_utils.utils import parse_response, GridConverter

# from GRPOTrainer import GRPOTrainer
from trainers.TTT_GRPOTrainer import TTT_GRPOTrainer, CustomProgressCallback
import prompts as prompts_getter
import numpy as np
from typing import List
import wandb
from vllm import SamplingParams

from accelerate.utils import broadcast_object_list, gather_object

from trl.models.utils import unwrap_model_for_generation

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
    parser.add_argument("--neighborhood_sampling_strategy", type=str, choices=["best", "mix"], default="best")
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
    parser.add_argument("--only_inference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inf_batch_size", type=int, default=10)
    parser.add_argument("--save_datasets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inject_best_at_lowest_score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_early_stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_seq_len", type=int, default=2048)

    parser.add_argument("--use_barc_format", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_induction", action=argparse.BooleanOptionalAction, default=False)
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
        neighborhood_sampling_strategy=params["neighborhood_sampling_strategy"],
        n_neighbors=params["n_neighbors"],
        callbacks=[CustomProgressCallback()],
        use_barc_format=params["use_barc_format"],
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
    with torch.no_grad():
        prompts = test_dataset["dataset"]
        solution = test_dataset["solution"]
        prompts_text = [maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"] for prompt in prompts]
        if params["use_vllm"]:
            # Make sure model is up-to-date in vllm
            if trainer.state.global_step != trainer._last_loaded_step:
                trainer._move_model_to_vllm()
                trainer._last_loaded_step = trainer.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if trainer.accelerator.is_main_process:
                outputs = trainer.llm.generate(
                    all_prompts_text,
                    sampling_params=SamplingParams(temperature=0.0, n=1, max_tokens=512),
                    use_tqdm=False,
                )
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                trainer.accelerator.process_index * len(prompts),
                (trainer.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=DEVICE) for ids in completion_ids]

            completions = trainer.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        else:
            prompt_inputs = tokenizer(
                prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            completions = []
            for i in range(0, len(prompt_inputs["input_ids"]), params["inf_batch_size"]):
                prompt_ids, prompt_mask = (
                    prompt_inputs["input_ids"][i : i + params["inf_batch_size"]],
                    prompt_inputs["attention_mask"][i : i + params["inf_batch_size"]],
                )
                with unwrap_model_for_generation(_model_for_inference, trainer.accelerator) as unwrapped_model:
                    completion_ids = unwrapped_model.generate(
                        input_ids=prompt_ids.to(DEVICE),
                        attention_mask=prompt_mask.to(DEVICE),
                        do_sample=False,
                        max_new_tokens=512,
                    )
                    prompt_length = prompt_inputs["input_ids"].size(1)
                    completions.extend(
                        tokenizer.batch_decode(completion_ids[:, prompt_length:], skip_special_tokens=True)
                    )

    # Aggregate results
    results = {}
    gridConverter = GridConverter(params["use_barc_format"])
    for completion in completions:
        parsed_completion = gridConverter.decode(completion)
        parsed_completion = completion if parsed_completion.size == 0 else parsed_completion
        # Get black-box score if completion is valid otherwise 0
        score = trainer.get_bb_score(solution, parsed_completion) if isinstance(parsed_completion, np.ndarray) else 0
        parsed_completion = str(parsed_completion)
        # Track completions and their scores
        results[parsed_completion] = results.get(parsed_completion, {"score": score, "count": 0})
        results[parsed_completion]["count"] += 1

    data["test_samples"] = [
        {"completion": x[0], "score": x[1]["score"], "count": x[1]["count"]}
        for x in sorted(results.items(), key=lambda x: x[1]["count"], reverse=True)
    ]

    data["test_majority"] = data["test_samples"][0]
    data["test_best"] = sorted(data["test_samples"], key=lambda x: x["score"], reverse=True)[0]
    data["test_solved_majority"] = data["test_samples"][0]["score"] == 1.0
    data["test_solved_majority_pass2"] = data["test_solved_majority"] or (
        data["test_samples"][1]["score"] == 1.0 if len(data["test_samples"]) > 1 else False
    )
    data["test_solved_oracle"] = data["test_best"]["score"] == 1.0

    # Save these logs to wandb
    wandb.run.summary["test/solved_majority"] = data["test_solved_majority"]
    wandb.run.summary["test/score_majority"] = data["test_majority"]["score"]
    wandb.run.summary["test/solved_majority_pass2"] = data["test_solved_majority_pass2"]
    wandb.run.summary["test/score_majority_pass2"] = max(
        data["test_majority"]["score"], data["test_samples"][1]["score"] if len(data["test_samples"]) > 1 else 0
    )
    wandb.run.summary["test/solved_oracle"] = data["test_solved_oracle"]
    wandb.run.summary["test/best_score"] = data["test_best"]["score"]
    wandb.run.summary["test/best_completion"] = data["test_best"]

    print(f"TEST SOLVED @ pass1: {data['test_solved_majority']}")
    print(f"TEST SOLVED @ pass2: {data['test_solved_majority_pass2']}")
    print(f"TEST SOLVED ORACLE: {data['test_solved_oracle']}")
    if not data["test_solved_majority_pass2"]:
        print(f"BEST COMPLETION: {data['test_best']}")

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
