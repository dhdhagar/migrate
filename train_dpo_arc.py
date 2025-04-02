from unsloth import FastLanguageModel, PatchFastRL  # Needs to be first import

PatchFastRL("DPO", FastLanguageModel)

import os
import time
from datetime import datetime
import argparse
import json
import torch
from trl import OnlineDPOConfig, maybe_apply_chat_template

from arc_utils.utils import parse_response, hamming_distance

from trainers.TTT_DPOTrainer import TTT_DPOTrainer
import prompts as prompts_getter
import numpy as np
from typing import List
import wandb
from vllm import SamplingParams

from accelerate.utils import broadcast_object_list, gather_object

from trl.models.utils import unwrap_model_for_generation

from trl import BasePairwiseJudge

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
    parser.add_argument("--strategy", type=str, default="Greedy_Gold")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--related", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--task", type=str, default="semantle")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--online_temperature", type=float, default=1.0)
    parser.add_argument("--online_max_completion_length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.01)
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
    parser.add_argument("--min_training_size", type=int, default=80)
    parser.add_argument("--max_training_size", type=int, default=160)
    parser.add_argument("--max_validation_size", type=int, default=64)
    parser.add_argument("--max_test_size", type=int, default=64)
    parser.add_argument("--validation_interval", type=int, default=5)
    parser.add_argument("--use_vllm", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    return args


def create_dataset(params):
    if params["task"] != "arc":
        return [
            {"prompt": prompts_getter.get_prompt(params["task"], params["num_guesses"], params["target"])}
            for _ in range(params["steps"])
        ]
    else:
        return prompts_getter.get_arc_datasets(
            params["target"],
            **{k: v for k, v in params.items() if k in prompts_getter.get_arc_datasets.__code__.co_varnames},
        )


def setup_logging(params):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = f'logs/{params["strategy"]}/{params["target"]}'
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{timestamp}.log"
    wandb_id = f'{params["strategy"]}-{params["target"]}-{timestamp}'
    with open(logfile, "w") as file:
        file.write(json.dumps({"params": params, "guesses": [], "validation": [], "final_sample": []}, indent=2))
    return logdir, logfile, wandb_id


def setup_model(params):
    model, tokenizer = FastLanguageModel.from_pretrained(
        params["model"],
        device_map="auto",
        max_lora_rank=params["lora_rank"],
        gpu_memory_utilization=0.8,
        fast_inference=params["use_vllm"],
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


class HammingJudge(BasePairwiseJudge):
    def __init__(self, logfile):
        super(HammingJudge, self).__init__()
        self.logfile = logfile
        self.iteration = 0
        self.val_iteration = 0

    def log_response(self, responses, validation):
        with open(self.logfile, "r") as file:
            data = json.load(file)
            if validation:
                data["validation"].append({f"Iteration: {self.val_iteration}": responses})
                self.val_iteration += 1
            else:
                data["guesses"].append({f"Iteration: {self.iteration}": responses})
                self.iteration += 1

                train_rewards = []
                for res in responses[: params["batch_size"]]:
                    train_rewards.append(res[1][0])
                    train_rewards.append(res[1][1])
                print(train_rewards)

                wandb.log({"training/mean_rewards": np.mean(train_rewards)})
                wandb.log({"training/max_rewards": np.max(train_rewards)})
        with open(self.logfile, "w") as file:
            json.dump(data, file, indent=4)

    def get_bb_score(self, completion1, completion2):
        return 1 - hamming_distance(completion1, completion2)

    def judge(self, prompts, completions, shuffle_order=False, solutions=[]):
        parsed_completions = [(parse_response(x[0]), parse_response(x[1])) for x in completions]
        scores = [
            (self.get_bb_score(sol, x[0]), self.get_bb_score(sol, x[1]))
            for sol, x in zip(solutions, parsed_completions)
        ]
        preferences = [int(x[0] < x[1]) for x in scores]
        print("COMPLETTIONS", completions)
        print("SCORES", scores)
        self.log_response([(x, y) for x, y in zip(completions, scores)], validation=False)
        return preferences


def main(params):
    training_dataset, validation_dataset, test_dataset = create_dataset(params)
    print(len(training_dataset))
    # dataset = [dataset[0].copy() for i in range(100)]

    # model, tokenizer, peft_config = setup_model(params)
    model, tokenizer = setup_model(params)

    logdir, logfile, wandb_id = setup_logging(params)

    training_args = OnlineDPOConfig(
        output_dir="DPO",
        logging_steps=1,
        fp16=False,
        bf16=True,
        learning_rate=params["learning_rate"],
        max_new_tokens=params["online_max_completion_length"],
        max_length=2 * params["online_max_completion_length"],
        temperature=params["online_temperature"],
        beta=params["beta"],
        use_vllm=params["use_vllm"],
        report_to="wandb",
        run_name=wandb_id,
        per_device_train_batch_size=params["batch_size"],
        num_train_epochs=1,
        gradient_accumulation_steps=1,
    )
    trainer = TTT_DPOTrainer(
        model=model,
        processing_class=tokenizer,
        judge=HammingJudge(logfile=logfile),
        args=training_args,
        train_dataset=training_dataset,
        validation_example=validation_dataset,
        validation_interval=params["validation_interval"],
        logfile=logfile,
        target=params["target"],
        strategy=params["strategy"],
        sample_related=params["related"],
        task=params["task"],
        # generation_args={},
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
                prompt_inputs = trainer.processing_class(
                    prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
                )
                prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
                # Regular generation path
                with unwrap_model_for_generation(trainer.model, trainer.accelerator) as unwrapped_model:
                    completion_ids = unwrapped_model.generate(
                        input_ids=prompt_ids.to(DEVICE),
                        attention_mask=prompt_mask.to(DEVICE),
                        do_sample=False,
                        max_new_tokens=512,
                    )
                    prompt_length = prompt_inputs["input_ids"].size(1)
                    completions = trainer.processing_class.batch_decode(
                        completion_ids[:, prompt_length:], skip_special_tokens=True
                    )

        # Aggregate results
        results = {}
        for completion in completions:
            parsed_completion = parse_response(completion)
            parsed_completion = completion if parsed_completion.size == 0 else parsed_completion
            # Get black-box score if completion is valid otherwise 0
            score = (
                trainer.get_bb_score(solution, parsed_completion) if isinstance(parsed_completion, np.ndarray) else 0
            )
            parsed_completion = str(parsed_completion)
            # Track completions and their scores
            results[parsed_completion] = results.get(parsed_completion, {"score": score, "count": 0})
            results[parsed_completion]["count"] += 1

        data["final_sample"] = [
            {"completion": x[0], "score": x[1]["score"], "count": x[1]["count"]} for x in results.items()
        ]
        with open(logfile, "w") as file:
            json.dump(data, file, indent=4)

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
