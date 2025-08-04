from unsloth import FastLanguageModel, PatchFastRL  # Needs to be first import

PatchFastRL("GRPO", FastLanguageModel)

import os
import time
from datetime import datetime
import json
import torch

from trl import GRPOConfig

from trainers.TTT_GRPOTrainer import TTT_GRPOTrainer, CustomProgressCallback
import wandb
from inference import run_induction_inference, run_transduction_inference
import numpy as np

from utils.arc_utils.induction_prompts import ARC_InductionPrompts
from utils.arc_utils.arc_induction_task import ARCInductionTask

from utils.semantle_utils.semantle_task import SemantleTask
from utils.molecule_utils.molecule_task import MoleculeTask
from arguments import parse_arguments

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataset(params, task, tokenizer=None):
    training_dataset, validation_dataset, test_dataset = task.prompts.build_dataset(
        params["target"],
        **{k: v for k, v in params.items() if k in task.prompts.build_dataset.__code__.co_varnames},
    )
    return training_dataset, validation_dataset, test_dataset


def setup_additional_logging(params, validation_data, test_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = f'logs/{params["task"]}/{params["strategy"]}/{params["target"]}'
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{timestamp}.log"
    wandb_id = f'{params["strategy"]}-{params["target"]}-{timestamp}'
    if params["wandb_prefix"] is not None:
        wandb_id = f"{params['wandb_prefix']}-{wandb_id}"
    init_data = {
        "params": params,
        "task": {
            "test": [{"problem": str(x["problem"]), "solution": str(x["solution"])} for x in test_data],
        },
        "guesses": [],
        "validation": [],
    }
    if len(validation_data) > 0:
        init_data["task"]["validaiton"] = {
            "problem": str(validation_data["problem"]),
            "solution": str(validation_data["solution"]),
        }
    with open(logfile, "w") as file:
        file.write(json.dumps(init_data, indent=2))
    return init_data, logdir, logfile, wandb_id


def setup_model(params):
    model, tokenizer = FastLanguageModel.from_pretrained(
        params["model"],
        device_map="auto",
        max_lora_rank=params["lora_rank"],
        gpu_memory_utilization=0.7,
        fast_inference=params["use_vllm"],
        max_seq_length=8192,
        full_finetuning=False,
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
        max_seq_length=8192,
        # use_gradient_checkpointing = "unsloth"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


# Placeholder reward func. We process and compute our own rewards in the trainer
def reward_len(completions, **kwargs):
    return 1.0


def main(params):
    task_map = {"arc": ARCInductionTask, "semantle": SemantleTask, "molecule": MoleculeTask}
    task_cls = task_map[params["task"]]
    task = task_cls(**params)

    # if params["task"] == "arc_induction":
    #     task = ARCInductionTask(params["task"], params["target"], params["migrate_gamma"], None, None)
    # elif params["task"] == "semantle":
    #     task = SemantleTask(
    #         params["task"], params["target"], params["migrate_gamma"], params["batch_size"], params["num_guesses"]
    #     )
    # elif params["task"] == "molecule":
    #     task = MoleculeTask(
    #         params["task"], params["target"], params["migrate_gamma"], params["batch_size"], params["num_guesses"]
    #     )

    model, tokenizer = setup_model(params)

    training_dataset, validation_dataset, test_dataset = create_dataset(params, task, tokenizer=tokenizer)

    data, logdir, logfile, wandb_id = setup_additional_logging(params, validation_dataset, test_dataset)
    wandb.init(project="ttt-arc", id=wandb_id)

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
        generation_batch_size=params["batch_size"],
        num_train_epochs=params["num_train_epochs"],
        temperature=params["online_temperature"],
        num_generations=params["num_generations"],
        max_completion_length=params["online_max_completion_length"],
        beta=params["beta"],
        report_to="wandb",
        run_name=wandb_id,
        max_prompt_length=params["max_seq_len"],
        disable_tqdm=True,  # To avoid double tqdm bars because of CustomProgressCallback
        seed=params["trl_seed"],
        epsilon_high=0.28,
        scale_rewards=False,
        num_iterations=params["num_iterations"],
        use_vllm=params["use_vllm"],
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.7,
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
        task=task,
        dataset_file=params["dataset_file"],
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
        sampling_strategy=params["migrate_sampling_strategy"],
        migrate_gamma=params["migrate_gamma"],
        migrate_alpha=params["migrate_alpha"],
        migrate_beta=params["migrate_beta"],
        opro_sampling=params["opro_sampling"],
        callbacks=[CustomProgressCallback()],
        use_barc_format=params["use_barc_format"],
        use_induction=params["use_induction"],
        warmstart_seed=params["warmstart_seed"],
        greedy_topk=params["greedy_topk"],
        include_scores=params["include_scores"],
        compute_entropy=params["compute_entropy"],
        semantle_warmstart_file=params["semantle_warmstart_file"],
    )

    start_time = time.time()
    if not params["only_inference"]:
        trainer.train()
        wandb.config.update(params)

    # Update wandb run with tags
    if params["wandb_tags"] is not None:
        wandb.run.tags = params["wandb_tags"].split(",")

    # Log training time
    train_time = time.time() - start_time
    with open(logfile, "r") as file:
        data = json.load(file)
        data["duration"] = train_time
    with open(logfile, "w") as file:
        json.dump(data, file, indent=2)

    print("\n==================\nRUNNING ON TEST\n==================")
    _model_for_inference = trainer.model
    if params["task"] == "arc":
        programs_from_training = None
        if not params["only_inference"]:
            programs_from_training = []
            with open(logfile, "r") as file:
                data = json.load(file)
                for batch in data["guesses"]:
                    for guess in batch[0]["completions_scores"]:
                        programs_from_training.append(guess[0])
        data.update(
            run_induction_inference(
                trainer,
                tokenizer,
                _model_for_inference,
                training_dataset,
                test_dataset,
                params,
                programs_from_training,
                task,
            )
        )

        # Save these logs to wandb
        wandb.run.summary["test/solved_majority"] = (
            [x["score"] == 1 for x in data["pass1"]] if data["pass1"] is not None else 0
        )
        wandb.run.summary["test/score_majority"] = (
            [x["score"] for x in data["pass1"]] if data["pass1"] is not None else 0
        )
        wandb.run.summary["test/solved_majority_pass2"] = (
            [x["score"] == 1 for x in data["pass2"]] if data["pass2"] is not None else 0
        )
        wandb.run.summary["test/score_majority_pass2"] = (
            [x["score"] for x in data["pass2"]] if data["pass2"] is not None else 0
        )
        wandb.run.summary["test/solved_oracle"] = [x == 1 for x in data["oracle"]["score"]]
        wandb.run.summary["test/best_score"] = data["oracle"]["score"]
        wandb.run.summary["test/best_completion"] = data["oracle"]["output"]

        print(f"TEST SOLVED ORACLE: {data['oracle']['score'] == 1}")
        if np.mean(data["oracle"]["score"]) < 1:
            print(f"BEST COMPLETION: {data['oracle']['output']}")

        with open(logfile, "w") as file:
            json.dump(data, file, indent=4)

    wandb.run.summary["log_fpath"] = logfile
    wandb.finish()

    # Save model
    if params["save_model"]:
        model.save_pretrained(logdir)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up W&B
    os.environ["WANDB_PROJECT"] = "ttt-arc"
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    # Get arguments
    args = parse_arguments()
    params = vars(args)

    main(params)
