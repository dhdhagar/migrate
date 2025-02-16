import os
import time
import argparse
import json
from datetime import datetime
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
import prompts as prompts_getter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--4bit", action="store_true", default=False)
    parser.add_argument("--target", "-t", type=str, default="computer")
    parser.add_argument("--strategy", "-s", type=str, default="random")
    parser.add_argument("--batch_size", "-b", type=int, default=10)
    parser.add_argument("--n_reps", "-n", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmstart", type=float, default=0)
    parser.add_argument("--g", type=int, default=5)
    parser.add_argument("--related", action="store_true", default=False)
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--task", type=str, default="semantle")
    args = parser.parse_args()
    return args


def create_dataset(params):
    return [{"prompt": prompts_getter.get_prompt(params["task"], params["batch_size"])} for _ in range(params["steps"])]


def setup_logging(params):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = f'logs/{params["strategy"]}/{params["date"]}/{params["target"]}'
    os.makedirs(logdir, exist_ok=True)
    logfile = f"{logdir}/{timestamp}.log"
    with open(logfile, "w") as file:
        json.dump({"Params": params, "Guesses": [], "Related": [], "Final_Sample": ""}, file, indent=4)
    return logfile


def setup_model(params):
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
        params["model"], device_map="auto", quantization_config=quant_config, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(params["model"])
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, peft_config


def main():
    args = parse_arguments()
    params = vars(args)

    dataset = create_dataset(params)

    model, tokenizer, peft_config = setup_model(params)

    judge = SimPairJudge(params["target"], "princeton-nlp/sup-simcse-roberta-large")

    logfile = setup_logging(params)

    training_args = OnlineDPOConfig(
        output_dir="OnlineDPO",
        logging_steps=10,
        fp16=False,
        bf16=True,
        use_cpu=True,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        temperature=0.9,
        max_new_tokens=int(32 + (5 * params["batch_size"])),
        beta=0.01,
        # disable_dropout=False,
    )
    trainer = SemantleOnlineDPOTrainer(
        model=model,
        judge=judge,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        target=params["target"],
        batch_size=params["batch_size"],
        logfile=logfile,
        strategy=params["strategy"],
        warmstart=params["warmstart"],
        g=params["g"],
        n_reps=params["n_reps"],
        sample_related=params["related"],
        task=params["task"],
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

    # Sample from final trained model and log
    sample_prompt = [{"prompt": prompts_getter.get_prompt(params["task"], 20)}]
    inputs = tokenizer.apply_chat_template(sample_prompt, tokenize=True, return_tensors="pt").to(DEVICE)
    output = model.generate(inputs, num_return_sequences=1, max_new_tokens=512, temperature=0.9)[0][len(inputs[0]) :]
    with open(logfile, "r") as file:
        data = json.load(file)
    data["Final_Sample"] = tokenizer.decode(output, skip_special_tokens=True)
    with open(logfile, "w") as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    main()
