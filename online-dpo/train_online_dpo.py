import argparse
import json
from datetime import datetime
from datasets import load_dataset, Dataset
from trl import OnlineDPOConfig, BasePairwiseJudge
from SemantleOnlineDPOTrainer import SemantleOnlineDPOTrainer, OnlineDPOTrainerV2
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimPairJudge(BasePairwiseJudge):
    def __init__(
        self,
        target,
        embedder,
    ):
        super(SimPairJudge, self).__init__()
        self.target = target
        self.model_sim = AutoModel.from_pretrained(embedder).to(DEVICE)
        self.tokenizer_sim = AutoTokenizer.from_pretrained(embedder)
        self.similarities = []

    def plot_similarities(self):
        maxes = [np.max(x) for x in self.similarities]
        means = [np.mean(x) for x in self.similarities]
        plt.plot(maxes, label="batch max")
        plt.plot(means, label="batch mean")
        plt.plot(np.maximum.accumulate(maxes))
        plt.ylabel("Cosine Similarity")
        plt.xlabel("Global Step")
        plt.title(f"Semantle: {self.target}")
        plt.legend()
        plt.savefig(f"scores_{self.target}.png")

    def get_sim(self, x1, x2):
        texts = [f"What is a {x1}?", f"What is a {x2}?"]
        inputs = self.tokenizer_sim(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            embeddings = self.model_sim(
                **inputs, output_hidden_states=True, return_dict=True
            ).pooler_output
        cosine_sim = 1 - cosine(embeddings[0].cpu(), embeddings[1].cpu())
        return cosine_sim

    def judge(self, prompts, completions, shuffle_order=False):
        out = []
        sims = []
        best_sims = []
        for completion in completions:
            sim0 = 0
            sim1 = 0
            try:
                completion0 = json.loads(completion[0])["response"]
                for guess in completion0:
                    sim0 = max(sim0, self.get_sim(guess, self.target))
            except:
                sim0 = self.get_sim(completion[0], self.target)
            try:
                completion1 = json.loads(completion[1])["response"]
                for guess in completion1:
                    sim1 = max(sim1, self.get_sim(guess, self.target))
            except:
                sim1 = self.get_sim(completion[1], self.target)
            sims.append(sim0)
            sims.append(sim1)
            # print(sim0, sim1)
            out.append(sim0 < sim1)

            try:
                best_sims.append(
                    {
                        "word": json.loads(completion[0])["response"][0],
                        "sim": sim0,
                    }
                )
            except:
                pass
        self.similarities.append(sims)
        return out, best_sims


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", "-t", type=str, default="computer")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    params = vars(args)
    # model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    # model_name = "Qwen/Qwen2-0.5B-Instruct"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    #
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
        model_name,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    judge = SimPairJudge(target_word, "princeton-nlp/sup-simcse-roberta-large")
    # judge = PairRMJudge()
    num_guesses = 5
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
        for _ in range(200)
    ]
    train_dataset = Dataset.from_list(dataset)
    del dataset

    training_args = OnlineDPOConfig(
        output_dir=f"{model_name}-OnlineDPO",
        logging_steps=10,
        fp16=False,
        bf16=True,
        use_cpu=True,
        learning_rate=1e-5,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        logfile=f"logs/logfile_{timestamp}_{target_word}.log",
    )
    # trainer = OnlineDPOTrainerV2(
    #     model=model,
    #     judge=judge,
    #     args=training_args,
    #     processing_class=tokenizer,
    #     train_dataset=train_dataset,
    #     peft_config=peft_config,
    #     tokenizer=tokenizer,
    # )
    trainer.train()
    judge.plot_similarities()
