import json
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
        plt.plot(self.similarities)
        plt.plot(np.maximum.accumulate(self.similarities))
        plt.ylabel("Cosine Similarity")
        plt.xlabel("Global Step")
        plt.savefig("scores.png")
        plt.show()

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
        self.similarities.append(np.max(sims))
        return out


if __name__ == "__main__":
    # model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    # model_name = "Qwen/Qwen2-0.5B-Instruct"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"

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
    judge = SimPairJudge("computer", "princeton-nlp/sup-simcse-roberta-large")
    # judge = PairRMJudge()
    num_guesses = 5
    dataset = [
        {
            "prompt": [
                {
                    "content": f'Your task is to guess a hidden word from the English dictionary. Stick to proper, single-word English words. Now, guess exactly n={num_guesses} new word(s) that could be the hidden word. Be creative! (Note: give only a list of word(s) in the provided JSON format, e.g. {{"response": ["word1", "word2",...]}})',
                    "role": "user",
                }
            ]
        }
        for _ in range(500)
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
    trainer = SemantleOnlineDPOTrainer(
        model=model,
        judge=judge,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        target="computer",
        num_guesses=num_guesses,
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
