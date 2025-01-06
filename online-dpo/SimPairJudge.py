import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
from trl import BasePairwiseJudge

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

            # sim0 = self.get_sim(completion[0], self.target)
            # sim1 = self.get_sim(completion[1], self.target)
            sims.append(sim0)
            # sims.append(sim1)
            print("================")
            print(sim0)
            print(completion[0])
            print("****************")
            print(sim1)
            print(completion[1])
            print("================")
            print(sim0, sim1)
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
