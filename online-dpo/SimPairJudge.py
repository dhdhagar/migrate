import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
from trl import BasePairwiseJudge

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Only a placeholder, pairs are created within the trainer class now
class SimPairJudge(BasePairwiseJudge):
    def __init__(self, target, embedder):
        super(SimPairJudge, self).__init__()
        self.target = target
        self.model_sim = AutoModel.from_pretrained(embedder).to(DEVICE)
        self.tokenizer_sim = AutoTokenizer.from_pretrained(embedder)
        self.scores = {target: 1.0}

    # TODO: Move this to the trainer class
    def get_sim(self, x1, x2):
        texts = [f"What is a {x1}?", f"What is a {x2}?"]
        inputs = self.tokenizer_sim(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            embeddings = self.model_sim(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        cosine_sim = 1 - cosine(embeddings[0].cpu(), embeddings[1].cpu())
        return cosine_sim

    def judge(self, prompts, completions, shuffle_order=False):
        return [0]
