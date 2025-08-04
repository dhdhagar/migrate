from utils.base_task import Task
from utils.semantle_utils.semantle_prompts import SemantlePrompts
from typing import Any
import numpy as np
import json
import re
from sentence_transformers import SentenceTransformer
import random


def extract_json(response):
    match = re.findall(r"{.*}", response, re.DOTALL)
    if len(match) > 0 and "response" in match[0]:
        return match[0]
    else:
        return response


def cosine_similarity(x1, x2):
    x1_norm = np.linalg.norm(x1)
    x2_norm = np.linalg.norm(x2)
    return float(np.dot(x1, x2) / (x1_norm * x2_norm))


class SemantleTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_guesses = kwargs.get("num_guesses")
        self.prompts = SemantlePrompts(self)
        self.embedder = SentenceTransformer(kwargs.get("embedder"))

    def decode_response(self, response: str) -> Any:
        """
        Decode LLM response into desired format
        """
        try:
            return [x.strip().lower() for x in json.loads(extract_json(response))["response"]]
        except Exception as _:
            return []

    def encode_representation(self, representation) -> str:
        """
        Encode task's desired format into string
        """
        return ""

    def get_bb_score(self, solution, attempt) -> float:
        """
        Compute black-box score
        """
        if solution == attempt:
            return 1.0
        texts = [f"What is a {solution}?", f"What is a {attempt}?"]
        embeddings = self.embedder.encode(texts)
        return cosine_similarity(embeddings[0], embeddings[1])

    def clean_completion(self, completion) -> list:
        guesses = self.decode_response(completion)
        responses = []
        for i in range(len(guesses)):
            if i < len(guesses):
                responses.append(guesses[i])
            else:
                responses.append(completion)
        return responses

    def evaluate_completion(self, completion, inputs) -> list:
        guesses = self.decode_response(completion)
        scores = []
        for i in range(len(guesses)):
            scores.append(self.get_bb_score(inputs[0]["solution"][0], guesses[i]))
        return scores

    def create_batch_completions(self, guesses, scores, past_guesses):
        n = 10
        pairs = list(zip(guesses, scores))
        if len(pairs) > n:
            pairs = random.sample(pairs, k=n)
        elif len(pairs) < n:
            pairs += random.choices(pairs, k=n - len(pairs))
        guesses, scores = zip(*pairs)

        group_size = 2

        guesses, scores = zip(*sorted(zip(guesses, scores), key=lambda x: x[1], reverse=True))
        completions, rewards = [], []
        for i in range(self.batch_size):
            batch = list(guesses[i * group_size : i * group_size + group_size])
            random.shuffle(batch)
            if all(" " not in word for word in batch):
                completions.append(json.dumps({"response": batch}))
            else:
                if i == 0:
                    completions.append(json.dumps({"response": [guesses[0]] * group_size}))
                else:
                    completions.append(guesses[i * group_size])
            rewards.append(np.max(scores[i * group_size : i * group_size + group_size]).item())
        return completions, rewards

    def evaluate_and_log(self, completions, inputs, iteration, logfile):
        responses, bb_scores = [], []
        for i, completion in enumerate(completions):
            response = self.clean_completion(completion)
            bb_score = self.evaluate_completion(completion, inputs)
            responses += response
            bb_scores += bb_score

        self.log_repsonse(completions, bb_scores, iteration, logfile, inputs)

        num_completions = len(responses)
        while len(responses) < self.num_guesses:
            if num_completions > 0:
                idx = random.randint(0, num_completions - 1)
                responses.append(responses[idx])
                bb_scores.append(bb_scores[idx])
            else:
                idx = random.randint(0, len(completions) - 1)
                responses.append(completions[idx])
                bb_scores.append(0)

        return responses, bb_scores

    def log_repsonse(self, completions, bb_scores, iteration, logfile, inputs):
        response = (
            {
                "Iteration": iteration,
                "completions_scores": list(zip(completions, [bb_scores])),
                "problem": str(inputs[0]["problem"]),
                "solution": str(inputs[0]["solution"]),
            },
        )
        with open(logfile, "r") as file:
            data = json.load(file)
            data["guesses"].append(response)
        with open(logfile, "w") as file:
            json.dump(data, file, indent=4)
