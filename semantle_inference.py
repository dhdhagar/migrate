from unsloth import FastLanguageModel
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from trl import maybe_apply_chat_template
from sentence_transformers import SentenceTransformer
import heapq
from tqdm import tqdm
from datetime import datetime
import os
import glob
import random
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from rdkit import Chem
from rdkit.Chem import QED
from dockstring import load_target
from func_timeout import func_timeout
from multiprocessing import Pool
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "meta-llama/Llama-3.2-3B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    device_map="auto",
    gpu_memory_utilization=0.8,
    fast_inference=False,
    max_seq_length=512,
)
embedder_name = "princeton-nlp/sup-simcse-roberta-large"
embedder = SentenceTransformer(embedder_name, trust_remote_code=True)


def create_prompt(num_guesses, context=[], strat="random_sample"):
    system_prompt = """\
You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no \
additional conversation. All your responses should be in JSON format, i.e. '{key: value}', where the key is always \
"response" and the value can be a string, int, list, or dict, depending on the context."""
    user_prompt = """\
Your task is to guess a hidden word from the English dictionary. Stick to proper, single-word English words. Now, guess \
exactly n=%s new word(s) that could be the hidden word. Be creative! (Note: give only a list of word(s) in the provided \
JSON format, e.g. {"response": ["word1", "word2",...]})"""

    system_prompt_for_neighbors = """\
You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no \
additional conversation. All your responses should be in JSON format, i.e. '{key: value}', where the key is always \
"response" and the value can be a string, int, list, or dict, depending on the context."""
    user_prompt_for_neighbors = """\
Your task is to guess words related to a word from the English dictionary. Stick to proper, single-word English words. \
Now, guess exactly n=%s new word(s) that could be related to the word(s):\n\n%s\n\nBe creative! (Note: give only a list of \
word(s) in the provided JSON format, e.g. {"response": ["word1", "word2",...]})"""

    system_prompt_for_opro = """\
You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no \
additional conversation. All your responses should be in JSON format, i.e. '{key: value}', where the key is always \
"response" and the value can be a string, int, list, or dict, depending on the context."""
    user_prompt_for_opro = """\
Your task is to guess a hidden test word from the English dictionary. Stick to proper, single-word English words. Use \
the below series of your previous guesses (in increasing order of their similarity to the hidden word in terms of their \
*meaning*) to make a new guess. Your new guess should not have been made before and should score higher than your \
previous guesses.\n\nHere are your top previous guesses (from worst to best):\n\n%s\n\nNow, guess exactly n=%s new word(s) that \
could give you a target score of 1.0. Be creative! (Note: give only a list of word(s) in the provided JSON format, \
e.g. {"response": ["word1", "word2",...]})"""

    if strat == "random_sample":
        return [
            {"content": system_prompt, "role": "system"},
            {"content": user_prompt % num_guesses, "role": "user"},
        ]
    elif "ns" in strat:
        return [
            {"content": system_prompt_for_neighbors, "role": "system"},
            {"content": user_prompt_for_neighbors % (num_guesses, "\n\n".join(context)), "role": "user"},
        ]
    elif "opro" in strat:
        return [
            {"content": system_prompt_for_opro, "role": "system"},
            {"content": user_prompt_for_opro % ("\n\n".join(context), num_guesses), "role": "user"},
        ]

    if strat == "random_sample":
        return [
            {"content": system_prompt, "role": "system"},
            {"content": user_prompt % (target, num_guesses), "role": "user"},
        ]
    elif "ns" in strat:
        return [
            {"content": system_prompt_for_neighbors, "role": "system"},
            {"content": user_prompt_for_neighbors % (target, "\n\n".join(context), num_guesses), "role": "user"},
        ]
    elif "opro" in strat:
        return [
            {"content": system_prompt_for_neighbors, "role": "system"},
            {
                "content": user_prompt_for_opro % (target, "SMILES: " + "\n\nSMILES: ".join(context), num_guesses),
                "role": "user",
            },
        ]


def cosine_similarity(x1, x2):
    x1_norm = np.linalg.norm(x1)
    x2_norm = np.linalg.norm(x2)
    return float(np.dot(x1, x2) / (x1_norm * x2_norm))


def get_bb_score(solution, attempt) -> float:
    """
    Compute black-box score
    """
    if solution == attempt:
        return 1.0
    texts = [f"What is a {solution}?", f"What is a {attempt}?"]
    embeddings = embedder.encode(texts)
    return cosine_similarity(embeddings[0], embeddings[1])


def run_semantle(
    target,
    seed,
    top_k,
    logfile,
    strat,
    warmstart_file,
    iterations=100,
    num_guesses=10,
    batch_size=1,
    alpha=0.2,
    beta=0.8,
    archive_size=10,
    migration_interval=40,
):
    past_guesses = []
    seen_guesses = set()
    guesses_per_iter = num_guesses * batch_size

    warmstarts = []
    with open(warmstart_file, "r") as file:
        warmstarts = json.load(file)[target][str(seed)]
        for word_score in warmstarts:
            heapq.heappush(past_guesses, (word_score[1], word_score[0]))
            seen_guesses.add(word_score[0])

    for i in tqdm(range(iterations)):
        best_guess = heapq.nlargest(top_k, past_guesses)[::-1]
        if strat[:10] == "random_top":
            random.shuffle(best_guess)
            best_guess = [random.choice(best_guess)]

        if strat == "random_ns":
            prompts = [
                create_prompt(int(num_guesses / 2), [f"Word: {x[1]}" for x in best_guess], strat="random_sample"),
                create_prompt(int(num_guesses / 2), [f"Word: {x[1]}" for x in best_guess], strat="ns"),
            ]
            prompts_text = [
                maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"] for prompt in prompts
            ] * batch_size
        else:
            prompts = [create_prompt(num_guesses, [f"Word: {x[1]}" for x in best_guess], strat=strat)]
            prompts_text = [
                maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"] for prompt in prompts
            ] * batch_size

            print(prompts_text[0])

        prompt_inputs = tokenizer(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)

        with torch.no_grad():
            retries = 0
            guesses = []
            alt_guesses = [[] for _ in range(4)]
            completions = []
            while retries < 3 and len(guesses) < guesses_per_iter:
                completion_ids = model.generate(
                    input_ids=prompt_ids.to(DEVICE),
                    attention_mask=prompt_mask.to(DEVICE),
                    do_sample=True,
                    temperature=1.0,
                    max_new_tokens=256,
                )
                completions = tokenizer.batch_decode(completion_ids[:, prompt_length:], skip_special_tokens=True)
                for j, completion in enumerate(completions):
                    try:
                        guesses += [x.strip().lower() for x in json.loads(completion)["response"]]
                        alt_guesses[j] += [x.strip().lower() for x in json.loads(completion)["response"]]
                        alt_guesses[j] = alt_guesses[j][:num_guesses]
                    except Exception as e:
                        pass
                if len(guesses) < guesses_per_iter:
                    retries += 1

            scores = [get_bb_score(target, guess) for guess in guesses]
            alt_scores = [[get_bb_score(target, guess) for guess in gs] for gs in alt_guesses]
            guesses = guesses[:guesses_per_iter]
            scores = scores[:guesses_per_iter]
            while len(scores) < guesses_per_iter:
                scores.append(0)
                guesses.append("")

            for res, score in zip(guesses, scores):
                if res not in seen_guesses:
                    heapq.heappush(past_guesses, (score, res))
                    seen_guesses.add(res)

            # log output
            response = (
                {
                    "Iteration": i,
                    "completions_scores": [completions, scores],
                },
            )
            with open(logfile, "r") as file:
                data = json.load(file)
                data["guesses"].append(response)
            with open(logfile, "w") as file:
                json.dump(data, file, indent=4)


targets = [
    "airbase",
    "birthstone",
    "cement",
    "computer",
    "filament",
    "machetes",
    "meatloaf",
    "mob",
    "polyethylene",
    "skillet",
]
seeds = [42, 43, 44, 45, 46]


def run_experiment(seeds, strat, warmstart_file, alpha=0.2, beta=0.8, archive_size=10, migration_interval=40):
    for seed in seeds:
        for target in targets:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logdir = f"logs/semantle/{strat}/{target}"
            os.makedirs(logdir, exist_ok=True)
            logfile = f"{logdir}/{timestamp}.log"
            with open(logfile, "w") as file:
                file.write(json.dumps({"guesses": []}, indent=2))

            if strat == "random_sample":
                run_semantle(target, seed, 0, logfile, strat, warmstart_file, batch_size=10, iterations=10)
            if strat == "ns_1k":
                run_semantle(target, seed, 1, logfile, strat, warmstart_file)
            if strat == "ns_3k":
                run_semantle(target, seed, 3, logfile, strat, warmstart_file)
            if strat == "opro_1k":
                run_semantle(target, seed, 1, logfile, strat, warmstart_file)
            if strat == "opro_10k":
                run_semantle(target, seed, 10, logfile, strat, warmstart_file)
            if strat == "random_ns":
                run_semantle(target, seed, 1, logfile, strat, warmstart_file)
            if strat == "random_top3_ns":
                run_semantle(target, seed, 3, logfile, strat, warmstart_file)


parser = argparse.ArgumentParser()
parser.add_argument("--strat", type=str, default="random_sample")
parser.add_argument("--semantle_warmstart_file", type=str, default="warmstart/warmstart-20.json")
args = vars(parser.parse_args())

run_experiment(seeds, args["strat"], args["semantle_warmstart_file"])
