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
    max_seq_length=1024,
)


def create_prompt(target, num_guesses, context=[], strat="random_sample"):
    system_prompt = """\
You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no \
additional conversation. All your responses should be in JSON format, i.e. '{key: value}', where the key is always \
"response" and the value can be a string, int, list, or dict, depending on the context."""
    user_prompt = """\
Your task is to find the optimal drug molecule that has both a high druglikeness (QED) as well as a strong binding \
affinity (vina) with the protein %s. For docking, lower is better (less than −10 is considered good) and for \
druglikeness, 1 is the best and 0 is the worst (greater than 0.8 is considered good). While both properties are \
important, the docking score is 10 times as important as the druglikeness score. If you propose an invalid molecule or \
make a repeat guess, you will get no score, so stick to valid SMILES strings.

Now, guess exactly n=%s new molecule(s).

(Note: give only a list of SMILES string(s) in the provided JSON format, e.g. {"response": ["SMILES1", "SMILES2", ...]})"""

    system_prompt_for_neighbors = """\
You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no \
additional conversation. All your responses should be in JSON format, i.e. '{key: value}', where the key is always \
"response" and the value can be a string, int, list, or dict, depending on the context."""
    user_prompt_for_neighbors = """\
Your task is to find the optimal drug molecule that has both a high druglikeness (QED) as well as a strong binding \
affinity (vina) with the protein %s. For docking, lower is better (less than −10 is considered good) and for \
druglikeness, 1 is the best and 0 is the worst (greater than 0.8 is considered good). While both properties are \
important, the docking score is 10 times as important as the druglikeness score. If you propose an invalid molecule or \
make a repeat guess, you will get no score, so stick to valid SMILES strings!

Here is my guess for a molecule:
SMILES: %s

Now, guess exactly n=%s new variation(s) of my molecule that could improve the scores to reach the optimal molecule.

(Note: give only a list of SMILES string(s) in the provided JSON format, e.g. {"response": ["SMILES1", "SMILES2", ...]})"""
    user_prompt_for_opro = """\
Your task is to find the optimal drug molecule that has both a high druglikeness (QED) as well as a strong binding \
affinity (vina) with the protein %s. For docking, lower is better (less than −10 is considered good) and for \
druglikeness, 1 is the best and 0 is the worst (greater than 0.8 is considered good). While both properties are \
important, the docking score is 10 times as important as the druglikeness score. If you propose an invalid molecule or \
make a repeat guess, you will get no score, so stick to valid SMILES strings!

Here are your top previous guesses (from worst to best):

%s

Now, guess exactly n=%s new molecules that could score higher than your previous guesses.

(Note: give only a list of SMILES string(s) in the provided JSON format, e.g. {"response": ["SMILES1", "SMILES2", ...]})"""

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


def dock_mol(smile, protein, score_range):
    score, qed, vina = 0, None, None
    try:
        mol = Chem.MolFromSmiles(smile)
        qed = QED.qed(mol)
        loaded_target = load_target(protein)
        # vina, aux = loaded_target.dock(smile, num_cpus=10)
        vina, aux = func_timeout(30, loaded_target.dock, args=(smile,), kwargs={"num_cpus": 2})

        # Normalize score between 0 and 1
        diff = score_range[1] - score_range[0]
        score = 1 - ((vina + (1 - qed) - score_range[0]) / diff)
        # print("Guess:", attempt)
        # print("QED:", qed)
        # print("VINA:", vina)
        # print("Score:", score)
        # Clip score between 0 and 1 just in case
        score = max(0, min(1, score))
    except:
        score = 0
    return score, qed, vina


def get_bb_score(solution, attempt) -> float:
    """
    Compute black-box score
    """
    score, qed, vina = 0, None, None
    timeout = 30
    score_range = [-12, -2]
    try:
        # score, qed, vina = func_timeout(timeout, dock_mol, args=(attempt, solution, score_range))
        score, qed, vina = dock_mol(attempt, solution, score_range)
    except:
        print("Timed out")

    return [score, qed, vina]


def run_dockstring(
    target,
    seed,
    top_k,
    logfile,
    strat,
    iterations=40,
    num_guesses=1,
    batch_size=5,
    alpha=0.2,
    beta=0.8,
    archive_size=10,
    migration_interval=40,
):
    past_guesses = []
    seen_guesses = set()
    guesses_per_iter = num_guesses * batch_size

    for i in tqdm(range(iterations)):
        best_guess = heapq.nlargest(top_k, past_guesses)[::-1]
        if strat[:10] == "random_top":
            random.shuffle(best_guess)
            best_guess = [random.choice(best_guess)]

        strategy = "random_sample" if len(best_guess) == 0 else strat
        best_guess = [x[1] for x in best_guess]
        prompts = [create_prompt(target, num_guesses, best_guess, strat=strategy)]
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
                        guesses += [json.loads(completion)["response"][0].strip()]
                    except Exception as e:
                        pass
                if len(guesses) < guesses_per_iter:
                    retries += 1

            guesses = guesses[:guesses_per_iter]
            # with Pool(processes=5) as pool:
            #     scores = pool.starmap(get_bb_score, [(target, guess) for guess in guesses])
            scores = [get_bb_score(target, guess) for guess in guesses]
            print(scores)
            while len(scores) < guesses_per_iter:
                scores.append([0, None, None])
                guesses.append("")

            for res, score in zip(guesses, scores):
                if res not in seen_guesses:
                    heapq.heappush(past_guesses, (score[0], res))
                    seen_guesses.add(res)

            # log output
            response = (
                {
                    "Iteration": i,
                    "completions_scores": [
                        [completion, {"score": score[0], "qed": score[1], "vina": score[2]}]
                        for completion, score in zip(completions, scores)
                    ],
                },
            )
            with open(logfile, "r") as file:
                data = json.load(file)
                data["guesses"].append(response)
            with open(logfile, "w") as file:
                json.dump(data, file, indent=4)


targets = [
    "IGF1R",
    "JAK2",
    "KIT",
    "LCK",
    "MAPK14",
    "MAPKAPK2",
    "MET",
    "PTK2",
    "PTPN1",
    "SRC",
    "ABL1",
    "AKT1",
    "AKT2",
    "CDK2",
    "CSF1R",
    "EGFR",
    "KDR",
    "MAPK1",
    "FGFR1",
    "ROCK1",
    "MAP2K1",
    "PLK1",
    "HSD11B1",
    "PARP1",
    "PDE5A",
    "PTGS2",
    "ACHE",
    "MAOB",
    "CA2",
    "GBA",
    "HMGCR",
    "NOS1",
    "REN",
    "DHFR",
    "ESR1",
    "ESR2",
    "NR3C1",
    "PGR",
    "PPARA",
    "PPARD",
    "PPARG",
    "AR",
    "THRB",
    "ADAM17",
    "F10",
    "F2",
    "BACE1",
    "CASP3",
    "MMP13",
    "DPP4",
    "ADRB1",
    "ADRB2",
    "DRD2",
    "DRD3",
    "ADORA2A",
    "CYP2C9",
    "CYP3A4",
    "HSP90AA1",
]
seeds = [1, 2, 3]


def run_experiment(seeds, strat, alpha=0.2, beta=0.8, archive_size=10, migration_interval=40):
    for seed in seeds:
        for target in targets:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logdir = f"logs/dockstring/{strat}/{target}"
            os.makedirs(logdir, exist_ok=True)
            logfile = f"{logdir}/{timestamp}.log"
            with open(logfile, "w") as file:
                file.write(json.dumps({"guesses": []}, indent=2))

            if strat == "random_sample":
                run_dockstring(target, seed, 0, logfile, strat)
            if strat == "ns_1k":
                run_dockstring(target, seed, 1, logfile, strat)
            if strat == "ns_10k":
                run_dockstring(target, seed, 10, logfile, strat)
            if strat == "opro_1k":
                run_dockstring(target, seed, 1, logfile, strat)
            if strat == "opro_5k":
                run_dockstring(target, seed, 5, logfile, strat)
            if strat == "opro_10k":
                run_dockstring(target, seed, 10, logfile, strat)
            if strat == "random_ns":
                run_dockstring(target, seed, 1, logfile, strat)
            if strat == "random_top3_ns":
                run_dockstring(target, seed, 3, logfile, strat)


parser = argparse.ArgumentParser()
parser.add_argument("--strat", type=str, default="random_sample")
args = vars(parser.parse_args())

run_experiment(seeds, args["strat"])
