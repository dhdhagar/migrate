from utils.base_task import Task
from utils.molecule_utils.molecule_prompts import MoleculePrompts
from typing import Any
import numpy as np
import json
import re
from sentence_transformers import SentenceTransformer
import random
from rdkit import Chem
from rdkit.Chem import QED
from dockstring import load_target
from func_timeout import func_timeout, FunctionTimedOut
from multiprocessing import Pool
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def extract_json(response):
    match = re.findall(r"{.*}", response, re.DOTALL)
    if len(match) > 0 and "response" in match[0]:
        return match[0]
    else:
        return response


def dock_mol(smile, protein, score_range):
    score, qed, vina = 0, None, None
    try:
        mol = Chem.MolFromSmiles(smile)
        qed = QED.qed(mol)
        loaded_target = load_target(protein)
        vina, aux = loaded_target.dock(smile, num_cpus=2)

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


class MoleculeTask(Task):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        # self.num_guesses = num_guesses
        self.prompts = MoleculePrompts(self)
        self.score_range = [-12, -2]

    def decode_response(self, response: str) -> Any:
        """
        Decode LLM response into desired format
        """
        try:
            return json.loads(extract_json(response))["response"][0].strip()
        except Exception as _:
            return response

    def encode_representation(self, representation) -> str:
        """
        Encode task's desired format into string
        """
        return ""

    def get_bb_score(self, solution, attempt):
        """
        Compute black-box score
        """
        score, qed, vina = 0, None, None
        timeout = 90
        try:
            score, qed, vina = func_timeout(timeout, dock_mol, args=(attempt, solution, self.score_range))
        except:
            print("Timed out")

        return [score, qed, vina]

    def clean_completion(self, completion) -> list:
        try:
            smile = json.loads(extract_json(completion))["response"][0].strip()
            return [json.dumps({"response": [smile]})]
        except Exception as _:
            return [completion]

    def evaluate_completion(self, completion, inputs) -> list:
        try:
            smile = json.loads(extract_json(completion))["response"][0].strip()
            return [self.get_bb_score(self.target, smile)]
        except Exception as _:
            return [[0, None, None]]

    def evaluate_and_log(self, completions, inputs, iteration, logfile):
        responses, bb_scores = [], []
        # for i, completion in enumerate(completions):
        #     response = self.clean_completion(completion)
        #     bb_score = self.evaluate_completion(completion, inputs)
        #     responses += response
        #     bb_scores += bb_score
        #
        # Multiprocessing verison
        for i, completion in enumerate(completions):
            response = self.clean_completion(completion)
            responses += response
        with Pool(processes=5) as pool:
            bb_scores = pool.starmap(self.evaluate_completion, [(c, inputs) for c in completions])
        bb_scores = [x[0] for x in bb_scores]

        self.log_repsonse(completions, bb_scores, iteration, logfile, inputs)
        bb_scores = [x[0] for x in bb_scores]

        return responses, bb_scores

    def log_repsonse(self, completions, bb_scores, iteration, logfile, inputs):
        bb_scores = [{"score": x[0], "qed": x[1], "vina": x[2]} for x in bb_scores]
        response = (
            {
                "Iteration": iteration,
                "completions_scores": list(zip(completions, bb_scores)),
                "problem": str(inputs[0]["problem"]),
                "solution": str(inputs[0]["solution"]),
            },
        )
        with open(logfile, "r") as file:
            data = json.load(file)
            data["guesses"].append(response)
        with open(logfile, "w") as file:
            json.dump(data, file, indent=4)
