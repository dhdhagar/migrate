from abc import ABC, abstractmethod
from typing import Any
import json


class Task(ABC):
    def __init__(self, **kwargs):
        self.task_name = kwargs.get("task")
        self.target = kwargs.get("target")
        self.migrate_alpha = kwargs.get("migrate_alpha")
        self.migrate_beta = kwargs.get("migrate_beta")
        self.migrate_gamma = kwargs.get("migrate_gamma")
        self.batch_size = kwargs.get("batch_size")
        self.num_guesses = kwargs.get("num_guesses")

    @abstractmethod
    def decode_response(self, response: str) -> Any:
        """
        Decode LLM response into desired format
        """

    @abstractmethod
    def encode_representation(self, representation) -> str:
        """
        Encode task's desired format into string
        """
        pass

    @abstractmethod
    def get_bb_score(self, solution, attempt) -> Any:
        """
        Compute black-box score
        """
        pass

    @abstractmethod
    def clean_completion(self, completion) -> list:
        pass

    @abstractmethod
    def evaluate_completion(self, completion, inputs) -> list:
        pass

    def log_repsonse(self, completions, bb_scores, iteration, logfile, inputs):
        """
        Additional logging
        """
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
