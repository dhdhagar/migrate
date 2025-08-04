from utils.base_task import Task
from utils.arc_utils.execute import execute_transformation
from utils.arc_utils.utils import parse_code, hamming_distance
from utils.arc_utils.induction_prompts import ARC_InductionPrompts
import numpy as np
import json


class ARCInductionTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompts = ARC_InductionPrompts(self)
        self.color_map = {
            0: "Black",
            1: "Blue",
            2: "Red",
            3: "Green",
            4: "Yellow",
            5: "Gray",
            6: "Pink",
            7: "Orange",
            8: "Purple",
            9: "Brown",
        }

    def decode_response(self, response, input_grid: np.ndarray = None, verbose: bool = False):
        """
        Decode LLM response (program) and input grid into an numpy array output
        """
        parsed_codes = parse_code(response)
        if parsed_codes:
            code = parsed_codes[-1]
            grid = execute_transformation(code, input_grid, function_name="transform", verbose=verbose)
            return grid if isinstance(grid, np.ndarray) else np.array([[]])
        else:
            return np.array([[]])

    def encode_representation(self, representation, include_prefix: bool = False) -> str:
        """
        Encode numpy grid or program representation into expected generated output
        """
        if isinstance(representation, np.ndarray):
            output = ""
            for i in range(representation.shape[0]):
                for j in range(representation.shape[1]):
                    output += self.color_map[representation[i][j]] + " "
                output = output[:-1] + "\n"
            if include_prefix:
                return f"The output grid for the test input grid is:\n\n```\n{output[:-1]}\n```"
            return output[:-1]
        else:
            parsed_codes = parse_code(representation)
            if parsed_codes:
                completion_prefix = """\
Let's solve this puzzle using Python code with the common library functions. We'll first reason about the problem and \
then write the code to solve it. The `transform` function will take the input grid and return the output grid. Here is \
the Python code with the comments describing how to solve the problem:"""
                completion = completion_prefix + f"\n```{parsed_codes[-1]}```"
                return completion
            else:
                return representation

    def get_bb_score(self, solution, attempt):
        """
        Compute black-box score
        """
        return 1 - hamming_distance(solution, attempt)

    def evaluate_completion(self, completion, inputs) -> list:
        train_problems = [np.array(x) for x in inputs[0]["problem"]]
        train_solutions = [np.array(x) for x in inputs[0]["solution"]]
        guesses = [self.decode_response(completion, input_grid=problem) for problem in train_problems]
        bb_scores = [np.mean([self.get_bb_score(solution, guess) for solution, guess in zip(train_solutions, guesses)])]
        return bb_scores

    def clean_completion(self, completion) -> list:
        return [self.encode_representation(completion, include_prefix=True)]

    def evaluate_and_log(self, completions, inputs, iteration, logfile):
        responses, bb_scores = [], []
        for i, completion in enumerate(completions):
            response = self.clean_completion(completion)
            bb_score = self.evaluate_completion(completion, inputs)
            responses += response
            bb_scores += bb_score

        self.log_repsonse(completions, bb_scores, iteration, logfile, inputs)

        return responses, bb_scores
