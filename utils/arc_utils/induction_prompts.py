from utils.base_prompts import Prompts
from utils.arc_utils.utils import parse_code
import random
import json
import numpy as np
import itertools


class ARC_InductionPrompts(Prompts):
    def __init__(self, task, **kwargs):
        super().__init__(task)
        self.system_prompt_w_context = """\
You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming. \
Your task is to analyze puzzle and provide Python solutions."""
        self.system_prompt = "\
You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming. \
Your task is to provide Python solutions."
        self.user_prompt_w_context = """\
Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new \
test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells \
(colors) separated by spaces and rows by newlines. Here are the input and output grids for the reference examples:\n%s
Here is the input grid for the test example:\nInput:\n%s\n\nWrite a Python function `transform` that can convert any \
given input grid to its corresponding output grid based on the pattern observed in the reference examples."""
        self.user_prompt = """\
Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines. Here is the \
input grid for the test example:\nInput:\n%s\n\nWrite a Python function `transform` that can convert any given input \
grid to its corresponding output grid."""
        self.system_prompt_for_neighbors = """\
You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming. \
Your task is to analyze puzzle and provide Python solutions.

Given input-output grid pairs as reference examples, carefully observe the patterns to predict \
the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as \
strings, with cells (colors) separated by spaces and rows by newlines. 

Here are the input and output grids for the reference examples:
%s
The goal is to write a Python function `transform` that can convert any given input grid to its corresponding output \
grid based on the pattern observed in the reference examples."""
        self.user_prompt_for_neighbors = """\
Here is my guess for the function:
%s

Provide a variation of my guess that could be the correct answer."""
        self.user_prompt_for_opro = """\
Here is your previous guess for the function:
%s

Now, write a new program that could perform better than your previous guess."""

    def build_prompt(self, context_examples: list = [], test_input: np.ndarray | None = None, use_alternate=False):
        context_str = ""
        for i, example in enumerate(context_examples):
            input_str = self.task.encode_representation(np.array(example["input"]))
            output_str = self.task.encode_representation(np.array(example["output"]))
            context_str += f"Example {i+1}\nInput:\n{input_str}\n\nOutput:\n{output_str}\n\n\n"

        if len(context_examples) > 0:
            system_prompt = self.system_prompt_w_context
            user_prompt = (
                self.user_prompt
                if test_input is None
                else self.user_prompt_w_context % (context_str, self.task.encode_representation(test_input))
            )
        else:
            system_prompt = self.system_prompt
            user_prompt = self.user_prompt % (self.task.encode_representation(test_input))
        prompt = [
            {"content": system_prompt, "role": "system"},
            {"content": user_prompt, "role": "user"},
        ]
        return prompt

    def build_dataset(
        self,
        task_id,
        dataset_file,
        dataset_solutions_file,
        min_training_size=50,
        max_training_size=80,
        use_permutations=False,
        max_seq_len=2048,
    ):
        # Load Dataset
        with open(dataset_file, "r", encoding="utf-8") as file:
            data_problems = json.load(file)[task_id]
        with open(dataset_solutions_file, "r", encoding="utf-8") as file:
            data_solutions = json.load(file)[task_id]

        # Create training dataset
        training_dataset = []
        context_combinations = list(
            itertools.permutations(data_problems["train"], len(data_problems["train"]))
            if use_permutations
            else itertools.combinations(data_problems["train"], len(data_problems["train"]))
        )
        for context in context_combinations:
            for user_input in data_solutions:
                training_dataset.append(
                    {
                        "prompt": self.build_prompt(list(context), np.array(user_input)),
                        "problem": [x["input"] for x in data_problems["train"]],
                        "solution": [x["output"] for x in data_problems["train"]],
                    }
                )

        # Multiply the dataset until it's longer than the minimum length
        print("Training dataset size:", len(training_dataset))
        if len(training_dataset) < min_training_size:
            expanded_training_dataset = []
            for item in training_dataset:
                expanded_training_dataset.extend([item] * ((min_training_size // len(training_dataset)) + 1))
            training_dataset = expanded_training_dataset[-max_training_size:]
            print("Extending training dataset to:", len(training_dataset))
        else:
            # Sort by prompt length
            training_dataset = training_dataset[-max_training_size:]
            print("Clipping training dataset size to:", len(training_dataset))
        random.shuffle(training_dataset)
        print("Example of training prompt:\n", training_dataset[-1]["prompt"])

        # No validation dataset for induction
        validation_dataset = []

        # Create a prompt for each test case
        test_dataset = [
            {
                "prompt": self.build_prompt(
                    context_examples=data_problems["train"], test_input=np.array(test_input["input"])
                ),
                "problem": test_input["input"],
                "solution": test_output,
            }
            for test_input, test_output in zip(data_problems["test"], data_solutions)
        ]

        return training_dataset, validation_dataset, test_dataset

    def get_neighborhood_samples_prompt(self, target_input, target_output, alt=False):
        context_str = ""
        for i, (prob, sol) in enumerate(zip(target_input["problems"], target_input["solutions"])):
            input_str = self.task.encode_representation(np.array(prob))
            output_str = self.task.encode_representation(np.array(sol))
            context_str += f"Example {i+1}\nInput:\n{input_str}\n\nOutput:\n{output_str}\n\n\n"

        program = parse_code(target_output[0])[0]
        program = f"```python\n{program}\n```"
        return [
            {"content": self.system_prompt_for_neighbors % context_str, "role": "system"},
            {"content": self.user_prompt_for_neighbors % program, "role": "user"},
        ]

    def get_opro_samples_prompt(self, target_input, target_output, alt=False):
        context_str = ""
        for i, (prob, sol) in enumerate(zip(target_input["problems"], target_input["solutions"])):
            input_str = self.task.encode_representation(np.array(prob))
            output_str = self.task.encode_representation(np.array(sol))
            context_str += f"Example {i+1}\nInput:\n{input_str}\n\nOutput:\n{output_str}\n\n\n"

        program = parse_code(target_output[0])[0]
        program = f"```python\n{program}\n```"
        return [
            {"content": self.system_prompt_for_neighbors % context_str, "role": "system"},
            {"content": self.user_prompt_for_opro % program, "role": "user"},
        ]
