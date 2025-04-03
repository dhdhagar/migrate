import json
import itertools
import numpy as np
import random
from trl import maybe_apply_chat_template
from typing import List, Dict
from abc import ABC, abstractmethod
from arc_utils.utils import GridConverter


def get_semantle_prompt(batch_size):
    return [
        {
            "content": "You are a helpful chatbot with high attention to detail who is not talkative and responds "
            "only with the answer and no additional conversation. All your responses should be in JSON format, i.e. "
            '{key: value}, where the key is always "response" and the value can be a string, int, list, or dict, '
            "depending on the context.",
            "role": "system",
        },
        {
            "content": "Your task is to guess a hidden word from the English dictionary. Stick to proper, "
            f"single-word English words. Now, guess exactly n={batch_size} new word(s) that could be the hidden word. Be "
            'creative! (Note: give only a list of word(s) in the provided JSON format, e.g. {"response": '
            '["word1", "word2",...]})',
            "role": "user",
        },
    ]


def get_semantle_related_prompt(batch_size, chosen_completion):
    return [
        {
            "content": "You are a helpful chatbot with high attention to detail who is not talkative and responds "
            "only with the answer and no additional conversation. All your responses should be in JSON format, "
            'i.e. {key: value}, where the key is always "response" and the value can be a string, int, list, '
            "or dict, depending on the context.",
            "role": "system",
        },
        {
            "content": "Your task is to guess words related to a word from the English dictionary. Stick to proper, "
            f"single-word English words. Now, guess exactly n={batch_size} new word(s) that could be related to the word(s) "
            f'"{chosen_completion}". Be creative! (Note: give only a list of word(s) in the provided JSON format, e.g. '
            '{"response": ["word1", "word2",...]})',
            "role": "user",
        },
    ]


def get_arc_related_prompt(user_input, user_output):
    #     system_prompt_w_context = """You are a helpful chatbot with high attention to detail who is not talkative and \
    # responds only with the answer and no additional conversation. Consider the underlying transformation in the \
    # following examples and provide a similar grid to the output solution in the test case.
    #
    # Note: the output grid can be of a different shape than the input grid and the input-output grids are python arrays. \
    # Your answer must follow the same format.
    #
    # Here are some examples using this transformation:
    # %s
    #
    # Now considering the transformation, provide a similar ouput grid to the one in the test case."""
    system_prompt = """You are a helpful chatbot with high attention to detail who is not talkative and \
responds only with the answer and no additional conversation. Consider the underlying transformation in the \
following example and provide a similar grid to the output solution.

Note: the output grid can be of a different shape than the input grid and the input-output grids are python arrays. \
Your answer must follow the same format.

Now provide a very similar ouput grid to the one in the test case."""

    prompt = [
        {"content": system_prompt, "role": "system"},
        {"content": f"{user_input} {user_output}", "role": "user"},
    ]
    return prompt


def get_mol_prompt(batch_size, protein):
    return [
        {
            "content": "You are a helpful chatbot with high attention to detail who is not talkative and responds "
            "only with the answer and no additional conversation. All your responses should be in JSON format, i.e. "
            '{key: value}, where the key is always "response" and the value can be a string, int, list, or dict, '
            "depending on the context.",
            "role": "system",
        },
        {
            "content": f"Your task is to propose a molecule to bind to the {protein} protein with a high druglikeness and "
            f"low docking score. Stick to proper molecules in the SMILES notation. Now, prompose exactly n={batch_size}"
            "new molecule(s). Be creative! (Note: give only a list of word(s) in the provided JSON format, e.g. "
            '{"response": ["SMILES1", "SMILES2",...]})',
            "role": "user",
        },
    ]


class ARC_Prompts(ABC):
    def __init__(self, use_barc_format):
        self.use_barc_format = use_barc_format
        self.gridConverter = GridConverter(use_barc_format)

    @abstractmethod
    def build_prompt(self, context_examples: List[np.ndarray], test_input: str) -> List[Dict[str, str]]:
        pass


class ARC_Induction(ARC_Prompts):
    def __init__(self, use_barc_format):
        super().__init__(use_barc_format)

        # Prompts for BARC
        if self.use_barc_format:
            self.system_prompt_w_context = """You are a world-class puzzle solver with exceptional pattern recognition skills. \
Your task is to analyze puzzles, spot patterns, and provide direct solutions."""
            self.user_prompt_w_context = """Given input-output grid pairs as reference examples, carefully observe the patterns to predict \
the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, \
with cells (colors) separated by spaces and rows by newlines. Here are the input and output grids for the reference examples:\n%s
Here is the input grid for the test example:\nInput:\n%s\n\nWrite a Python function ‘transform‘ that can convert any given input grid to
its corresponding output grid based on the pattern observed in the reference examples"""
            # TODO: Rewrite prompt for no context case if necessary
            self.user_prompt = """Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines. \
Here is the input grid for the test example:\nInput:\n%s\n\nDirectly provide the output grids corresponding to the given test input \
grids, based on the patterns observed in the reference examples."""
            # TODO: Write prompt for neighborhood sampling with BARC
            self.system_prompt_for_neighbors = ""
        # Prompts for TTT (None since a TTT induction model does not exist)
        else:
            self.system_prompt_w_context = ""
            self.system_prompt = ""
            self.system_prompt_for_neighbors = ""

    def build_prompt(self, context_examples: List[np.ndarray], test_input: str) -> List[Dict[str, str]]:
        if self.use_barc_format:
            context_str = ""
            for i, example in enumerate(context_examples):
                input_str = self.gridConverter.encode(np.array(example["input"]))
                output_str = self.gridConverter.encode(np.array(example["output"]))
                context_str += f"Example {i+1}\nInput:\n{input_str}\n\nOutput:\n{output_str}\n\n\n"

            if len(context_examples) > 0:
                system_prompt = self.system_prompt_w_context
                user_prompt = self.user_prompt_w_context % (context_str, self.gridConverter.encode(test_input))
            else:
                system_prompt = self.system_prompt
                user_prompt = self.user_prompt % (self.gridConverter.encode(test_input))
            prompt = [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
            ]
        else:
            raise Exception("TTT Induction is not implemented")
        return prompt


class ARC_Transduction(ARC_Prompts):
    def __init__(self, use_barc_format):
        super().__init__(use_barc_format)

        # Prompts for BARC
        if use_barc_format:
            self.system_prompt_w_context = "You are a world-class puzzle solver with exceptional pattern recognition skills. \
Your task is to analyze puzzles, spot patterns, and provide direct solutions."
            self.system_prompt = """You are a helpful chatbot with high attention to detail who is not talkative and responds only \
with the answer and no additional conversation. There is a specific grid transformation that we want to use on all \
input grids. Guess the transformation we want and apply it to the provided test case. 

Note: the output grid can be of a different shape than the input grid and the input-output grids are python arrays. \
Your answer must follow the same format.

Now apply the transformation to the provided test case."""
            # TODO: Write prompt for neighborhood sampling with BARC
            self.system_prompt_for_neighbors = ""
            self.user_prompt = """Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid \
for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells (colors) \
separated by spaces and rows by newlines.\nHere are the input and output grids for the reference examples:\n%sHere is the input \
grid for the test example:\nInput:\n%s\n\n\nDirectly provide the output grids corresponding to the given test input grids, based on \
the patterns observed in the reference examples."""
        # Prompts for TTT
        else:
            self.system_prompt_w_context = """You are a helpful chatbot with high attention to detail who is not talkative and \
responds only with the answer and no additional conversation. Figure out the underlying transformation in the \
following examples and apply it to the test case.

Note: the output grid can be of a different shape than the input grid and the input-output grids are python arrays. \
Your answer must follow the same format.

Here are some examples using this transformation:
%s

Now apply the transformation to the provided test case."""
            self.system_prompt = """You are a helpful chatbot with high attention to detail who is not talkative and responds only \
with the answer and no additional conversation. There is a specific grid transformation that we want to use on all \
input grids. Guess the transformation we want and apply it to the provided test case. 

Note: the output grid can be of a different shape than the input grid and the input-output grids are python arrays. \
Your answer must follow the same format.

Now apply the transformation to the provided test case."""
            self.system_prompt_for_neighbors = """You are a helpful chatbot with high attention to detail who is not talkative and \
responds only with the answer and no additional conversation. There is a specific grid transformation that we want to \
use on all input grids and we are trying to guess the output grid for a specific provided input grid.

Here is the input grid and my guess for the output grid:
%s -> %s

Provide a variation of my guess that could be the correct answer."""

    def build_prompt(self, context_examples: List[Dict[str, str]], test_input: np.ndarray) -> List[Dict[str, str]]:
        if self.use_barc_format:
            context_str = ""
            for i, example in enumerate(context_examples):
                input_str = self.gridConverter.encode(np.array(example["input"]))
                output_str = self.gridConverter.encode(np.array(example["output"]))
                context_str += f"Example {i+1}\nInput:\n{input_str}\n\nOutput:\n{output_str}\n\n\n"

            system_prompt = self.system_prompt_w_context if len(context_examples) > 0 else self.system_prompt
            prompt = [
                {"content": system_prompt, "role": "system"},
                {"content": self.user_prompt % (context_str, self.gridConverter.encode(test_input)), "role": "user"},
            ]
        else:
            context_str = ""
            for example in context_examples:
                input_str = self.gridConverter.encode(np.array(example["input"]))
                output_str = self.gridConverter.encode(np.array(example["output"]))
                context_str += f"{input_str} -> {output_str}#\n"

            system_prompt = self.system_prompt_w_context % context_str if len(context_str) > 0 else self.system_prompt
            prompt = [
                {"content": system_prompt, "role": "system"},
                {"content": f"{self.gridConverter.encode(test_input)} -> ", "role": "user"},
            ]
        return prompt


def create_arc_prompts(
    possible_context_examples: List[Dict[str, str]],
    user_input: np.ndarray,
    do_permutation=False,
    use_barc_format=False,
    use_induction=False,
):
    dataset = []
    prompt_builder = ARC_Induction(use_barc_format) if use_induction else ARC_Transduction(use_barc_format)
    # Loop over all possible context sizes
    for i in range(len(possible_context_examples) + 1):
        if do_permutation:
            context_combinations = list(itertools.permutations(possible_context_examples, i))
        else:
            context_combinations = list(itertools.combinations(possible_context_examples, i))

        # Loop over all possible contexts of size i
        for context in context_combinations:
            dataset.append(prompt_builder.build_prompt(list(context), user_input))
    return dataset


def get_arc_neighborhood_samples_prompt(target_input, target_output, use_barc_format=False, use_induction=False):
    prompts = ARC_Induction(use_barc_format) if use_induction else ARC_Transduction(use_barc_format)
    return [
        {"content": prompts.system_prompt_for_neighbors % (target_input, target_output), "role": "system"},
        {"content": f"{target_input} -> ", "role": "user"},
    ]


def get_arc_datasets(
    task_id,
    arc_dataset_file,
    arc_dataset_solutions_file,
    min_training_size=50,
    max_training_size=80,
    max_validation_size=64,
    max_test_size=64,
    use_permutations=False,
    tokenizer=None,
    max_seq_len=2048,
    use_barc_format=False,
):
    """
    Create ARC training, validation, and testing prompt datasets.

    Parameters:
    - task_d (str): ARC task id.
    - arc_dataset_file (str):  Path to ARC dataset file
    - arc_dataset_solution_file (str): Path to ARC dataset solution file
    - minimum_training_size (int): Minimum size of the training dataset
    - do_permutation (bool): Whether ordering should matter for the context examples


    Returns:
    - training_dataset: A list of dictionaries with the training prompt and the corresponding training solution
    - validation_dataset: A dictionary with the validation dataset and the validation solution
    - test_dataset: A dictionary with the test dataset and the test solution
    """
    with open(arc_dataset_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    with open(arc_dataset_solutions_file, "r", encoding="utf-8") as handle:
        data_solutions = json.load(handle)
    training_examples = data[task_id]["train"][1:]
    validation_example = data[task_id]["train"][0]

    print("Available input-output examples", len(data[task_id]["train"]))

    # Create training prompts
    training_dataset = []
    for i, leave_out in enumerate(training_examples):
        leave_out_input = np.array(leave_out["input"])
        possible_context_examples = training_examples[:i] + training_examples[i + 1 :]
        dataset = create_arc_prompts(possible_context_examples, leave_out_input, use_permutations, use_barc_format)
        dataset = [
            {"prompt": x, "problem": np.array(leave_out["input"]), "solution": np.array(leave_out["output"])}
            for x in dataset
        ]
        # Only keep prompts that are shorter than the maximum sequence length
        if max_seq_len is not None:
            dataset = [
                x
                for x in dataset
                if len(tokenizer(maybe_apply_chat_template({"prompt": x["prompt"]}, tokenizer)["prompt"])["input_ids"])
                <= max_seq_len
            ]
        training_dataset += dataset

    # Multiply the dataset until it's longer than the minimum length
    print("Training dataset size", len(training_dataset))
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

    # Create validation prompts
    validation_input = np.array(validation_example["input"])
    validation_dataset = create_arc_prompts(training_examples, validation_input, use_permutations, use_barc_format)
    if len(validation_dataset) > max_validation_size:
        validation_dataset = validation_dataset[-max_validation_size:]
        print("Clipping validation dataset size to:", len(validation_dataset))
    validation_dataset = {
        "dataset": validation_dataset,
        "problem": np.array(validation_example["input"]),
        "solution": np.array(validation_example["output"]),
    }

    # Create test prompts
    all_training_examples = data[task_id]["train"]
    leave_out_input = str(np.array(data[task_id]["test"][0]["input"]))
    test_input = np.array(data[task_id]["test"][0]["input"])
    test_dataset = create_arc_prompts(all_training_examples, test_input, use_permutations, use_barc_format)
    print("EXAMPLE PROMPT:", test_dataset[-1])
    # Only keep prompts that are shorter than the maximum sequence length
    if max_seq_len is not None:
        test_dataset = [
            x
            for x in test_dataset
            if len(tokenizer(maybe_apply_chat_template({"prompt": x}, tokenizer)["prompt"])["input_ids"]) <= max_seq_len
        ]
    print("Test dataset size:", len(test_dataset))
    if len(test_dataset) > max_test_size:
        test_dataset = test_dataset[-max_test_size:]
        print("Clipping test dataset size to:", len(test_dataset))
    test_dataset = {"dataset": test_dataset, "problem": test_input, "solution": np.array(data_solutions[task_id][0])}

    return training_dataset, validation_dataset, test_dataset


def get_prompt(task, batch_size, target):
    if task == "semantle":
        return get_semantle_prompt(batch_size)
    elif task == "chem":
        return get_mol_prompt(batch_size, "KIT")
