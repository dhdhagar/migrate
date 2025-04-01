import json
import itertools
import numpy as np
import random
import arc_utils.utils as arc_utils


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


class ARC:
    system_prompt_w_context = """You are a helpful chatbot with high attention to detail who is not talkative and \
responds only with the answer and no additional conversation. Figure out the underlying transformation in the \
following examples and apply it to the test case.

Note: the output grid can be of a different shape than the input grid and the input-output grids are python arrays. \
Your answer must follow the same format.

Here are some examples using this transformation:
%s

Now apply the transformation to the provided test case."""

    system_prompt = """You are a helpful chatbot with high attention to detail who is not talkative and responds only \
with the answer and no additional conversation. There is a specific grid transformation that we want to use on all \
input grids. Guess the transformation we want and apply it to the provided test case. 

Note: the output grid can be of a different shape than the input grid and the input-output grids are python arrays. \
Your answer must follow the same format.

Now apply the transformation to the provided test case."""


def create_arc_prompts(possible_context_examples, user_input, do_permutation=False):
    dataset = []
    # Loop over all possible context sizes
    for i in range(len(possible_context_examples) + 1):
        if do_permutation:
            context_combinations = list(itertools.permutations(possible_context_examples, i))
        else:
            context_combinations = list(itertools.combinations(possible_context_examples, i))

        # Loop over all possible contexts from leaving out i and of size j
        for context in context_combinations:
            # Build context examples
            context_str = ""
            for example in context:
                input_str = str(np.array(example["input"]))
                output_str = str(np.array(example["output"]))
                context_str += f"{input_str} -> {output_str}#\n"

            system_prompt = ARC.system_prompt_w_context % context_str if len(context_str) > 0 else ARC.system_prompt

            prompt = [
                {"content": system_prompt, "role": "system"},
                {"content": f"{user_input} -> ", "role": "user"},
            ]
            dataset.append(prompt)
    return dataset


def get_arc_datasets(
    task_id,
    arc_dataset_file,
    arc_dataset_solutions_file,
    min_training_size=50,
    max_training_size=80,
    max_validation_size=64,
    max_test_size=64,
    use_permutations=False,
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
    print("PERMUTE", use_permutations)

    training_dataset = []
    for i, leave_out in enumerate(training_examples):
        leave_out_input = str(np.array(leave_out["input"]))
        possible_context_examples = training_examples[:i] + training_examples[i + 1 :]
        dataset = create_arc_prompts(possible_context_examples, leave_out_input, use_permutations)
        dataset = [{"prompt": x, "solution": np.array(leave_out["output"])} for x in dataset]
        training_dataset += dataset
    # Multiply the dataset until it's longer than the minimum length
    print("Training dataset size", len(training_dataset))
    training_dataset = training_dataset * ((min_training_size // len(training_dataset)) + 1)
    print("Training dataset size", len(training_dataset))
    random.shuffle(training_dataset)
    training_dataset = training_dataset[:max_training_size]

    all_training_examples = data[task_id]["train"]
    leave_out_input = str(np.array(data[task_id]["test"][0]["input"]))
    test_dataset = create_arc_prompts(all_training_examples, leave_out_input, use_permutations)[-max_test_size:]
    print("Test dataset size", len(test_dataset))
    test_dataset = {"dataset": test_dataset, "solution": np.array(data_solutions[task_id][0])}

    dataset = create_arc_prompts(training_examples, str(np.array(validation_example["input"])), use_permutations)[
        -max_validation_size:
    ]
    print("Validation dataset size", len(dataset))
    validation_dataset = {"dataset": dataset, "solution": np.array(validation_example["output"])}

    return training_dataset, validation_dataset, test_dataset


def get_prompt(task, batch_size, target):
    if task == "semantle":
        return get_semantle_prompt(batch_size)
    elif task == "chem":
        return get_mol_prompt(batch_size, "KIT")
