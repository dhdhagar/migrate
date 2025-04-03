import json
import itertools
import numpy as np
import random
from trl import maybe_apply_chat_template


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

    system_prompt_for_neighbors = """You are a helpful chatbot with high attention to detail who is not talkative and \
responds only with the answer and no additional conversation. There is a specific grid transformation that we want to \
use on all input grids and we are trying to guess the output grid for a specific provided input grid.

Here is the input grid and my guess for the output grid:
%s -> %s

Provide a variation of my guess that could be the correct answer."""


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


def get_arc_neighborhood_samples_prompt(target_input, target_output):
    return [
        {"content": ARC.system_prompt_for_neighbors % (target_input, target_output), "role": "system"},
        {"content": f"{target_input} -> ", "role": "user"},
    ]


def get_arc_datasets(
        task_id,
        arc_dataset_file,
        arc_dataset_solution_file,
        minimum_training_size=64,
        maximum_training_size=64,
        maximum_eval_size=64,
        do_permutation=False,
        tokenizer=None,
        max_seq_len=None
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
    with open(arc_dataset_solution_file, "r", encoding="utf-8") as handle:
        data_solutions = json.load(handle)
    training_examples = data[task_id]["train"][1:]
    validation_example = data[task_id]["train"][0]

    print("\nAvailable input-output examples:", len(data[task_id]["train"]))

    # Creating training prompts
    training_dataset = []
    for i, leave_out in enumerate(training_examples):
        leave_out_input = str(np.array(leave_out["input"]))
        possible_context_examples = training_examples[:i] + training_examples[i + 1:]
        dataset = create_arc_prompts(possible_context_examples, leave_out_input, do_permutation)
        dataset = [{"prompt": x, "problem": np.array(leave_out["input"]), "solution": np.array(leave_out["output"])} for
                   x in dataset]
        # Only keep prompts that are shorter than the maximum sequence length
        if max_seq_len is not None:
            dataset = [x for x in dataset if len(
                tokenizer(maybe_apply_chat_template({"prompt": x["prompt"]}, tokenizer)['prompt'])[
                    "input_ids"]) <= max_seq_len]
        training_dataset += dataset
    # Reverse the order so that the longest prompts are first
    training_dataset = training_dataset[::-1]

    # Multiply the dataset until it's longer than the minimum length
    print("Training dataset size:", len(training_dataset))
    if len(training_dataset) < minimum_training_size:
        training_dataset = training_dataset * ((minimum_training_size // len(training_dataset)) + 1)
        print("Extending training dataset to:", len(training_dataset))
    if len(training_dataset) > maximum_training_size:
        training_dataset = training_dataset[:maximum_training_size]
        print("Clipping training dataset size to:", len(training_dataset))
    random.shuffle(training_dataset)

    # Create validation prompts
    validation_input = str(np.array(validation_example["input"]))
    validation_dataset = create_arc_prompts(training_examples, validation_input, do_permutation)
    # Only keep prompts that are shorter than the maximum sequence length
    if max_seq_len is not None:
        validation_dataset = [x for x in validation_dataset if len(
            tokenizer(maybe_apply_chat_template({"prompt": x}, tokenizer)['prompt'])["input_ids"]) <= max_seq_len]
    print("Validation dataset size:", len(validation_dataset))
    if len(validation_dataset) > maximum_eval_size:
        validation_dataset = validation_dataset[-maximum_eval_size:]
        print("Clipping validation dataset size to:", len(validation_dataset))
    validation_dataset = {"dataset": validation_dataset, "problem": np.array(validation_example["input"]),
                          "solution": np.array(validation_example["output"])}

    # Creating test prompts
    all_training_examples = data[task_id]["train"]
    test_input = np.array(data[task_id]["test"][0]["input"])
    test_dataset = create_arc_prompts(all_training_examples, str(test_input), do_permutation)
    # Only keep prompts that are shorter than the maximum sequence length
    if max_seq_len is not None:
        test_dataset = [x for x in test_dataset if len(
            tokenizer(maybe_apply_chat_template({"prompt": x}, tokenizer)['prompt'])["input_ids"]) <= max_seq_len]
    print("Test dataset size:", len(test_dataset))
    if len(test_dataset) > maximum_eval_size:
        test_dataset = test_dataset[-maximum_eval_size:]
        print("Clipping test dataset size to:", len(test_dataset))
    test_dataset = {"dataset": test_dataset, "problem": test_input, "solution": np.array(data_solutions[task_id][0])}

    print()
    return training_dataset, validation_dataset, test_dataset


def get_prompt(task, batch_size, target):
    if task == "semantle":
        return get_semantle_prompt(batch_size)
    elif task == "chem":
        return get_mol_prompt(batch_size, "KIT")
