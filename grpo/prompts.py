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


def get_arc_prompt(task_id, arc_dataset_file):
    with open(arc_dataset_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    training_examples = data[task_id]["train"][1:]  # Leave first example out for validation

    batches = []
    # Loop over context sizes starting from the largest
    for i in range(len(training_examples) - 1, -1, -1):
        batch = []
        for j, leave_out in enumerate(training_examples):
            leave_out_input = np.array(leave_out["input"])
            leave_out_str = np.array2string(leave_out_input, separator=" ", formatter={"all": lambda x: str(x)})

            context_examples = training_examples[:j] + training_examples[j + 1 :]
            context_combinations = list(itertools.combinations(context_examples, i))  # Get contexts of with i examples

            for context_example in context_combinations:
                # Process grids into string for prompt
                context_str = ""
                for example in context_example:
                    # For correct grid string representation
                    input_str = str(np.array(example["input"])).replace(",", "")
                    output_str = str(np.array(example["output"])).replace(",", "")
                    context_str += f"{input_str} -> {output_str}#\n"

                if context_str != "":
                    system_prompt = f"You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no additional conversation. Figure out the underlying transformation in the following examples and apply it to the test case. Here are some examples from this transformation, your answer must follow the format. \nThe input-output grids are provided as python arrays:\n{context_str}"
                    # TTT prompt
                    # system_prompt = f"Figure out the underlying transformation in the following examples and apply it to the test case. Here are some examples from this transformation, your answer must follow the format. \nThe input-output grids are provided as python arrays:\n{context_str}"
                else:
                    # Prompt for training data that have no context examples
                    system_prompt = "You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no additional conversation. Figure out the underlying transformation and apply it to the test case. The output grid can be a different shape from the input grid.\nThe input-output grids are provided as python arrays:\n"

                prompt = [
                    {"content": system_prompt, "role": "system"},
                    {"content": f"{leave_out_str} -> ", "role": "user"},
                ]
                batch.append({"prompt": prompt, "leave_out_idx": j + 1})
        random.shuffle(batch)
        batches.append(batch)

    # Create a dataset with 4 different prompts with the most examples in their context
    training_dataset = []
    for batch in batches:
        training_dataset += batch
        if len(training_dataset) >= 4:
            break
    training_dataset = training_dataset[:4]
    return training_dataset


def get_prompt(task, batch_size, target):
    if task == "semantle":
        return get_semantle_prompt(batch_size)
    elif task == "chem":
        return get_mol_prompt(batch_size, "KIT")
    elif task == "arc":
        return get_arc_prompt(batch_size, target)
