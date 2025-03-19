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


class ARC:
    system_prompt_w_context = f"""You are a helpful chatbot with high attention to detail who is not talkative and \
    responds only with the answer and no additional conversation. Figure out the underlying transformation in the \
    following examples and apply it to the test case. 

    Note: the output grid can be of a different shape than the input grid and the input-output grids are python arrays. \
    Your answer must follow the same format.

    Here are some examples using this transformation:
    %s

    Now apply the transformation to the provided test case."""

    system_prompt = f"""You are a helpful chatbot with high attention to detail who is not talkative and responds only \
    with the answer and no additional conversation. There is a specific grid transformation that we want to use on all \
    input grids. Guess the transformation we want and apply it to the provided test case. 

    Note: the output grid can be of a different shape than the input grid and the input-output grids are python arrays. \
    Your answer must follow the same format.

    Now apply the transformation to the provided test case."""


def get_arc_prompt(task_id, arc_dataset_file, readable_prompt=True, all_combinations=False, batches_n=4):
    with open(arc_dataset_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    training_examples = data[task_id]["train"][1:]  # Leave first example out for validation
    validation_example = data[task_id]["train"][0]

    batches = []
    # Loop over context sizes starting from the largest
    for i in range(len(training_examples) - 1, -1, -1):
        batch = []
        for j, leave_out in enumerate(training_examples):
            leave_out_input = np.array(leave_out["input"])
            leave_out_str = np.array2string(leave_out_input, separator=" ",
                                            formatter={"all": lambda x: str(x)}) if readable_prompt else str(
                leave_out_input.tolist())
            leave_out_output = np.array(leave_out["output"])

            context_examples = training_examples[:j] + training_examples[j + 1:]
            if not all_combinations:
                # Get contexts with i examples
                context_combinations = list(itertools.combinations(context_examples, i))
            else:
                context_combinations = [comb for _i in range(1, i + 1) for comb in
                                        list(itertools.combinations(context_examples, _i))]

            for context_example in context_combinations:
                # Process grids into string for prompt
                context_str = ""
                for example in context_example:
                    # For correct grid string representation
                    input_str = str(np.array(example["input"])).replace(",", "") if readable_prompt \
                        else str(example["input"])
                    output_str = str(np.array(example["output"])).replace(",", "") if readable_prompt \
                        else str(example["output"])
                    context_str += f"{input_str} -> {output_str}#\n"

                system_prompt = ARC.system_prompt_w_context % context_str if len(context_str) > 0 else ARC.system_prompt

                prompt = [
                    {"content": system_prompt, "role": "system"},
                    {"content": f"{leave_out_str} -> ", "role": "user"},
                ]
                batch.append({"prompt": prompt, "leave_out_idx": j, "solution": leave_out_output})  # earlier: j + 1
        random.shuffle(batch)
        batches.append(batch)

    # Create a dataset with 4 different prompts with the most examples in their context
    training_dataset = []
    for batch in batches:
        training_dataset += batch
        if batches_n is not None and len(training_dataset) >= batches_n:
            break
    training_dataset = training_dataset[:batches_n]

    # Create validation example
    context_str = ""
    for example in training_examples:
        input_str = str(np.array(example["input"])).replace(",", "") if readable_prompt \
            else str(example["input"])
        output_str = str(np.array(example["output"])).replace(",", "") if readable_prompt \
            else str(example["output"])
        context_str += f"{input_str} -> {output_str}#\n"

    valid_input_str = str(np.array(validation_example["input"])).replace(",", "") if readable_prompt \
        else str(validation_example["input"])
    validation_prompt = [
        {
            "content": ARC.system_prompt_w_context % context_str if len(context_str) > 0 else ARC.system_prompt,
            "role": "system",
        },
        {"content": f"{valid_input_str} -> ", "role": "user"},
    ]
    validation_example = {"prompt": validation_prompt, "solution": np.array(validation_example["output"])}

    # Create test example
    test_example = data[task_id]["test"][0]
    context_str = ""
    for example in data[task_id]["train"]:
        input_str = str(np.array(example["input"])).replace(",", "") if readable_prompt \
            else str(example["input"])
        output_str = str(np.array(example["output"])).replace(",", "") if readable_prompt \
            else str(example["output"])
        context_str += f"{input_str} -> {output_str}#\n"

    test_input_str = str(np.array(test_example["input"])).replace(",", "") if readable_prompt \
        else str(test_example["input"])
    test_prompt = [
        {
            "content": ARC.system_prompt_w_context % context_str if len(context_str) > 0 else ARC.system_prompt,
            "role": "system",
        },
        {"content": f"{test_input_str} -> ", "role": "user"},
    ]
    test_example = {"prompt": test_prompt, "solution": np.array(test_example["output"])}

    return training_dataset, validation_example, test_example


def get_prompt(task, batch_size, target):
    if task == "semantle":
        return get_semantle_prompt(batch_size)
    elif task == "chem":
        return get_mol_prompt(batch_size, "KIT")
    elif task == "arc":
        return get_arc_prompt(batch_size, target)
