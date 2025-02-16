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
            f"single-word English words. Now, guess exactly n={batch_size} new word(s) that could be related to the word "
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


def get_prompt(task, batch_size):
    if task == "semantle":
        return get_semantle_prompt(batch_size)
    elif task == "molecule":
        return get_mol_prompt(batch_size, "placeholder")
