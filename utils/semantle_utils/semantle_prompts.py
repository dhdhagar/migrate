from utils.base_prompts import Prompts
import random
import json
import numpy as np
import itertools
from typing import Dict, List


class SemantlePrompts(Prompts):
    def __init__(self, task, **kwargs):
        super().__init__(task)
        self.system_prompt = """\
You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no \
additional conversation. All your responses should be in JSON format, i.e. '{key: value}', where the key is always \
"response" and the value can be a string, int, list, or dict, depending on the context."""
        self.user_prompt = """\
Your task is to guess a hidden word from the English dictionary. Stick to proper, single-word English words. Now, guess \
exactly n=%s new word(s) that could be the hidden word. Be creative! (Note: give only a list of word(s) in the provided \
JSON format, e.g. {"response": ["word1", "word2",...]})"""

        self.system_prompt_for_neighbors = """\
You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no \
additional conversation. All your responses should be in JSON format, i.e. '{key: value}', where the key is always \
"response" and the value can be a string, int, list, or dict, depending on the context."""
        self.user_prompt_for_neighbors = """\
Your task is to guess words related to a word from the English dictionary. Stick to proper, single-word English words. \
Now, guess exactly n=%s new word(s) that could be related to the word(s):\n%s\nBe creative! (Note: give only a list of \
word(s) in the provided JSON format, e.g. {"response": ["word1", "word2",...]})"""

        self.system_prompt_for_opro = """\
You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no \
additional conversation. All your responses should be in JSON format, i.e. '{key: value}', where the key is always \
"response" and the value can be a string, int, list, or dict, depending on the context."""
        self.user_prompt_for_opro = """\
Your task is to guess a hidden test word from the English dictionary. Stick to proper, single-word English words. Use \
the below series of your previous guesses (in increasing order of their similarity to the hidden word in terms of their \
*meaning*) to make a new guess. Your new guess should not have been made before and should score higher than your \
previous guesses.\n\nHere are your top previous guesses (from worst to best):%s\nNow, guess exactly n=%s new word(s) that \
could give you a target score of 1.0000. Be creative! (Note: give only a list of word(s) in the provided JSON format, \
e.g. {"response": ["word1", "word2",...]})"""

    def build_prompt(self, use_alternate=False, alt_num_guesses=2) -> List[Dict[str, str]]:
        """
        Constructs a single prompt of the following format:
            [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
            ]

        Returns:
        - List[Dict[str, str]]: A single prompt
        """
        if not use_alternate:
            return [
                {"content": self.system_prompt, "role": "system"},
                {"content": self.user_prompt % self.task.migrate_alpha, "role": "user"},
            ]
        else:
            return [
                {"content": self.system_prompt, "role": "system"},
                {"content": self.user_prompt % alt_num_guesses, "role": "user"},
            ]

    def build_dataset(
        self,
        task_id,
        min_training_size=50,
        max_training_size=80,
        max_seq_len=2048,
    ):
        training_dataset = [
            {"prompt": self.build_prompt(), "problem": [self.task.target], "solution": [self.task.target]}
        ]
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
        print("Example of training prompt:\n", training_dataset[-1]["prompt"])

        validation_dataset = []
        test_dataset = []

        return training_dataset, validation_dataset, test_dataset

    def get_neighborhood_samples_prompt(self, target_input, target_output, alt=False):
        context = "\n\n".join(target_output)
        context = f"\n{context}\n"
        return [
            {"content": self.system_prompt_for_neighbors, "role": "system"},
            # {"content": self.user_prompt_for_neighbors % (self.task.migrate_gamma, target_output), "role": "user"},
            {
                "content": self.user_prompt_for_neighbors % (self.task.migrate_gamma if not alt else 2, context),
                "role": "user",
            },
        ]

    def get_opro_samples_prompt(self, target_input, target_output, alt=False):
        if len(target_output) == 1:
            context = f' "{target_output[0]}."\n'
        else:
            context = "\n\n".join(target_output)
            context = f"\n\n{context}\n"
        return [
            {"content": self.system_prompt_for_opro, "role": "system"},
            {
                "content": self.user_prompt_for_opro % (context, self.task.migrate_gamma if not alt else 2),
                "role": "user",
            },
        ]
