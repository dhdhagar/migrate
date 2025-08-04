from abc import ABC, abstractmethod
from typing import List, Dict
from utils.base_task import Task


class Prompts(ABC):
    def __init__(self, task: Task, **kwargs):
        self.task = task

    #     self.use_barc_format = use_barc_format
    #     self.gridConverter = GridConverter(use_barc_format)

    @abstractmethod
    def build_prompt(self) -> List[Dict[str, str]]:
        """
        Constructs a single prompt of the following format:
            [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
            ]

        Returns:
        - List[Dict[str, str]]: A single prompt
        """
        pass
