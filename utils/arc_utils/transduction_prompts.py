from utils.base_prompts import Prompts


class ARC_Transduction(Prompts):
    def __init__(self, use_barc_format, **kwargs):
        self.use_barc_format = use_barc_format
        if use_barc_format:
            # Prompts for BARC
            self._initialize_barc_prompts()
        else:
            # Prompts for TTT
            self._initialize_ttt_prompts()

    def _initialize_barc_prompts(self):
        self.system_prompt_w_context = "\
You are a world-class puzzle solver with exceptional pattern recognition skills. Your task is to analyze puzzles, spot \
patterns, and provide direct solutions."
        self.system_prompt = """\
You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no \
additional conversation. There is a specific grid transformation that we want to use on all input grids. Guess the \
transformation we want and apply it to the provided test case.

Note: the output grid can be of a different shape than the input grid and the input-output grids are python arrays. \
Your answer must follow the same format.

Now apply the transformation to the provided test case."""
        self.user_prompt = """\
Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new \
test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells \
(colors) separated by spaces and rows by newlines.\nHere are the input and output grids for the reference examples:\n%s\
Here is the input grid for the test example:\nInput:\n%s\n\n\nDirectly provide the output grids corresponding to the \
given test input grids, based on the patterns observed in the reference examples."""
        # TODO: Write prompt for neighborhood sampling with BARC
        self.system_prompt_for_neighbors = ""

    def _initialize_ttt_prompts(self):
        self.system_prompt_w_context = """\
You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no \
additional conversation. Figure out the underlying transformation in the following examples and apply it to the test case.

Note: the output grid can be of a different shape than the input grid and the input-output grids are python arrays. \
Your answer must follow the same format.

Here are some examples using this transformation:
%s

Now apply the transformation to the provided test case."""
        self.system_prompt = """\
You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no \
additional conversation. There is a specific grid transformation that we want to use on all input grids. Guess the \
transformation we want and apply it to the provided test case.

Note: the output grid can be of a different shape than the input grid and the input-output grids are python arrays. \
Your answer must follow the same format.

Now apply the transformation to the provided test case."""
        self.system_prompt_for_neighbors = """\
You are a helpful chatbot with high attention to detail who is not talkative and responds only with the answer and no \
additional conversation. There is a specific grid transformation that we want to use on all input grids and we are \
trying to guess the output grid for a specific provided input grid.

Here is the input grid and my guess for the output grid:
%s -> %s

Provide a variation of my guess that could be the correct answer."""
