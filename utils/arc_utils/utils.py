import re
import numpy as np


def parse_code(paragraph: str) -> list:
    """
    This function extracts all Markdown code blocks from a given paragraph.
    Args:
        paragraph (str): The input paragraph containing the Markdown code blocks.
    Returns:
        list: A list of extracted code blocks.
    """
    # Regular expression to match Markdown code blocks
    code_block_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    # Find all code blocks in the paragraph
    matches = code_block_pattern.findall(paragraph)

    # Strip any leading/trailing whitespace from each code block
    code_blocks = [match.strip() for match in matches]

    if code_blocks:
        return code_blocks

    # assume that it does not begin with python
    code_block_pattern = re.compile(r"```(.*?)```", re.DOTALL)
    matches = code_block_pattern.findall(paragraph)
    code_blocks = [match.strip() for match in matches]
    return code_blocks


def hamming_distance(solution, attempt):
    """
    Parses a string representation of a 2D array into a NumPy ndarray.

    Parameters:
    - solution (np.ndarray): The solution ARC grid represented as a 2D array
    - attempt (np.ndarray): The proposed ARC grid represented as a 2D array

    Returns:
    - float: The normalized hamming distance between the solution and attempt.
    """
    # attempt must be a 2D array
    if attempt is not None and len(attempt.shape) == 2 and attempt.size <= solution.size:
        output_size = solution.shape[0] * solution.shape[1]
        # if shapes are different find the minimum shape
        min_width = min(solution.shape[1], attempt.shape[1])
        min_height = min(solution.shape[0], attempt.shape[0])
        solution = solution[:min_height, :min_width]
        attempt = attempt[:min_height, :min_width]
        # remaining number of elements
        additional_elements = output_size - (min_height * min_width)
        return (int(np.sum(solution != attempt)) + additional_elements) / output_size
    return 1.0
