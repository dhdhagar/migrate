import re
import numpy as np
import glob
import json
from arc_utils.execute import execute_transformation, wrapped_execute_transformation
from pebble import ProcessPool, ProcessExpired

def color_to_number(color):
    mapping = {
        "BLACK": 0,
        "BLUE": 1,
        "RED": 2,
        "GREEN": 3,
        "YELLOW": 4,
        "GREY": 5,
        "GRAY": 5,  # Alternative spelling
        "PINK": 6,
        "ORANGE": 7,
        "PURPLE": 8,
        "BROWN": 9,
    }
    n = mapping.get(color.upper(), -1)
    assert n != -1, f"Color {color} not found in mapping"
    return n


def parse_code(paragraph):
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


class GridConverter:
    color_map = {
        0: "Black",
        1: "Blue",
        2: "Red",
        3: "Green",
        4: "Yellow",
        5: "Gray",
        6: "Pink",
        7: "Orange",
        8: "Purple",
        9: "Brown",
    }

    def __init__(self, use_barc_format, use_induction=False):
        self.inv_map = {v: k for k, v in self.color_map.items()}
        self.use_barc_format = use_barc_format
        self.use_induction = use_induction

    def encode(self, grid: np.ndarray, include_prefix: bool = False) -> str:
        # Encode numpy grid representation into expected generated output
        if self.use_barc_format:
            output = ""
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    output += self.color_map[grid[i][j]] + " "
                output = output[:-1] + "\n"
            if include_prefix:
                return f"The output grid for the test input grid is:\n\n```\n{output[:-1]}\n```"
            return output[:-1]
        else:
            return str(grid)

    def decode(self, encoded_str: str, input_grid=None) -> np.ndarray:
        # Decode generated outputs into numpy grid representation
        if self.use_barc_format:
            if self.use_induction:
                parsed_codes = parse_code(encoded_str)
                if parsed_codes:
                    code = parsed_codes[0]
                    grid = execute_transformation(code, input_grid, function_name="transform")
                    # if isinstance(grid, str):
                    #     return grid
                    return grid if isinstance(grid, np.ndarray) else np.array([[]])
                else:
                    return np.array([[]])
            else:
                try:
                    if "```" in encoded_str:
                        parsed_encoded_str = encoded_str.split("```")[1].strip()
                        grid = parsed_encoded_str.split("\n")
                        grid = [row.split() for row in grid if row.strip()]
                        grid = [[color_to_number(cell) for cell in row] for row in grid]
                        return np.array(grid)
                    else:
                        return np.array([[]])
                except:
                    return np.array([[]])
        else:
            return parse_response(encoded_str)
    
    def batch_decode(self, encoded_str, input_grid=None):
        # Decode generated outputs into numpy grid representation
        if self.use_barc_format:
            if self.use_induction:
                parsed_codes = parse_code(encoded_str)
                if parsed_codes:
                    code = parsed_codes[0]
                    
                    job_args = [(code, grid, 1, "transform", True) for i, grid in enumerate(input_grid)]
                    
                    ordered_results = [np.array([[]])] * len(job_args)
                    with ProcessPool(max_workers=4) as pool:
                        future = pool.map(wrapped_execute_transformation, job_args)
                        iterator = future.result()
                        while True:
                            try:
                                result = next(iterator)
                                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], int):
                                    job_id, value = result
                                    ordered_results[job_id] = value
                            except StopIteration:
                                break
                            except TimeoutError as error:
                                print("function took longer than %d seconds" % timeout)
                            except ProcessExpired as error:
                                print("%s. Exit code: %d" % (error, error.exitcode))
                            except Exception as error:
                                print("function raised %s" % error)
                    
                    return ordered_results
                else:
                    return [np.array([[]])] * len(encoded_str)
            else:
                try:
                    if "```" in encoded_str:
                        parsed_encoded_str = encoded_str.split("```")[1].strip()
                        grid = parsed_encoded_str.split("\n")
                        grid = [row.split() for row in grid if row.strip()]
                        grid = [[color_to_number(cell) for cell in row] for row in grid]
                        return np.array(grid)
                    else:
                        return np.array([[]])
                except:
                    return np.array([[]])
        else:
            return parse_response(encoded_str)


def parse_response(output: str) -> np.ndarray:
    """
    Parses a LLM string response for an ARC grid.

    Parameters:
    - output (str): A string response from a LLM.

    Returns:
    - np.ndarray: A NumPy array of type int8 representing the parsed 2D array.
    """

    # Find instances of the start and end of an ARC grid

    start_idx = [match.start() for match in re.finditer(r"\[\[", output)]
    end_idx = [match.start() for match in re.finditer(r"\]\]", output)]
    if len(start_idx) > 0 and len(end_idx) > 0:
        # Parse the last grid found in the response into a 2D array
        arr = parse_numpy_from_str(output[start_idx[-1] : end_idx[-1] + 2])
        return arr
    return np.array([[]])


def parse_numpy_from_str(array_str: str) -> np.ndarray:
    """
    Parses a string representation of a 2D array into a NumPy ndarray.

    Parameters:
    - array_str (str): A string representation of a 2D array, where rows are separated by newlines.

    Returns:
    - np.ndarray: A NumPy array of type int8 representing the parsed 2D array.
    """
    try:
        # Remove the surrounding brackets from the string
        clean_str = array_str.replace("#", "").replace(",", "").replace("[", "").replace("]", "")

        # Split the cleaned string by whitespace to get individual elements and convert them to integers
        elements = list(map(int, clean_str.split()))

        # Determine the number of rows by counting the newline characters and adding one
        rows = array_str.count("\n") + 1

        # Calculate the number of columns by dividing the total number of elements by the number of rows
        cols = len(elements) // rows

        # Create the NumPy array with the determined shape and convert it to type int8
        array = np.array(elements).reshape((rows, cols)).astype(np.int8)

        return array
    except Exception as e:
        # Print the exception message and the original string for debugging purposes
        print(e)
        print(array_str)
        # Return a default 1x1 array with a zero element in case of an error
        # raise e
        return np.array([[]])


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
    if attempt is not None and len(attempt.shape) == 2:
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


def get_training_scores(tasks, log_dir, iterations):
    results = {task: [] for task in tasks}
    for task in tasks:
        files = glob.glob(f"{log_dir}/{task}/*.log")
        for file_name in files[-1:]:
            with open(file_name, "r") as file:
                data = json.load(file)
            batch_size = len(data["Guesses"]) // iterations
            for i in range(iterations):
                batch_scores = []
                for j in range(batch_size):
                    try:
                        batch = data["Guesses"][i * batch_size + j]
                        batch = batch[list(batch.keys())[0]]
                        scores = [1 - x[1] for x in batch[:-1]]
                        batch_scores.append([np.mean(scores), np.min(scores)])
                    except Exception as e:
                        print(e, task, i * batch_size + j)
                        if len(batch_scores) > 0:
                            batch_scores.append(batch_scores[-1])
                        else:
                            batch_scores.append([1, 1])
                batch_scores = np.array(batch_scores)
                try:
                    results[task].append([np.mean(batch_scores[:, 0]), np.mean(batch_scores[:, 1])])
                except Exception as _:
                    results[task].append([1, 1])
    return results


def get_training_time(tasks, log_dir):
    durations = []
    for task in tasks:
        files = glob.glob(f"{log_dir}/{task}/*.log")
        for file_name in files[-1:]:
            with open(file_name, "r") as file:
                data = json.load(file)
            try:
                durations.append(data["Duration"])
            except Exception as _:
                pass
    return durations
