import re
import numpy as np
import glob
import json
import torch
import random


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
        arr = parse_numpy_from_str(output[start_idx[-1]: end_idx[-1] + 2])
        return arr
    return np.array([[]])


def parse_numpy_from_str(array_str: str, verbose=False) -> np.ndarray:
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
        if verbose:
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


def pro_loss(p, start_neg_idx_to_ignore=None):
    """
    Compute the PRO (https://arxiv.org/pdf/2306.17492) loss for a batch of scores p.
    Args:
        p: Tensor of shape (batch_size, n) containing scores.
    Returns:
        Scalar tensor representing the loss.
    """
    batch_size, n = p.shape

    indices = torch.arange(n - 1, device=p.device).view(1, -1, 1)  # Shape: (1, n-1, 1)
    p_expanded = p.unsqueeze(1).expand(-1, n - 1, -1)  # Shape: (batch_size, n-1, n)
    mask = torch.arange(n, device=p.device).view(1, 1, -1) >= indices  # Mask for denominator summation
    inf_mask = torch.full(mask.shape, float('-inf'), device=p.device)
    inf_mask[mask] = 0  # Only allow valid indices
    denominators = torch.logsumexp(p_expanded + inf_mask, dim=2)  # Shape: (batch_size, n-1)
    numerators = p[:, :-1]  # Shape: (batch_size, n-1)
    loss = - (numerators - denominators)[:, :start_neg_idx_to_ignore].sum(dim=1).mean()

    return loss


def add_to_batch(self, replacement, completions, rewards):
    # Helper utility for ARCTrainer to add a replacement completion to the batch
    if replacement is not None:
        idx = random.randint(0, len(completions) - 1)
        if self.inject_best_at_lowest_score:
            idx = np.argmin(rewards)
            self.best_idx_replaced = idx
        # Only replace if the best guess is better than the current guess
        if replacement[1] > rewards[idx]:
            completions[idx] = replacement[0]
            rewards[idx] = replacement[1]
        else:
            self.best_idx_replaced = None


def run_neighborhood_sampling(self, completions, rewards, gold_solution, n_neighbors):
    neigh_samples, neigh_scores = self.get_neighborhood_samples(self.arc_prob, gold_solution, n_neighbors)
    with open(self.logfile, "r") as fh:
        logdata = json.load(fh)
    if "neighborhood_samples" not in logdata:
        logdata["neighborhood_samples"] = {}
    # Add zipped list of samples, scores
    logdata["neighborhood_samples"][f"iteration_{self.iteration}"] = list(zip(neigh_samples, neigh_scores))
    with open(self.logfile, "w") as fh:
        fh.write(json.dumps(logdata, indent=2))

    if self.neighborhood_sampling_strategy == "best":
        # Add neighbors to online samples and keep the best ones
        n_batch = len(completions)
        completions.extend(neigh_samples)
        rewards.extend(neigh_scores)
        # Sort by reward and keep the best ones
        completions, rewards = zip(
            *sorted(zip(completions, rewards), key=lambda x: x[1], reverse=True)[:n_batch])
        completions, rewards = list(completions), list(rewards)

    elif self.neighborhood_sampling_strategy == "mix":
        # Add half of the neighbors to half of the online samples
        n_batch = len(completions)
        _completions = completions[:n_batch // 2]
        _rewards = rewards[:n_batch // 2]
        _completions.extend(neigh_samples[:n_batch // 2])
        _rewards.extend(neigh_scores[:n_batch // 2])
        # If length of _completions is less than n_batch, add online samples
        if len(_completions) < n_batch:
            _completions.extend(completions[n_batch // 2:])
            _rewards.extend(rewards[n_batch // 2:])
        completions, rewards = _completions[:n_batch], _rewards[:n_batch]

    return completions, rewards
