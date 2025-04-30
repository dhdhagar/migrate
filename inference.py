from arc_utils.utils import GridConverter
from trl import maybe_apply_chat_template
import torch
from tqdm import tqdm
import numpy as np
from trl.models.utils import unwrap_model_for_generation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_completions(trainer, tokenizer, _model_for_inference, prompts, params):
    with torch.no_grad():
        prompts_text = [maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"] for prompt in prompts]
        prompt_inputs = tokenizer(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        completions = []
        for i in tqdm(range(0, len(prompt_inputs["input_ids"]), params["inf_batch_size"])):
            prompt_ids, prompt_mask = (
                prompt_inputs["input_ids"][i : i + params["inf_batch_size"]],
                prompt_inputs["attention_mask"][i : i + params["inf_batch_size"]],
            )
            with unwrap_model_for_generation(_model_for_inference, trainer.accelerator) as unwrapped_model:
                completion_ids = unwrapped_model.generate(
                    input_ids=prompt_ids.to(DEVICE),
                    attention_mask=prompt_mask.to(DEVICE),
                    do_sample=params["inf_temperature"] != 0,
                    temperature=params["inf_temperature"],
                    max_new_tokens=params["inf_max_new_tokens"],
                )
                prompt_length = prompt_inputs["input_ids"].size(1)
                completions.extend(tokenizer.batch_decode(completion_ids[:, prompt_length:], skip_special_tokens=True))
        return completions


def get_majority_vote(data, results):
    output_counts = {}
    for x in results:
        output = x["output"]
        if output in output_counts:
            output_counts[output]["count"] += 1
        else:
            output_counts[output] = {"score": x["test_score"], "count": 1}
    majority_outputs = sorted(output_counts.items(), key=lambda x: x[1]["count"], reverse=True)
    majority_outputs = [{"output": x[0], "score": x[1]["score"], "count": x[1]["count"]} for x in majority_outputs]
    data["filtered_outputs"] = majority_outputs
    data["num_filtered_outputs"] = len(results)
    data["pass1"] = majority_outputs[0]
    data["pass2"] = max(majority_outputs[:2], key=lambda x: x["score"])
    return data


def run_transduction_inference(trainer, tokenizer, _model_for_inference, test_dataset, params):
    # Generate completions
    prompts = test_dataset["dataset"]
    completions = generate_completions(trainer, tokenizer, _model_for_inference, prompts, params)

    # Aggregate results
    results = []
    gridConverter = GridConverter(params["use_barc_format"])
    test_solution = test_dataset["solution"]
    for completion in completions:
        # Evaluate test example
        parsed_completion = gridConverter.decode(completion)
        valid_grid = parsed_completion.size != 0
        parsed_completion = completion if valid_grid else parsed_completion
        # Get black-box score if completion is valid otherwise 0
        test_score = trainer.get_bb_score(test_solution, parsed_completion) if valid_grid else 0
        test_output = str(parsed_completion)

        # Track individual girds and their scores
        results.append({"output": test_output, "test_score": test_score, "valid_grid": valid_grid})

    # Compute & record summary metrics
    results = sorted(results, key=lambda x: x["test_score"], reverse=True)
    data = {"test_samples": results, "pass1": None, "pass2": None, "oracle": None}

    oracle = max(results, key=lambda x: x["test_score"])
    oracle_votes = len([x for x in results if x["output"] == oracle["output"]])
    data["oracle"] = {"output": oracle["output"], "score": oracle["test_score"], "count": oracle_votes}

    # Compute & log pass@1 and pass@2 results
    filtered_results = [x for x in results if x["valid_grid"]]
    if len(filtered_results) > 0:
        data = get_majority_vote(data, results)

    return data


def run_induction_inference(
    trainer, tokenizer, _model_for_inference, training_dataset, test_dataset, params, programs_from_training
):
    # Collect programs to evaluate
    completions = []
    if programs_from_training is not None:
        completions.extend(programs_from_training)
    if params['inf_num_samples'] > 0:
        # Generate completions
        prompts = test_dataset["dataset"]
        prompts = prompts * (params["inf_num_samples"] // len(prompts) + 1)
        prompts = prompts[: params["inf_num_samples"]]
        completions.extend(generate_completions(trainer, tokenizer, _model_for_inference, prompts, params))

    # Aggregate results
    results = []
    gridConverter = GridConverter(params["use_barc_format"], use_induction=True)
    test_problem = test_dataset["problem"]
    test_solution = test_dataset["solution"]
    for completion in completions:
        # Evaluate test example
        parsed_completion = gridConverter.decode(completion, input_grid=test_problem)
        # Get black-box score if completion is valid otherwise 0
        test_score = (
            trainer.get_bb_score(test_solution, parsed_completion) if isinstance(parsed_completion, np.ndarray) else 0
        )
        test_output = str(parsed_completion)

        # Evaluate training example
        train_scores = []
        for train_problem, train_solution in zip(training_dataset[0]["problem"], training_dataset[0]["solution"]):
            train_completion = gridConverter.decode(completion, input_grid=np.array(train_problem))
            train_completion = "Error executing code" if train_completion.size == 0 else train_completion
            train_score = (
                trainer.get_bb_score(np.array(train_solution), train_completion)
                if isinstance(train_completion, np.ndarray)
                else 0
            )
            train_scores.append(train_score)

        # Track individual programs and their scores
        results.append(
            {"output": test_output, "test_score": test_score, "train_score": np.mean(train_scores), "code": completion}
        )

    # Compute & record summary metrics
    results = sorted(results, key=lambda x: x["test_score"], reverse=True)
    data = {
        "test_samples": results,
        "filtered_outputs": [],
        "num_filtered_outputs": 0,
        "pass1": None,
        "pass2": None,
        "oracle": None,
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "false_negatives": 0,
    }

    oracle = max(results, key=lambda x: x["test_score"])
    oracle_votes = len([x for x in results if x["output"] == oracle["output"]])
    train_solved = [x["train_score"] == 1 for x in results]
    test_solved = [x["test_score"] == 1 for x in results]

    data["oracle"] = {"output": oracle["output"], "score": oracle["test_score"], "count": oracle_votes}
    data["true_positives"] = np.mean([int(x == 1 and y == 1) for x, y in zip(train_solved, test_solved)])
    data["false_positives"] = np.mean([int(x == 1 and y == 0) for x, y in zip(train_solved, test_solved)])
    data["true_negatives"] = np.mean([int(x == 0 and y == 0) for x, y in zip(train_solved, test_solved)])
    data["false_negatives"] = np.mean([int(x == 0 and y == 1) for x, y in zip(train_solved, test_solved)])

    # Compute & log pass@1 and pass@2 results
    filtered_results = [x for x in results if x["train_score"] == 1 and x["output"] != "Error executing code"]
    if len(filtered_results) > 0:
        data = get_majority_vote(data, results)

    return data
