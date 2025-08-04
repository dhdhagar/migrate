from trl import maybe_apply_chat_template
import torch
from tqdm import tqdm
import numpy as np
from trl.models.utils import unwrap_model_for_generation
import heapq
import random
from trl.extras.profiling import profiling_context

from trl.import_utils import is_vllm_available

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_completions(trainer, tokenizer, _model_for_inference, prompts, task, training_dataset, params):
    past_guesses = []
    seen_guesses = set()
    completions = []
    with torch.no_grad():
        max_prompt_length = 2048
        for i in tqdm(range(0, len(prompts), params["inf_batch_size"])):
            prompts_text = [
                maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"]
                for prompt in prompts[i : i + params["inf_batch_size"]]
            ]

            if params["migrate_sampling_strategy"] != "none" and len(past_guesses) > 0:
                best_guesses = heapq.nlargest(params["greedy_topk"], past_guesses)
                best_guess = [random.choice(best_guesses)]
                solution = [x[1] for x in best_guess[::-1]]

                train_problems = [np.array(x) for x in training_dataset[0]["problem"]]
                train_solutions = [np.array(x) for x in training_dataset[0]["solution"]]
                problem = {"problems": train_problems, "solutions": train_solutions}

                if params["migrate_sampling_strategy"] == "opro":
                    prompt_obj = task.prompts.get_opro_samples_prompt(problem, solution)
                else:
                    prompt_obj = task.prompts.get_neighborhood_samples_prompt(problem, solution)

                ns_prompts = [maybe_apply_chat_template({"prompt": prompt_obj}, tokenizer)["prompt"]]
                migrate_gamma = params["migrate_gamma"]
                prompts_text[:migrate_gamma] = migrate_gamma * ns_prompts
                print(ns_prompts[0])

            prompt_inputs = tokenizer(
                prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
            prompt_ids = prompt_ids[:, -max_prompt_length:]
            prompt_mask = prompt_mask[:, -max_prompt_length:]

            batch = []
            if params["use_vllm"]:
                if trainer.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=trainer.guided_decoding_regex)
                else:
                    guided_decoding = None
                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": trainer.repetition_penalty,
                    "temperature": trainer.temperature,
                    "top_p": trainer.top_p,
                    "top_k": -1 if trainer.top_k is None else trainer.top_k,
                    "min_p": 0.0 if trainer.min_p is None else trainer.min_p,
                    "max_tokens": trainer.max_completion_length,
                    "guided_decoding": guided_decoding,
                }
                generation_kwargs.update(trainer.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if trainer.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(trainer.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=trainer.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(trainer, "vLLM.generate"):
                    all_outputs = trainer.llm.generate(
                        all_prompts_text, sampling_params=sampling_params, use_tqdm=False
                    )

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if trainer.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=trainer.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

                batch = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
                del completion_ids
                torch.cuda.empty_cache()
            else:

                with unwrap_model_for_generation(_model_for_inference, trainer.accelerator) as unwrapped_model:
                    completion_ids = unwrapped_model.generate(
                        input_ids=prompt_ids.to(DEVICE),
                        attention_mask=prompt_mask.to(DEVICE),
                        do_sample=params["inf_temperature"] != 0,
                        temperature=params["inf_temperature"],
                        max_new_tokens=params["inf_max_new_tokens"],
                    )
                    prompt_length = prompt_inputs["input_ids"].size(1)
                    batch = tokenizer.batch_decode(completion_ids[:, prompt_length:], skip_special_tokens=True)
            completions.extend(batch)

            # Evaluate training example
            for completion in batch:
                train_scores = []
                for train_problem, train_solution in zip(
                    training_dataset[0]["problem"], training_dataset[0]["solution"]
                ):
                    train_completion = task.decode_response(completion, input_grid=np.array(train_problem))
                    if params["use_induction"]:
                        train_completion = "Error executing code" if train_completion.size == 0 else train_completion
                    else:
                        train_completion = completion if train_completion.size == 0 else train_completion
                    train_score = (
                        task.get_bb_score(np.array(train_solution), train_completion)
                        if isinstance(train_completion, np.ndarray)
                        else 0
                    )
                    train_scores.append(train_score)
                train_score = np.mean(train_scores)
                print("TRAINING SCORE", train_score)
                # Early stopping
                if train_score == 1:
                    print("EARLY STOPPING")
                    return completions

                if completion not in seen_guesses and train_score > 0:
                    heapq.heappush(past_guesses, (train_score, completion))
                    seen_guesses.add(completion)

        return completions


def get_majority_vote(data, results):
    # subtasks = len(results[0]["output"])
    subtasks = len(results)
    output_counts = [{} for _ in range(subtasks)]
    for i in range(subtasks):
        for x in results:
            for output, score in zip(x["output"], x["test_score"]):
                if output in output_counts[i]:
                    output_counts[i][output]["count"] += 1
                else:
                    output_counts[i][output] = {"score": score, "count": 1}
    majority_outputs = [
        sorted(output_count.items(), key=lambda x: x[1]["count"], reverse=True) for output_count in output_counts
    ]
    majority_outputs = [
        [{"output": x[0], "score": x[1]["score"], "count": x[1]["count"]} for x in majority_output]
        for majority_output in majority_outputs
    ]
    data["filtered_outputs"] = majority_outputs
    data["num_filtered_outputs"] = len(results)
    data["pass1"] = [majority_output[0] for majority_output in majority_outputs]
    data["pass2"] = [max(majority_output[:2], key=lambda x: x["score"]) for majority_output in majority_outputs]
    return data


def run_transduction_inference(trainer, tokenizer, _model_for_inference, train_dataset, test_dataset, params, task):
    # Generate completions
    prompts = test_dataset["dataset"]
    completions = generate_completions(trainer, tokenizer, _model_for_inference, prompts, task, train_dataset, params)

    # Aggregate results
    results = []
    test_solution = test_dataset["solution"]
    for completion in completions:
        # Evaluate test example
        parsed_completion = task.decode_response(completion)
        valid_grid = parsed_completion.size != 0
        parsed_completion = completion if valid_grid else parsed_completion
        # Get black-box score if completion is valid otherwise 0
        test_score = task.get_bb_score(test_solution, parsed_completion) if valid_grid else 0
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
        data = get_majority_vote(data, filtered_results)

    return data


def run_induction_inference(
    trainer, tokenizer, _model_for_inference, training_dataset, test_dataset, params, programs_from_training, task
):
    # Collect programs to evaluate
    completions = []
    if programs_from_training is not None:
        completions.extend(programs_from_training)
    if params["inf_num_samples"] > 0:
        # Generate completions
        prompts = [x["prompt"] for x in test_dataset]
        prompts = prompts * (params["inf_num_samples"] // len(prompts) + 1)
        prompts = prompts[: params["inf_num_samples"]]
        completions.extend(
            generate_completions(trainer, tokenizer, _model_for_inference, prompts, task, training_dataset, params)
        )
    print("DONE", len(completions))

    # Aggregate results
    results = []
    test_problems = [x["problem"] for x in test_dataset]
    test_solutions = [x["solution"] for x in test_dataset]
    for completion in completions:
        test_scores = []
        test_outputs = []
        for test_problem, test_solution in zip(test_problems, test_solutions):
            # Evaluate test example
            parsed_completion = task.decode_response(completion, input_grid=test_problem)
            if params["use_induction"]:
                parsed_completion = "Error executing code" if parsed_completion.size == 0 else parsed_completion
            else:
                parsed_completion = completion if parsed_completion.size == 0 else parsed_completion
            # Get black-box score if completion is valid otherwise 0
            test_scores.append(
                task.get_bb_score(np.array(test_solution), parsed_completion)
                if isinstance(parsed_completion, np.ndarray)
                else 0
            )
            test_outputs.append(str(parsed_completion))

        # Evaluate training example
        train_scores = []
        for train_problem, train_solution in zip(training_dataset[0]["problem"], training_dataset[0]["solution"]):
            train_completion = task.decode_response(completion, input_grid=np.array(train_problem))
            if params["use_induction"]:
                train_completion = "Error executing code" if train_completion.size == 0 else train_completion
            else:
                train_completion = completion if train_completion.size == 0 else train_completion
            train_score = (
                task.get_bb_score(np.array(train_solution), train_completion)
                if isinstance(train_completion, np.ndarray)
                else 0
            )
            train_scores.append(train_score)

        # Track individual programs and their scores
        results.append(
            {
                "output": test_outputs,
                "test_score": test_scores,
                "train_score": np.mean(train_scores),
                "code": completion,
            }
        )

    # Compute & record summary metrics
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
    results = sorted(results, key=lambda x: np.mean(x["test_score"]), reverse=True)

    oracle = max(results, key=lambda x: np.mean(x["test_score"]))
    oracle_votes = len([x for x in results if str(x["output"]) == str(oracle["output"])])
    train_solved = [x["train_score"] == 1 for x in results]
    test_solved = [np.mean(x["test_score"]) == 1 for x in results]

    data["oracle"] = {"output": oracle["output"], "score": oracle["test_score"], "count": oracle_votes}
    data["true_positives"] = np.sum([int(x == 1 and y == 1) for x, y in zip(train_solved, test_solved)]) / np.sum(
        test_solved
    )
    data["false_negatives"] = np.sum([int(x == 0 and y == 1) for x, y in zip(train_solved, test_solved)]) / np.sum(
        test_solved
    )
    data["true_negatives"] = np.sum([int(x == 0 and y == 0) for x, y in zip(train_solved, test_solved)]) / (
        len(test_solved) - np.sum(test_solved)
    )
    data["false_positives"] = np.sum([int(x == 1 and y == 0) for x, y in zip(train_solved, test_solved)]) / (
        len(test_solved) - np.sum(test_solved)
    )

    # Compute & log pass@1 and pass@2 results
    filtered_results = [x for x in results if x["train_score"] == 1 and "Error executing code" not in x["output"]]
    if len(filtered_results) > 0:
        data = get_majority_vote(data, filtered_results)

    return data
