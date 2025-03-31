import json
import random
import itertools
import numpy as np
import torch
from torch import nn
from typing import Any, Union
from trl import (
    GRPOTrainer,
    maybe_apply_chat_template,
    is_conversational,
    apply_chat_template,
)
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import print_prompt_completions_sample, selective_log_softmax

from trl.extras.profiling import profiling_context
from trl.import_utils import is_rich_available
from accelerate.utils import broadcast_object_list, gather_object, gather
from transformers import is_wandb_available, ProgressCallback
from transformers.utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
)
import prompts as prompts_getter
import arc_utils.utils as arc_utils

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_wandb_available():
    import wandb

if is_apex_available():
    from apex import amp


def log_response(response, logfile):
    with open(logfile, "r") as file:
        data = json.load(file)
        data["guesses"].append(response)
    with open(logfile, "w") as file:
        json.dump(data, file, indent=2)


class CustomProgressCallback(ProgressCallback):
    def __init__(self):
        super(CustomProgressCallback, self).__init__()
        self.loss = None
        self.val_acc = None
        self.train_acc = None
        self.train_acc_max = None

    def on_step_end(self, args, state, control, **kwargs):
        super(CustomProgressCallback, self).on_step_end(args, state, control, **kwargs)

        if state.is_world_process_zero:
            self.training_bar.set_postfix({
                "loss": self.loss,
                "train_reward_mean": self.train_acc,
                "train_reward_max": self.train_acc_max,
                "val_reward_majority": self.val_acc,
            })

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            self.training_bar.write("")


class GRPOTrainer(GRPOTrainer):
    def __init__(
            self,
            target,
            strategy,
            logfile,
            task,
            arc_dataset_file,
            validation_dataset,
            validation_interval,
            generation_args,
            grpo_weight,
            nll_weight,
            pro_loss_weight,
            train_temperature=1.0,
            use_train_temp_schedule=False,
            inf_batch_size=10,
            inject_best_at_lowest_score=False,
            pro_loss_only_positive=False,
            use_early_stopping=True,
            neighborhood_sampling=False,
            neighborhood_sampling_strategy="best",
            n_neighbors=10,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.iteration = 0
        self.logfile = logfile
        self.target = target
        self.past_guesses = {}
        self.strategy = strategy
        self.neighborhood_sampling = neighborhood_sampling
        self.neighborhood_sampling_strategy = neighborhood_sampling_strategy
        self.n_neighbors = n_neighbors
        self.task = task

        self.arc_sol = None
        self.arc_leave_out = None
        self.arc_past_guesses = {}
        self.arc_dataset_file = arc_dataset_file
        self.validation_dataset = validation_dataset
        self.validation_interval = validation_interval

        self.generation_args = generation_args
        self.grpo_weight = grpo_weight
        self.nll_weight = nll_weight
        self.pro_loss_weight = pro_loss_weight
        self.train_temperature = train_temperature
        self.use_train_temp_schedule = use_train_temp_schedule
        self.inf_batch_size = inf_batch_size
        self.inject_best_at_lowest_score = inject_best_at_lowest_score
        self.pro_loss_only_positive = pro_loss_only_positive
        self.use_early_stopping = use_early_stopping

        for callback in self.callback_handler.callbacks:
            if type(callback) is CustomProgressCallback:
                self.progress_callback = callback

    # Compute black-box score
    def get_bb_score(self, completion1, completion2, verbose=True):
        score = 1 - arc_utils.hamming_distance(completion1, completion2)
        # if verbose:
        #     print("ATTEMPT:\n", completion2)
        #     print("SOLUTION:\n", completion1)
        #     print("SCORE:", score)
        return score

    # Record new completions and their black-box scores
    def update_past_guesses(self, responses, bb_scores):
        guesses = list(itertools.chain.from_iterable(responses))
        scores = list(itertools.chain.from_iterable(bb_scores))
        for word, score in zip(guesses, scores):
            self.past_guesses[word] = score
        self.arc_past_guesses[self.arc_leave_out] = self.past_guesses.copy()

    # Neighborhood sampling
    def get_neighborhood_samples(self, problem, solution, n_samples, unique=True):
        prompt_obj = prompts_getter.get_arc_neighborhood_samples_prompt(str(problem), str(solution))
        prompts_text = [maybe_apply_chat_template({"prompt": prompt_obj}, self.processing_class)["prompt"]] * n_samples
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        completions = []
        for i in range(0, len(prompt_ids), self.inf_batch_size):
            _prompt_ids, _prompt_mask = prompt_ids[i:i + self.inf_batch_size], prompt_mask[i:i + self.inf_batch_size]
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator,
                                             gather_deepspeed3_params=self.args.ds3_gather_for_generation) as unwrapped_model:
                completion_ids = unwrapped_model.generate(
                    input_ids=_prompt_ids.to(self.args.device),
                    attention_mask=_prompt_mask.to(self.args.device),
                    generation_config=self.generation_config,
                    temperature=self.temperature,
                    max_new_tokens=self.max_completion_length,
                    **self.generation_args,
                )
                prompt_length = _prompt_ids.size(1)
                completions.extend(
                    self.processing_class.batch_decode(
                        completion_ids[:, prompt_length:], skip_special_tokens=True
                    )
                )

        if unique:
            completions = list(set(completions))

        bb_scores, responses = [], []
        for completion in completions:
            guesses = [arc_utils.parse_response(completion)]
            bb_scores.append([self.get_bb_score(self.arc_sol, guess) for guess in guesses])
            responses.append([completion if x.size == 0 else str(x) for x in guesses])

        completions = list(itertools.chain.from_iterable(responses))
        rewards = list(itertools.chain.from_iterable(bb_scores))

        # Zip and sort completions and rewards
        completions, rewards = zip(*sorted(zip(completions, rewards), key=lambda x: x[1], reverse=True))
        completions, rewards = list(completions), list(rewards)

        return completions, rewards

    def run_validation(self):
        print("\n==================\nRUNNING VALIDATION\n==================")
        prompts = self.validation_dataset["dataset"]
        prompts_text = [
            maybe_apply_chat_template({"prompt": example}, self.processing_class)["prompt"] for example in prompts
        ]
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                with profiling_context(self, "vLLM.generate"):
                    completion_ids = self.vllm_client.generate(
                        prompts=ordered_set_of_prompts,
                        n=self.num_generations,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=self.max_completion_length,
                        guided_decoding_regex=self.guided_decoding_regex,
                    )
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=self.args.device) for ids in completion_ids]

            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        else:
            # Regular generation path
            prompt_inputs = self.processing_class(
                prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            completions = []
            for i in range(0, len(prompt_inputs["input_ids"]), self.inf_batch_size):
                prompt_ids, prompt_mask = prompt_inputs["input_ids"][i:i + self.inf_batch_size], prompt_inputs[
                                                                                                     "attention_mask"][
                                                                                                 i:i + self.inf_batch_size
                                                                                                 ]
                with unwrap_model_for_generation(self.model_wrapped, self.accelerator,
                                                 gather_deepspeed3_params=self.args.ds3_gather_for_generation) as unwrapped_model:
                    completion_ids = unwrapped_model.generate(
                        input_ids=prompt_ids.to(self.args.device),
                        attention_mask=prompt_mask.to(self.args.device),
                        do_sample=False,
                        max_new_tokens=self.max_completion_length,
                    )
                    prompt_length = prompt_inputs["input_ids"].size(1)
                    completions.extend(
                        self.processing_class.batch_decode(
                            completion_ids[:, prompt_length:], skip_special_tokens=True
                        )
                    )

        # Aggregate results
        results = {}
        for completion in completions:
            parsed_completion = arc_utils.parse_response(completion)
            parsed_completion = completion if parsed_completion.size == 0 else parsed_completion
            # Get black-box score if completion is valid otherwise 0
            score = (
                self.get_bb_score(self.validation_dataset["solution"], parsed_completion)
                if isinstance(parsed_completion, np.ndarray)
                else 0
            )
            parsed_completion = str(parsed_completion)
            # Track completions and their scores
            results[parsed_completion] = results.get(parsed_completion, {"score": score, "count": 0})
            results[parsed_completion]["count"] += 1

        # Do pass@2 majority voting -- also make sure majority is more than 1
        sorted_majority = sorted(results.items(), key=lambda x: x[1]["count"], reverse=True)
        if self.use_early_stopping and any(
                x[1]["score"] == 1.0 for x in sorted_majority[:2]):  # and sorted_majority[0][1]["count"] > 1:
            self.control.should_training_stop = True
            print("EARLY STOPPING: Validation majority voting (pass@2) reached 100% accuracy.")

        # Update training bar
        self.progress_callback.val_acc = np.round(sorted_majority[0][1]["score"], 4)
        wandb.log({"validation/majority_reward": sorted_majority[0][1]["score"]})
        wandb.log({"validation/majority_count": sorted_majority[0][1]["count"]})
        wandb.log({"validation/total_count": len(completions)})

        # print("VALIDATION SOLUTION", self.validation_dataset["solution"])
        # print("VALIDATION ATTEMPT", sorted_majority[0][0])
        # print("VALIDATION SCORE", sorted_majority[0][1]["score"])

        # Log validation results
        with open(self.logfile, "r") as file:
            data = json.load(file)
        data["validation"].append(
            {
                "iteration": self.iteration,
                "prompts": prompts_text,
                "results": [{"completion": x[0], "score": x[1]["score"], "count": x[1]["count"]} for x in
                            sorted(results.items(), key=lambda x: x[1]["count"], reverse=True)]
            }
        )
        with open(self.logfile, "w") as file:
            json.dump(data, file, indent=2)
        # Print scores
        print("Reward for the top-2 majority-voted completions:", [x[1]["score"] for x in sorted_majority[:2]])
        print(f"Validation results saved to {self.logfile}\n")

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        if self.iteration % self.validation_interval == 0:
            self.run_validation()
        mode = "eval" if self.control.should_evaluate or self.control.should_training_stop else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
            self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Run validation check

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]

        self.arc_leave_out = str(inputs[0]["solution"])
        self.arc_prob = np.array(inputs[0]["problem"])
        self.arc_sol = np.array(inputs[0]["solution"])

        # Load/initialize past guesses according to leave-out
        if self.arc_leave_out in self.arc_past_guesses:
            self.past_guesses = self.arc_past_guesses[self.arc_leave_out]
        else:
            self.past_guesses = {}
            self.arc_past_guesses[self.arc_leave_out] = self.past_guesses

        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        responses = []
        bb_scores = []
        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                with profiling_context(self, "vLLM.generate"):
                    completion_ids = self.vllm_client.generate(
                        prompts=ordered_set_of_prompts,
                        n=self.num_generations,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=self.max_completion_length,
                        guided_decoding_regex=self.guided_decoding_regex,
                    )
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]

            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        else:
            # Regular generation path
            completions = []
            for i in range(0, len(prompt_ids), self.inf_batch_size):
                _prompt_ids, _prompt_mask = prompt_ids[i:i + self.inf_batch_size], prompt_mask[
                                                                                   i:i + self.inf_batch_size
                                                                                   ]
                with unwrap_model_for_generation(self.model_wrapped, self.accelerator,
                                                 gather_deepspeed3_params=self.args.ds3_gather_for_generation) as unwrapped_model:
                    completion_ids = unwrapped_model.generate(
                        input_ids=_prompt_ids.to(device),
                        attention_mask=_prompt_mask.to(device),
                        generation_config=self.generation_config,
                        temperature=self.temperature,
                        max_new_tokens=self.max_completion_length,
                        **self.generation_args,
                    )
                    prompt_length = _prompt_ids.size(1)
                    completions.extend(
                        self.processing_class.batch_decode(
                            completion_ids[:, prompt_length:], skip_special_tokens=True
                        )
                    )

        for completion in completions:
            guesses = [arc_utils.parse_response(completion)]
            bb_scores.append([self.get_bb_score(self.arc_sol, guess) for guess in guesses])
            responses.append([completion if x.size == 0 else str(x) for x in guesses])
        # Sort by rewards
        completions, bb_scores, responses = zip(
            *sorted(zip(completions, bb_scores, responses), key=lambda x: x[1], reverse=True)
        )
        completions, bb_scores, responses = list(completions), list(bb_scores), list(responses)

        # Create completions and corresponding rewards based on the strategy
        completions = list(itertools.chain.from_iterable(responses))
        rewards = list(itertools.chain.from_iterable(bb_scores))
        if self.strategy == "gold":
            # Substitute in the gold/oracle solution
            gold_solution = (str(self.arc_sol), 1.0)

            if self.neighborhood_sampling:
                completions, rewards = arc_utils.run_neighborhood_sampling(self, completions, rewards, gold_solution[0],
                                                                           n_neighbors=self.n_neighbors)

            arc_utils.add_to_batch(self, gold_solution, completions, rewards)
        elif self.strategy == "greedy":
            # Substitute in the best solution generated so far
            best_guess = None
            if len(self.past_guesses) > 0:
                best_guess = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)[0]

            if self.neighborhood_sampling:
                completions, rewards = arc_utils.run_neighborhood_sampling(self, completions, rewards, best_guess[0],
                                                                           n_neighbors=self.n_neighbors)

            arc_utils.add_to_batch(self, best_guess, completions, rewards)
        elif self.strategy == "top_delta":
            raise NotImplementedError
        else:
            # Keep completions and rewards based on the online generated samples
            self.best_idx_replaced = None

            if self.neighborhood_sampling:
                completions, rewards = arc_utils.run_neighborhood_sampling(self, completions, rewards, None,
                                                                           n_neighbors=self.n_neighbors)

        # Compute mean and max rewards excluding the "greedy" replacement
        mean_reward_minus_replacement = np.mean(
            [rewards[i] for i in range(len(rewards)) if i != self.best_idx_replaced])
        max_reward_minus_replacement = np.max(
            [rewards[i] for i in range(len(rewards)) if i != self.best_idx_replaced])
        # Update training bar
        self.progress_callback.train_acc = np.round(mean_reward_minus_replacement, 4)
        wandb.log({"train/reward_minus_replacement_mean": mean_reward_minus_replacement})
        self.progress_callback.train_acc_max = np.round(max_reward_minus_replacement, 4)
        wandb.log({"train/reward_minus_replacement_max": max_reward_minus_replacement})

        # Record guesses into history and log repsonses
        self.update_past_guesses(responses, bb_scores)
        log_response(
            {
                f"iteration_{self.iteration}": [
                    (x, y)
                    for x, y in zip(
                        list(itertools.chain.from_iterable(responses)),
                        list(itertools.chain.from_iterable(bb_scores)),
                    )
                ],
                "problem": str(self.arc_prob),
                "solution": self.arc_leave_out,  # Record for tracing trajectory in post
            },
            self.logfile,
        )

        # Create final completions for computing loss
        prompt = {"prompt": prompts[0]}
        prompt_text = [maybe_apply_chat_template(prompt, self.processing_class)["prompt"]]
        prompt_ids = self.processing_class(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        completion_ids = self.processing_class(
            completions, return_tensors="pt", add_special_tokens=False, padding=True
        ).input_ids
        prompt_inputs_repeated = torch.repeat_interleave(prompt_ids, len(completion_ids), dim=0)
        prompt_completion_ids = torch.cat([prompt_inputs_repeated, completion_ids], dim=1).to(device)

        self.num_generations = len(completions)
        self.iteration += 1

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_mask = prompt_mask.to(device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Convert rewards to tensor
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)
        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()
            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd
                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def _get_per_token_logps_clone(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        _temp = self.train_temperature
        if self.use_train_temp_schedule:
            # Linearly scale temperature to 1 starting from 0 based on global training step
            _temp = self.train_temperature * (self.state.global_step / self.state.max_steps)
            _temp += (0 if _temp > 0 else 1e-2)
        logits = logits / _temp
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = None

        if self.grpo_weight > 0:
            loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            # TODO: Add support for using GRPO + NLL
        else:
            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
            per_token_logps = self._get_per_token_logps_clone(model, input_ids, attention_mask, logits_to_keep)

            completion_logps = None
            if self.nll_weight > 0:
                # breakpoint()
                completion_logps = (per_token_logps * attention_mask[:, -logits_to_keep:]).sum(dim=1) / attention_mask[
                                                                                                        :,
                                                                                                        -logits_to_keep:].sum(
                    dim=1)
                # Use the gold completion logp for NLL
                oracle_logp = completion_logps[self.best_idx_replaced]
                nll_loss = self.nll_weight * -oracle_logp
                if loss is None or loss == 0:
                    loss = nll_loss
                else:
                    loss += nll_loss
                self._metrics["train"]["nll_loss"].append(nll_loss.item())

            if self.pro_loss_weight > 0:
                if completion_logps is None:
                    completion_logps = (per_token_logps * attention_mask[:, -logits_to_keep:]).sum(
                        dim=1) / attention_mask[
                                 :,
                                 -logits_to_keep:].sum(
                        dim=1)
                sorted_order = torch.argsort(inputs["advantages"], descending=True)
                sorted_completion_logps = completion_logps[sorted_order]
                if len(sorted_completion_logps.size()) == 1:
                    sorted_completion_logps = sorted_completion_logps.unsqueeze(0)

                ignore_idx = None
                if self.pro_loss_only_positive:
                    try:
                        # Get the first negative advantage in the sorted order
                        first_zero_or_negative = torch.where(inputs["advantages"][sorted_order] <= 0)[0][0].item()
                        ignore_idx = first_zero_or_negative + 1
                    except:
                        print("No negative advantage found, using all completions for PRO loss")
                        pass

                _pro_loss = self.pro_loss_weight * arc_utils.pro_loss(sorted_completion_logps,
                                                                      start_neg_idx_to_ignore=ignore_idx)
                if loss is None or loss == 0:
                    loss = _pro_loss
                else:
                    loss += _pro_loss
                self._metrics["train"]["pro_loss"].append(_pro_loss.item())

        self.progress_callback.loss = np.round(loss.item(), 4)

        assert loss is not None, "At least one of the losses should be computed"
        return loss
