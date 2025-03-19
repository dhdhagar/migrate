import ast
import json
import random
import itertools
import numpy as np
import torch
from torch import nn
from typing import Any, Dict, Union
from trl import (
    GRPOTrainer,
    maybe_apply_chat_template,
    is_conversational,
    apply_chat_template,
)
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import pad
from accelerate.utils import broadcast_object_list, gather_object
from transformers import is_wandb_available, PreTrainedModel
from transformers.utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
    is_torch_xpu_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_mps_available,
)
from transformers.training_args import OptimizerNames
import prompts as prompts_getter
import arc_utils.utils as arc_utils

# from sift import SIFT, CTL, VTL, ITL
# from chem_utils import compute_qed_vina_score


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
        json.dump(data, file, indent=4)


def log_chosen(response, logfile):
    with open(logfile, "r") as file:
        data = json.load(file)
        data["chosen"].append(response)
    with open(logfile, "w") as file:
        json.dump(data, file, indent=4)


class GRPOTrainer(GRPOTrainer):
    def __init__(
            self,
            target,
            strategy,
            n_reps,
            logfile,
            sample_related,
            task,
            arc_dataset_file,
            validation_example,
            generation_args,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.iteration = 0
        self.logfile = logfile
        self.target = target
        self.past_guesses = {}
        self.strategy = strategy
        self.n_reps = n_reps
        self.sample_related = sample_related
        self.task = task

        self.arc_sol = None
        self.arc_leave_out_idx = None
        self.arc_past_guesses = {}
        self.arc_dataset_file = arc_dataset_file
        self.validation_example = validation_example

        self.generation_args = generation_args

        self.continue_training = True

    # Compute black-box score
    def get_bb_score(self, completion1, completion2, verbose=True):
        score = 1 - arc_utils.hamming_distance(completion1, completion2)
        if verbose:
            print("ATTEMPT:\n", completion2)
            print("SOLUTION:\n", completion1)
            print("SCORE:", score)
        return score

    # Record new completions and their black-box scores
    def update_past_guesses(self, responses, bb_scores):
        guesses = list(itertools.chain.from_iterable(responses))
        scores = list(itertools.chain.from_iterable(bb_scores))
        for word, score in zip(guesses, scores):
            self.past_guesses[word] = score
        self.arc_past_guesses[self.arc_leave_out_idx] = self.past_guesses.copy()

    # Neighborhood sampling
    # TODO: Add support for ARC
    def sample_related_completions(self, model, chosen_completion, n):
        inputs = {"prompt": prompts_getter.get_semantle_related_prompt(n, str(chosen_completion))}
        inputs = [maybe_apply_chat_template(inputs, self.processing_class)["prompt"]]
        prompt_inputs = self.processing_class(
            inputs, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids = inputs["input_ids"]
        prompt_mask = inputs["attention_mask"]
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            completion_ids = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
                **self.generation_args
            )

        related_completions = self.processing_class.decode(
            completion_ids[0, prompt_ids.size(1):], skip_special_tokens=True
        )
        try:
            related_completions = json.loads(related_completions)["response"]
            return related_completions
        except Exception as _:
            return None

    def run_validation(self, model: nn.Module):
        model.eval()
        with torch.no_grad():
            prompts_text = [maybe_apply_chat_template(self.validation_example, self.processing_class)["prompt"]]
            prompt_inputs = self.processing_class(
                prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            completion_ids = model.generate(
                input_ids=prompt_inputs["input_ids"].to(self.args.device),
                attention_mask=prompt_inputs["attention_mask"].to(self.args.device),
                max_new_tokens=512,
                # num_return_sequences=5,
                # temperature=0.3,
                do_sample=False,
                **self.generation_args
            )
            prompt_length = prompt_inputs["input_ids"].size(1)
            completions = self.processing_class.batch_decode(
                completion_ids[:, prompt_length:], skip_special_tokens=True
            )
            attempts = [arc_utils.parse_response(x) for x in completions]
            del completion_ids
            del prompt_inputs
            scores = []
            for attempt in attempts:
                if len(attempt) > 0:
                    scores.append(self.get_bb_score(self.validation_example["solution"], attempt))
                else:
                    scores.append(0)

            print("VALIDATION HAMMING", scores)
            with open(self.logfile, "r") as file:
                data = json.load(file)
                data["validation"].append(list(zip([str(x) for x in attempts], scores)))
            with open(self.logfile, "w") as file:
                json.dump(data, file, indent=4)
        return scores

    @staticmethod
    def get_per_token_logps(model, input_ids, num_logits_to_keep):
        # Get the per-token log probabilities for the completions for the model and the reference model

        # We add 1 to `num_logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids, num_logits_to_keep=num_logits_to_keep + 1).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids[:, -num_logits_to_keep:]):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        if not self.continue_training:  # TODO: Use EarlyStoppingCallback
            return torch.tensor(0.0, dtype=torch.float32, device=self.args.device).detach()

        # Evaluate validation after every epoch
        if self.iteration % self.args.gradient_accumulation_steps == 0:
            scores = self.run_validation(model)
            # if scores.count(0) >= 4:
            if scores.count(1.) > 0:  # Number of solved instances
                self.continue_training = False
                return torch.tensor(0.0, dtype=torch.float32, device=self.args.device).detach()
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        self.arc_leave_out_idx = inputs[0]["leave_out_idx"]
        self.arc_sol = np.array(inputs[0]["solution"])
        # print("LEAVE OUT IDX:", self.arc_leave_out_idx)
        # print("GOLD SOLUTION\n", self.arc_sol)

        # Load/initialize past guesses according to leave-out
        if self.arc_leave_out_idx in self.arc_past_guesses:
            self.past_guesses = self.arc_past_guesses[self.arc_leave_out_idx]
        else:
            self.past_guesses = {}
            self.arc_past_guesses[self.arc_leave_out_idx] = self.past_guesses
        # Training-Oracle: Add the gold solution to the past guesses (greedy_single will select this as chosen)
        sol_str = str(self.arc_sol)
        self.past_guesses[sol_str] = 1.0  # Black-box score of 1

        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]

        # Generate completions using either vLLM or regular generation
        # TODO: Make this support unsloth to use vLLM
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights(state_dict.items())
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False,
                                            **self.generation_args)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text) * self.num_generations

            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts) * self.num_generations,
                (self.accelerator.process_index + 1) * len(prompts) * self.num_generations,
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_inputs_repeated = torch.repeat_interleave(prompt_inputs["input_ids"], self.num_generations, dim=0)
            prompt_completion_ids = torch.cat([prompt_inputs_repeated, completion_ids], dim=1)
        else:
            # Regular generation path
            try:
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    responses = []
                    invalid_responses = []
                    bb_scores = []
                    retries = 0
                    # Keep generating `retries` times until we have `self.n_reps` valid responses
                    while len(responses) < self.n_reps and retries < 2:
                        completion_ids = unwrapped_model.generate(
                            input_ids=prompt_inputs["input_ids"].to(device),
                            attention_mask=prompt_inputs["attention_mask"].to(device),
                            generation_config=self.generation_config,
                            num_return_sequences=self.n_reps,
                            **self.generation_args
                        )
                        prompt_length = prompt_inputs["input_ids"].size(1)
                        completions = self.processing_class.batch_decode(
                            completion_ids[:, prompt_length:], skip_special_tokens=True
                        )
                        for completion in completions:
                            try:
                                guesses = [arc_utils.parse_response(completion)]
                                scores = [self.get_bb_score(guess, self.arc_sol) for guess in guesses]
                                bb_scores.append(scores)
                                responses.append([str(x) for x in guesses])
                            except Exception as _:
                                invalid_responses.append(completion)
                            if len(responses) == self.n_reps:
                                break
                        retries += 1
                    if len(responses) < 2:
                        raise Exception("Not enough valid responses")

                    # Create completions and corresponding rewards
                    # TODO: Refactor (self.strategy, repsonses, bb_scores) -> (completions, rewards)
                    completions, rewards = [], []
                    if self.strategy == "Oracle_Single":
                        completions = list(itertools.chain.from_iterable(responses))
                        rewards = list(itertools.chain.from_iterable(bb_scores))
                        # Substitute a random guess with the target
                        idx = random.randint(0, len(completions) - 1)
                        completions[idx] = self.target
                        rewards[idx] = 1.0
                    elif self.strategy == "Online_Single":
                        completions = list(itertools.chain.from_iterable(responses))
                        rewards = list(itertools.chain.from_iterable(bb_scores))
                    elif self.strategy == "Online_Mean":
                        completions = responses
                        rewards = [np.mean(scores) for scores in bb_scores]
                    elif self.strategy == "Online_Max":
                        completions = responses
                        rewards = [np.max(scores) for scores in bb_scores]
                    elif self.strategy == "Online_Batch_Mean":
                        completions = list(itertools.chain.from_iterable(responses))
                        scores = list(itertools.chain.from_iterable(bb_scores))
                        word_scores = [[word, score] for word, score in zip(completions, scores)]
                        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
                        word_scores = [word_scores[i: i + 2] for i in range(0, len(word_scores), 2)]
                        completions = [[x[0][0], x[1][0]] for x in word_scores]
                        rewards = [np.mean([x[0][1], x[1][1]]) for x in word_scores]
                    elif self.strategy == "Online_Batch_Max":
                        completions = list(itertools.chain.from_iterable(responses))
                        scores = list(itertools.chain.from_iterable(bb_scores))
                        word_scores = [[word, score] for word, score in zip(completions, scores)]
                        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
                        word_scores = [word_scores[i: i + 2] for i in range(0, len(word_scores), 2)]
                        completions = [[x[0][0], x[1][0]] for x in word_scores]
                        rewards = [np.max([x[0][1], x[1][1]]) for x in word_scores]
                    elif self.strategy == "Greedy_Single":
                        completions = list(itertools.chain.from_iterable(responses))
                        rewards = list(itertools.chain.from_iterable(bb_scores))
                        # Substitute a random guess with the best guess so far
                        if len(self.past_guesses) > 0:
                            best_guess = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)[0]
                            idx = random.randint(0, len(completions) - 1)
                            completions[idx] = best_guess[0]
                            rewards[idx] = best_guess[1]
                    elif self.strategy == "Greedy_Batch_Mean":
                        completions = list(itertools.chain.from_iterable(responses))
                        scores = list(itertools.chain.from_iterable(bb_scores))
                        word_scores = [[word, score] for word, score in zip(completions, scores)]
                        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
                        word_scores = [word_scores[i: i + 2] for i in range(0, len(word_scores), 2)]
                        # Substitute a random guess batch with the best guess batch so far
                        if len(self.past_guesses) > 0:
                            best_guess = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)[:2]
                            idx = random.randint(0, len(word_scores) - 1)
                            word_scores[idx] = best_guess  # type:ignore
                        completions = [[x[0][0], x[1][0]] for x in word_scores]
                        rewards = [np.mean([x[0][1], x[1][1]]) for x in word_scores]
                    elif self.strategy == "Greedy_Batch_Max":
                        completions = list(itertools.chain.from_iterable(responses))
                        scores = list(itertools.chain.from_iterable(bb_scores))
                        word_scores = [[word, score] for word, score in zip(completions, scores)]
                        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
                        word_scores = [word_scores[i: i + 2] for i in range(0, len(word_scores), 2)]
                        # Substitute a random guess batch with the best guess batch so far
                        if len(self.past_guesses) > 0:
                            best_guess = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)[:2]
                            idx = random.randint(0, len(word_scores) - 1)
                            word_scores[idx] = best_guess  # type:ignore
                        completions = [[x[0][0], x[1][0]] for x in word_scores]
                        rewards = [np.max([x[0][1], x[1][1]]) for x in word_scores]
                        # rewards = []
                        # rewards.append([np.max([x[0][1], x[1][1]]) for x in word_scores])
                        # rewards.append([int(x[0][0] != x[1][0]) for x in word_scores])
                    elif self.strategy == "TopDelta_Batch_Mean":
                        completions = list(itertools.chain.from_iterable(responses))
                        scores = list(itertools.chain.from_iterable(bb_scores))
                        word_scores = [[word, score] for word, score in zip(completions, scores)]
                        random.shuffle(word_scores)
                        if len(self.past_guesses) > 0:
                            word_scores = word_scores[:6]
                            past_guesses = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)
                            word_scores = word_scores + past_guesses[:2] + past_guesses[-2:]
                        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
                        word_scores = [word_scores[i: i + 2] for i in range(0, len(word_scores), 2)]
                        completions = [[x[0][0], x[1][0]] for x in word_scores]
                        rewards = [np.mean([x[0][1], x[1][1]]) for x in word_scores]
                    elif self.strategy == "TopDelta_Batch_Max":
                        completions = list(itertools.chain.from_iterable(responses))
                        scores = list(itertools.chain.from_iterable(bb_scores))
                        word_scores = [[word, score] for word, score in zip(completions, scores)]
                        random.shuffle(word_scores)
                        if len(self.past_guesses) > 0:
                            word_scores = word_scores[:6]
                            past_guesses = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)
                            word_scores = word_scores + past_guesses[:2] + past_guesses[-2:]
                        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
                        word_scores = [word_scores[i: i + 2] for i in range(0, len(word_scores), 2)]
                        completions = [[x[0][0], x[1][0]] for x in word_scores]
                        rewards = [np.max([x[0][1], x[1][1]]) for x in word_scores]
                    elif self.strategy == "Greedy_Batch_Mean_Related":
                        completions = list(itertools.chain.from_iterable(responses))
                        scores = list(itertools.chain.from_iterable(bb_scores))
                        word_scores = [[word, score] for word, score in zip(completions, scores)]
                        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
                        word_scores = [word_scores[i: i + 2] for i in range(0, len(word_scores), 2)]
                        # Substitute a random guess batch with the best guess batch so far
                        best_guesses = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)
                        if len(self.past_guesses) > 0:
                            random.shuffle(word_scores)
                            word_scores[0] = best_guesses[:2]  # type:ignore
                        completions = [[x[0][0], x[1][0]] for x in word_scores]
                        rewards = [np.mean([x[0][1], x[1][1]]) for x in word_scores]

                        related_completions = None
                        if len(self.past_guesses) > 0:
                            related_completions = self.sample_related_completions(
                                unwrapped_model, best_guesses[0][0], 5
                            )
                        if related_completions is not None:
                            print("RELATED", related_completions)
                            print("BEFORE", completions)
                            print("BEFORE", rewards)
                            random.shuffle(related_completions)
                            related_completions = related_completions[:4]
                            related_completions = [[x, self.get_bb_score(x, self.target)] for x in related_completions]
                            idx = random.sample(range(1, 5), 3)
                            # old_best = word_scores[0]
                            word_scores = (
                                    related_completions + word_scores[idx[0]] + word_scores[idx[1]] + word_scores[
                                idx[2]]
                            )
                            word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
                            # word_scores = old_best + word_scores
                            word_scores = [word_scores[i: i + 2] for i in range(0, len(word_scores), 2)]
                            completions = [[x[0][0], x[1][0]] for x in word_scores]
                            rewards = [np.mean([x[0][1], x[1][1]]) for x in word_scores]
                            print("AFTER", completions)
                            print("AFTER", rewards)
                    elif self.strategy == "Greedy_Batch_Max_Related":
                        completions = list(itertools.chain.from_iterable(responses))
                        scores = list(itertools.chain.from_iterable(bb_scores))
                        word_scores = [[word, score] for word, score in zip(completions, scores)]
                        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
                        word_scores = [word_scores[i: i + 2] for i in range(0, len(word_scores), 2)]
                        # Substitute a random guess batch with the best guess batch so far
                        best_guesses = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)
                        if len(self.past_guesses) > 0:
                            random.shuffle(word_scores)
                            word_scores[0] = best_guesses[:2]  # type:ignore
                        completions = [[x[0][0], x[1][0]] for x in word_scores]
                        rewards = [np.max([x[0][1], x[1][1]]) for x in word_scores]

                        related_completions = None
                        if len(self.past_guesses) > 0:
                            related_completions = self.sample_related_completions(
                                unwrapped_model, best_guesses[0][0], 5
                            )
                        if related_completions is not None:
                            print("RELATED", related_completions)
                            print("BEFORE", completions)
                            print("BEFORE", rewards)
                            random.shuffle(related_completions)
                            related_completions = related_completions[:4]
                            related_completions = [[x, self.get_bb_score(x, self.target)] for x in related_completions]
                            idx = random.sample(range(1, 5), 3)
                            # old_best = word_scores[0]
                            word_scores = (
                                    related_completions + word_scores[idx[0]] + word_scores[idx[1]] + word_scores[
                                idx[2]]
                            )
                            word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
                            # word_scores = old_best + word_scores
                            word_scores = [word_scores[i: i + 2] for i in range(0, len(word_scores), 2)]
                            completions = [[x[0][0], x[1][0]] for x in word_scores]
                            rewards = [np.max([x[0][1], x[1][1]]) for x in word_scores]
                            print("AFTER", completions)
                            print("AFTER", rewards)

                    print("COMPLETIONS:\n", completions)
                    print("REWARDS:\n", rewards)
                    print("MEAN REWARD:", np.mean(rewards))

                    # Record guesses into history and log repsonses
                    self.update_past_guesses(responses, bb_scores)
                    log_response(
                        {
                            f"Iteration: {self.iteration}": [
                                (x, y)
                                for x, y in zip(
                                    list(itertools.chain.from_iterable(responses)),
                                    list(itertools.chain.from_iterable(bb_scores)),
                                )
                            ],
                            "leave_out_idx": self.arc_leave_out_idx,  # Record for tracing trajectory in post
                        },
                        self.logfile,
                    )

                    # Create final completions for computing loss
                    prompt = {"prompt": prompts[0]}
                    prompt_text = [maybe_apply_chat_template(prompt, self.processing_class)["prompt"]]
                    prompt_inputs = self.processing_class(prompt_text, return_tensors="pt", add_special_tokens=False)
                    completion_ids = self.processing_class(
                        completions, return_tensors="pt", add_special_tokens=False, padding=True
                    ).input_ids
                    prompt_inputs_repeated = torch.repeat_interleave(
                        prompt_inputs["input_ids"], len(completion_ids), dim=0
                    )
                    prompt_completion_ids = torch.cat([prompt_inputs_repeated, completion_ids], dim=1).to(device)

                    self.num_generations = len(completions)
                    self.iteration += 1
            except Exception as _:
                # Add gold solution
                invalid_responses.append(str(self.arc_sol))

                # Create rewards for invalid responses + gold solution
                rewards  = [0.0] * len(invalid_responses)
                rewards[-1] = 1.0
                
                prompt = {"prompt": prompts[0]}
                prompt_text = [maybe_apply_chat_template(prompt, self.processing_class)["prompt"]]
                prompt_inputs = self.processing_class(prompt_text, return_tensors="pt", add_special_tokens=False)
                completion_ids = self.processing_class(
                    invalid_responses, return_tensors="pt", add_special_tokens=False, padding=True
                ).input_ids
                prompt_inputs_repeated = torch.repeat_interleave(
                    prompt_inputs["input_ids"], len(completion_ids), dim=0
                )
                prompt_completion_ids = torch.cat([prompt_inputs_repeated, completion_ids], dim=1).to(device)
                
                log_response({f"Iteration: {self.iteration}": [], "leave_out_idx": self.arc_leave_out_idx}, self.logfile)
                self.num_generations = len(invalid_responses)
                self.iteration += 1

        # Only compute loss if completion was valid and there are more than 1 completion in the group
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        num_logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        per_token_logps = self.get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep).to(device)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self.get_per_token_logps(
                    self.ref_model, prompt_completion_ids, num_logits_to_keep
                ).to(device)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self.get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep).to(
                        device
                    )

        # Compute the KL divergence between the model and the reference model
        per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        )

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)].to(device)
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        num_reward_funcs = len(rewards) if isinstance(rewards[0], list) else 1  # type: ignore
        self.reward_funcs = [self.reward_funcs[0]] * num_reward_funcs
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                # output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                if isinstance(rewards[i], list):  # type: ignore
                    output_reward_func = rewards[i]  # type: ignore
                else:
                    output_reward_func = rewards  # type: ignore
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        print("LOSS", loss)
        return loss
