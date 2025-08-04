import json
import random
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
from trl.extras.profiling import profiling_context
from trl.import_utils import is_vllm_available
from trl.trainer.utils import pad
from accelerate.utils import broadcast_object_list, gather_object, gather
from transformers import is_wandb_available, ProgressCallback
from transformers.utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
)
import trainers.utils.hf_grpo_clone as hf_grpo_clone
import heapq
import torch.nn.functional as F
from utils.evolution_database import EvolutionDatabase

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

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


def log_response(response, logfile):
    with open(logfile, "r") as file:
        data = json.load(file)
        data["guesses"].append(response)
    with open(logfile, "w") as file:
        json.dump(data, file, indent=4)


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
            self.training_bar.set_postfix(
                {
                    "loss": self.loss,
                    "train_reward_mean": self.train_acc,
                    "train_reward_max": self.train_acc_max,
                    "val_reward_majority": self.val_acc,
                }
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            self.training_bar.write("")


class TTT_GRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        target,
        strategy,
        logfile,
        task,
        dataset_file,
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
        sampling_strategy="none",
        migrate_gamma=10,
        migrate_beta=0,
        migrate_alpha=0,
        opro_sampling=False,
        use_barc_format=False,
        use_induction=False,
        warmstart_seed=42,
        loss_type="grpo",
        greedy_topk=1,
        evo_num_islands=4,
        evo_migration_interval=40,
        evo_migration_rate=0.05,
        evo_archive_size=5,
        evo_exploration_ratio=0.2,
        evo_exploitation_ratio=0.8,
        evo_exploration_topk=3,
        evo_exploitation_topk=3,
        include_scores=False,
        compute_entropy=False,
        semantle_warmstart_file=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.iteration = 0
        self.logfile = logfile
        self.target = target
        self.past_guesses = []
        self.seen_guesses = set()
        self.strategy = strategy
        self.sampling_strategy = sampling_strategy
        self.include_scores = include_scores
        self.migrate_gamma = migrate_gamma
        self.migrate_beta = migrate_beta
        self.migrate_alpha = migrate_alpha
        self.opro_sampling = opro_sampling
        self.task = task

        self.arc_sol = None
        self.arc_leave_out = None
        self.arc_past_guesses = {}
        self.dataset_file = dataset_file
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

        self.use_barc_format = use_barc_format
        self.use_induction = use_induction
        self.greedy_topk = greedy_topk
        self.loss_type = loss_type
        self.prompt_to_replace = []
        self.best_idx_replaced = None
        self.database = EvolutionDatabase(
            num_islands=evo_num_islands,
            migration_interval=evo_migration_interval,
            migration_rate=evo_migration_rate,
            archive_size=evo_archive_size,
            exploration_ratio=evo_exploration_ratio,
            exploitation_ratio=evo_exploitation_ratio,
            exploration_topk=evo_exploration_topk,
            exploitation_topk=evo_exploitation_topk,
        )
        self.greedy_insert = None
        self.compute_entropy = compute_entropy
        self.semantle_warmstart_file = semantle_warmstart_file

        if self.task.task_name == "semantle":
            with open(self.semantle_warmstart_file, "r") as file:
                warmstarts = json.load(file)[self.task.target][str(warmstart_seed)]
                for word in warmstarts:
                    heapq.heappush(self.past_guesses, (word[1], word[0]))
                    self.seen_guesses.add(word[0])
                    self.database.add(word)
                    self.database.next_island()

        for callback in self.callback_handler.callbacks:
            if type(callback) is CustomProgressCallback:
                self.progress_callback = callback

    # Record new completions and their black-box scores
    def update_past_guesses(self, responses, bb_scores):
        for res, score in zip(responses, bb_scores):
            if res not in self.seen_guesses:
                heapq.heappush(self.past_guesses, (score, res))
                self.seen_guesses.add(res)
        for completion_score in zip(responses, bb_scores):
            self.database.add(completion_score)
        self.database.next_island()
        if (self.iteration + 1) % self.database.migration_interval == 0:
            self.database.migrate()
        # self.arc_past_guesses[self.arc_leave_out] = self.past_guesses.copy()

    def _add_to_batch(self, replacement, completions, rewards):
        if replacement is not None:
            self.best_idx_replaced = len(completions)
            completions.extend([replacement[1] * self.migrate_beta])
            rewards.extend([replacement[0] * self.migrate_beta])
        else:
            self.best_idx_replaced = None

    def create_group_prompt_ids(self, inputs):

        # Create prompt for online samples
        # self.arc_leave_out = str(inputs[0]["solution"])
        # self.arc_prob = np.array(inputs[0]["problem"])
        # self.arc_sol = np.array(inputs[0]["solution"])
        # self.arc_prob = inputs[0]["problem"]
        # self.arc_sol = inputs[0]["solution"]

        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        if self.task.task_name == "semantle":
            # Special case for semantle since we generate a batch of solutions per completion
            self.prompt_to_replace = [{"prompt": self.task.prompts.build_prompt(use_alternate=True)}] * len(inputs)
        else:
            self.prompt_to_replace = inputs

        # n_online = len(prompts_text) - self.migrate_gamma
        n_online = self.migrate_alpha
        n_neighbors = self.migrate_gamma
        if self.task.task_name == "semantle":
            # Special case for semantle since we generate a batch of solutions per completion
            n_online = 0 if self.migrate_alpha == 0 else 1
            n_neighbors = 0 if self.migrate_gamma == 0 else 1
            prompts_text = prompts_text[:n_online]

        # Create prompt for neighborhood samples
        if (
            self.sampling_strategy != "none"
            and len(self.past_guesses) > 0
            and ("evo" not in self.sampling_strategy or not self.database.is_current_island_empty())
        ):
            prompts_text = prompts_text[:n_online]

            # Get selection for NS based on strategy
            if self.sampling_strategy == "random_topk":
                best_guesses = heapq.nlargest(self.greedy_topk, self.past_guesses)
                best_guess = [random.choice(best_guesses)]
            elif self.sampling_strategy == "opro":
                best_guess = heapq.nlargest(self.greedy_topk, self.past_guesses)
            else:
                best_guess = [[s.score, s.completion] for s in self.database.sample()]
            if self.task.task_name == "semantle":
                if self.include_scores:
                    solution = [f"Word: {x[1]}\nScore: {round(x[0], 4)}" for x in best_guess[::-1]]
                else:
                    solution = [f"Word: {x[1]}" for x in best_guess[::-1]]
            else:
                solution = [x[1] for x in best_guess[::-1]]
            self.greedy_insert = best_guess[-1]

            train_problems = [np.array(x) for x in inputs[0]["problem"]]
            train_solutions = [np.array(x) for x in inputs[0]["solution"]]
            problem = {"problems": train_problems, "solutions": train_solutions}

            if self.sampling_strategy == "opro":
                prompt_obj = self.task.prompts.get_opro_samples_prompt(problem, solution)
            else:
                prompt_obj = self.task.prompts.get_neighborhood_samples_prompt(problem, solution)
            ns_prompts = [maybe_apply_chat_template({"prompt": prompt_obj}, self.processing_class)["prompt"]]
            print(ns_prompts[0], n_neighbors)

            prompts_text += n_neighbors * ns_prompts
        print("total", len(prompts_text))

        # Tokenize prompts
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        return prompt_ids, prompt_mask, prompts_text

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompt_ids, prompt_mask, prompts_text = self.create_group_prompt_ids(inputs)

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
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
                            generation_kwargs=self.args.generation_kwargs,
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

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                }
                generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            # prompt_completion_ids = torch.cat([prompt_ids.to(device), completion_ids], dim=1)

            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            del completion_ids
            torch.cuda.empty_cache()
        else:
            # Regular generation path
            completions = []
            all_entropies = []
            for i in range(0, len(prompt_ids), self.inf_batch_size):
                _prompt_ids, _prompt_mask = (
                    prompt_ids[i : i + self.inf_batch_size],
                    prompt_mask[i : i + self.inf_batch_size],
                )
                with unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    retries = 0
                    completion = ""
                    while retries < 3:
                        completion_ids = unwrapped_model.generate(
                            _prompt_ids.to(self.args.device),
                            attention_mask=_prompt_mask.to(self.args.device),
                            max_new_tokens=self.max_completion_length,
                            do_sample=True,
                            pad_token_id=self.processing_class.pad_token_id,
                            bos_token_id=self.processing_class.bos_token_id,
                            eos_token_id=self.processing_class.eos_token_id,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=self.top_k,
                            min_p=self.min_p,
                        )

                        prompt_length = prompt_ids.size(1)
                        completion = self.processing_class.batch_decode(
                            completion_ids[:, prompt_length:], skip_special_tokens=True
                        )
                        if self.task.task_name != "semantle" or all(
                            len(self.task.decode_response(c)) > 0 for c in completion
                        ):
                            break
                        retries += 1
                    completions.extend(completion)

        responses = []
        bb_scores = []
        responses, bb_scores = self.task.evaluate_and_log(completions, inputs, self.iteration, self.logfile)
        print(responses)
        print(bb_scores)

        # Record guesses into history and log repsonses
        self.update_past_guesses(responses, bb_scores)
        completions = responses

        # Sort by rewards
        completions, bb_scores = zip(*sorted(zip(completions, bb_scores), key=lambda x: x[1], reverse=True))
        completions, bb_scores = list(completions), list(bb_scores)

        # Create completions and corresponding rewards based on the strategy
        rewards = bb_scores
        print("hmmm", self.migrate_beta, self.greedy_insert)
        if self.migrate_beta > 0:
            best_guess = None
            if self.greedy_insert:
                # Use greedy from NS
                self._add_to_batch(self.greedy_insert, completions, rewards)
                print("INSERTING", self.greedy_insert)
            elif len(completions) < len(inputs):
                # if not using NS
                best_guess = heapq.nlargest(self.greedy_topk, self.past_guesses)
                best_guess = random.choice(best_guess)
                self._add_to_batch(best_guess, completions, rewards)
                print("INSERTING", best_guess)
        elif self.strategy == "top_delta":
            raise NotImplementedError
        else:
            # Keep completions and rewards based on the online generated samples
            self.best_idx_replaced = None

        # Convert individual semantle words into batches
        if self.task.task_name == "semantle":
            completions, rewards = self.task.create_batch_completions(completions, rewards, self.past_guesses)
            print("COMPLETIONS", completions)
            print("REWARDS", rewards)

        # Compute mean and max rewards excluding the "greedy" replacement
        if self.best_idx_replaced is not None:
            mean_reward_minus_replacement = np.mean(rewards[: self.best_idx_replaced])
            max_reward_minus_replacement = np.max(rewards[: self.best_idx_replaced])
        else:
            mean_reward_minus_replacement = np.mean(rewards)
            max_reward_minus_replacement = np.max(rewards)

        # Update training bar
        self.progress_callback.train_acc = np.round(mean_reward_minus_replacement, 4)
        wandb.log({"train/reward_minus_replacement_mean": mean_reward_minus_replacement})
        self.progress_callback.train_acc_max = np.round(max_reward_minus_replacement, 4)
        wandb.log({"train/reward_minus_replacement_max": max_reward_minus_replacement})

        # Early stopping for Induction
        if self.use_induction and self.use_early_stopping and max_reward_minus_replacement == 1:
            self.control.should_training_stop = True
            print("EARLY STOPPING: Found a program that solved all the training examples.")

        prompt_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in self.prompt_to_replace
        ]
        prompt_inputs = self.processing_class(
            text=prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        completion_ids = self.processing_class(
            completions, return_tensors="pt", add_special_tokens=False, padding=True
        ).input_ids

        self.num_generations = len(completions)
        self.iteration += 1

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_ids.to(device)
        completion_ids = completion_ids.to(device)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # zero out completions with zero rewards
        incorrect_completions = torch.tensor(rewards).to(device) == 0
        completion_mask = completion_mask * (~incorrect_completions).unsqueeze(1).int()

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_mask = prompt_mask.to(device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep
                        )
            else:
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    # Convert None values to NaN
                    # output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    output_reward_func = rewards

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

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
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        return hf_grpo_clone._get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep)

    def _get_completion_logps(self, model, inputs, gold=False):
        if not gold:
            completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        else:
            completion_ids, completion_mask = inputs["gold_completion_ids"], inputs["gold_completion_mask"]
            prompt_ids, prompt_mask = (
                inputs["prompt_ids"][: len(completion_ids)],
                inputs["prompt_mask"][: len(completion_mask)],
            )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        completion_logps = (per_token_logps * attention_mask[:, -logits_to_keep:]).sum(dim=1) / attention_mask[
            :, -logits_to_keep:
        ].sum(dim=1)
        return completion_logps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = None

        if self.nll_weight > 0:
            # Use the gold completion logp for NLL
            completion_logps = self._get_completion_logps(model, inputs, gold=True)[0]
            nll_loss = self.nll_weight * -completion_logps
            if loss is None or loss == 0:
                loss = nll_loss
            else:
                loss += nll_loss
            self._metrics["train"]["nll_loss"].append(nll_loss.item())

        if self.pro_loss_weight > 0:
            completion_logps = self._get_completion_logps(model, inputs)
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

            _pro_loss = self.pro_loss_weight * pro_loss(sorted_completion_logps, start_neg_idx_to_ignore=ignore_idx)
            if loss is None or loss == 0:
                loss = _pro_loss
            else:
                loss += _pro_loss
            self._metrics["train"]["pro_loss"].append(_pro_loss.item())

        if self.grpo_weight > 0:
            grpo_loss = self.grpo_weight * hf_grpo_clone.compute_loss(
                self, model, inputs, return_outputs, num_items_in_batch
            )
            if loss is None or loss == 0:
                loss = grpo_loss
            else:
                loss += grpo_loss
            self._metrics["train"]["grpo_loss"].append(grpo_loss.item())

        self.progress_callback.loss = np.round(loss.item(), 4)

        assert loss is not None, "At least one of the losses should be computed"
        return loss

    def _get_per_token_logps_and_entropies(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None
    ) -> torch.Tensor:
        """Compute log‐probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                logits_to_keep=logits_to_keep + 1,
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            entropies = entropy_from_logits(logits)
            all_entropies.append(entropies)

        entropies = torch.cat(all_entropies, dim=0)
        return entropies


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
    inf_mask = torch.full(mask.shape, float("-inf"), device=p.device)
    inf_mask[mask] = 0  # Only allow valid indices
    denominators = torch.logsumexp(p_expanded + inf_mask, dim=2)  # Shape: (batch_size, n-1)
    numerators = p[:, :-1]  # Shape: (batch_size, n-1)
    loss = -(numerators - denominators)[:, :start_neg_idx_to_ignore].sum(dim=1).mean()

    return loss


def entropy_from_logits(logits, chunk_size: int = 1) -> torch.Tensor:
    """
    Compute the Shannon entropy (in nats) for each row of *logits* without
    materialising the full soft-max in memory.
    The batch dimension is processed in chunks of size `chunk_size` so that
    only a subset of rows is expanded to probabilities at any one time.
    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`. Entropy is taken along the last axis; all
            leading dimensions are preserved.
        chunk_size (`int`, *optional*, defaults to `1`):
            Number of rows to process per iteration.
    Returns:
        `torch.Tensor`:
            Entropy values with shape `logits.shape[:-1]`.
    """
    per_token_entropies = []
    for logits_chunk in logits.split(chunk_size, dim=0):
        logps = F.log_softmax(logits_chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        per_token_entropies.extend(chunk_entropy)

    per_token_entropies = torch.stack(per_token_entropies)
    return per_token_entropies


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)
