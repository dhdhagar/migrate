import json
import random
import itertools
import numpy as np
import torch
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from trl import (
    GRPOTrainer,
    maybe_apply_chat_template,
    is_conversational,
    apply_chat_template,
)
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import pad
from accelerate.utils.other import is_compiled_module
from accelerate.utils import broadcast_object_list, gather, gather_object
from transformers import is_wandb_available, PreTrainedModel, AutoModel, AutoTokenizer
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
from torch.nn.functional import cosine_similarity


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


def logResponse(response, logfile):
    with open(logfile, "r") as file:
        data = json.load(file)
        data["Guesses"].append(response)
    with open(logfile, "w") as file:
        json.dump(data, file, indent=4)


class SemantleGRPOTrainer(GRPOTrainer):
    def __init__(self, target, num_guesses, strategy, n_reps, logfile, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration = 0
        self.logfile = logfile
        self.target = target
        embedder = "princeton-nlp/sup-simcse-roberta-large"
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_sim = AutoModel.from_pretrained(embedder).to(DEVICE)
        self.tokenizer_sim = AutoTokenizer.from_pretrained(embedder)
        self.past_guesses = {}
        self.strategy = strategy
        self.num_guesses = num_guesses
        self.n_reps = n_reps

    def get_sim(self, word1, word2):
        texts = [f"What is a {word1}?", f"What is a {word2}?"]
        inputs = self.tokenizer_sim(texts, padding=True, truncation=True, return_tensors="pt").to(self.args.device)
        with torch.no_grad():
            embeddings = self.model_sim(**inputs, output_hidden_states=True, return_dict=True).pooler_output

        return cosine_similarity(embeddings[0], embeddings[1], dim=0).item()

    def update_past_guesses(self, responses, bb_scores):
        guesses = list(itertools.chain.from_iterable(responses))
        scores = list(itertools.chain.from_iterable(bb_scores))
        for word, score in zip(guesses, scores):
            self.past_guesses[word] = score

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        # Skip gradient step if loss is None
        if loss is not None:
            kwargs = {}
            # For LOMO optimizers you need to explicitly use the learnign rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss, **kwargs)
                # Finally we need to normalize the loss for reporting
                if num_items_in_batch is None:
                    return loss.detach() / self.args.gradient_accumulation_steps
                return loss.detach()
        else:
            return torch.tensor(0.0, dtype=torch.float32, device=self.args.device).detach()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        valid_completion = False
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
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
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
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                retries = 0
                while not valid_completion and retries < 5:
                    try:
                        completion_ids = unwrapped_model.generate(
                            input_ids=prompt_inputs["input_ids"].repeat(self.n_reps, 1),
                            attention_mask=prompt_inputs["attention_mask"].repeat(self.n_reps, 1),
                            generation_config=self.generation_config,
                        )
                        prompt_length = prompt_inputs["input_ids"].size(1)
                        responses = []
                        bb_scores = []
                        for completion in completion_ids:
                            res = self.processing_class.decode(completion[prompt_length:], skip_special_tokens=True)
                            guesses = json.loads(res)["response"][: self.num_guesses]
                            responses.append(guesses)
                            bb_scores.append([self.get_sim(guess, self.target) for guess in guesses])

                        # Create completions and corresponding rewards
                        # TODO: Refactor
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

                        # Update and log new guesses
                        self.update_past_guesses(responses, bb_scores)
                        logResponse(
                            {
                                f"Iteration: {self.iteration}": [
                                    (x, y)
                                    for x, y in zip(
                                        list(itertools.chain.from_iterable(responses)),
                                        list(itertools.chain.from_iterable(bb_scores)),
                                    )
                                ]
                            },
                            self.logfile,
                        )
                        # Change prompt to use the same number of words in each completion
                        n = 1 if isinstance(completions[0], list) else len(completions[0])
                        prompt = {
                            "prompt": [
                                {
                                    "content": 'You are a helpful chatbot with high attention to detail who is not talkative and responds only \
    with the answer and no additional conversation. All your responses should be in JSON format, i.e. {key: value}, where the \
    key is always "response" and the value can be a string, int, list, or dict, depending on the context.',
                                    "role": "system",
                                },
                                {
                                    "content": f'Your task is to guess a hidden word from the English dictionary. Stick to proper, single-word \
    English words. Now, guess exactly n={n} new word(s) that could be the hidden word. Be creative! (Note: give only \
    a list of word(s) in the provided JSON format, e.g. {{"response": ["word1", "word2",...]}})',
                                    "role": "user",
                                },
                            ]
                        }
                        prompt_text = [maybe_apply_chat_template(prompt, self.processing_class)["prompt"]]
                        prompt_inputs = self.processing_class(
                            prompt_text,
                            return_tensors="pt",
                            padding=True,
                            padding_side="left",
                            add_special_tokens=False,
                        )
                        completion_ids = []
                        for i in range(len(completions)):
                            ids = self.processing_class(
                                json.dumps({"response": completions[i]}, indent=4),
                                return_tensors="pt",
                            ).input_ids.view(-1)
                            completion_ids.append(ids)
                        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
                        prompt_inputs_repeated = torch.repeat_interleave(
                            prompt_inputs["input_ids"], len(completion_ids), dim=0
                        )
                        prompt_completion_ids = torch.cat([prompt_inputs_repeated, completion_ids], dim=1)
                        self.num_generations = len(completion_ids)
                        valid_completion = True
                        self.iteration += 1
                    except Exception as e:
                        retries += 1
                        # print(e)

        if valid_completion:
            prompt_length = prompt_inputs["input_ids"].size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Get the per-token log probabilities for the completions for the model and the reference model
            def get_per_token_logps(model, input_ids, num_logits_to_keep):
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

            num_logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
            per_token_logps = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep).to(device)

            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps = get_per_token_logps(
                        self.ref_model, prompt_completion_ids, num_logits_to_keep
                    ).to(device)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep).to(
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
                    output_reward_func = rewards
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
            return loss
        else:
            logResponse({f"Iteration: {self.iteration}": []}, self.logfile)
            self.iteration += 1
            return None
