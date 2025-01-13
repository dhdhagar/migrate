import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import jinja2
from trl import (
    OnlineDPOTrainer,
    maybe_apply_chat_template,
    is_conversational,
    apply_chat_template,
)

from trl.trainer.utils import (
    SIMPLE_CHAT_TEMPLATE,
    truncate_right,
    get_reward,
    empty_cache,
)
from trl.models.utils import unwrap_model_for_generation
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import is_apex_available, is_wandb_available
from transformers.training_args import OptimizerNames
from transformers.utils import is_peft_available, is_sagemaker_mp_enabled, logging
import itertools
import random
from packaging import version

if is_peft_available():
    from peft import PeftModel, get_peft_model

if is_apex_available():
    from apex import amp


if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_wandb_available():
    import wandb

logger = logging.get_logger(__name__)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Log LLM repsonse
def logResponse(response, logfile):
    try:
        with open(logfile, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(response)
    with open(logfile, "w") as file:
        json.dump(data, file, indent=4)


class SemantleOnlineDPOTrainer(OnlineDPOTrainer):

    def __init__(
        self, tokenizer, target, num_guesses, strategy, logfile, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ref_tokenizer = tokenizer
        self.target = target
        self.num_guesses = num_guesses
        self.logfile = logfile
        self.best_guesses = []
        self.strategy = strategy

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        model.train()

        # Apply chat template and tokenize the input.
        # We do this on-the-fly to enable the use of reward models and policies with different tokenizers / chat templates.
        # batch_size = len(next(iter(inputs.values())))
        batch_size = 1
        copy_inputs = inputs.copy()
        prompts = inputs["prompt"]
        inputs = [{k: v[i] for k, v in inputs.items()} for i in range(batch_size)]
        inputs = [maybe_apply_chat_template(x, self.processing_class) for x in inputs]
        inputs = [
            self.tokenize_row(
                x, self.model.config.is_encoder_decoder, self.processing_class
            )
            for x in inputs
        ]
        inputs = self.data_collator(inputs)

        # Sample 2 completions per prompt of size `max_new_tokens` from the model
        inputs = self._prepare_inputs(inputs)
        num_examples, context_length = inputs["prompt_input_ids"].shape
        prompt_ids = inputs["prompt_input_ids"].repeat(2, 1)
        prompt_mask = inputs["prompt_attention_mask"].repeat(2, 1)
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            output = unwrapped_model.generate(
                input_ids=prompt_ids[0].unsqueeze(0),
                attention_mask=prompt_mask[0].unsqueeze(0),
                generation_config=self.generation_config,
            )
            output = output.repeat(2, 1)

        # Re-init with "n=1" prompt
        for i in range(len(copy_inputs["prompt"])):
            # Change user prompt to specify "n=1", system prompt remains the same
            copy_inputs["prompt"][i][1][
                "content"
            ] = 'Your task is to guess a hidden word from the English dictionary. Stick to proper, single-word English words. Now, guess exactly n=1 new word(s) that could be the hidden word. Be creative! (Note: give only a list of word(s) in the provided JSON format, e.g. {"response": ["word1", "word2",...]})'
        prompts = copy_inputs["prompt"]
        inputs = [{k: v[i] for k, v in copy_inputs.items()} for i in range(batch_size)]
        inputs = [maybe_apply_chat_template(x, self.processing_class) for x in inputs]
        inputs = [
            self.tokenize_row(
                x, self.model.config.is_encoder_decoder, self.processing_class
            )
            for x in inputs
        ]
        inputs = self.data_collator(inputs)
        inputs = self._prepare_inputs(inputs)
        num_examples, context_length = inputs["prompt_input_ids"].shape
        prompt_ids = inputs["prompt_input_ids"].repeat(2, 1)
        prompt_mask = inputs["prompt_attention_mask"].repeat(2, 1)

        response = self.ref_tokenizer.decode(
            output[0, context_length:], skip_special_tokens=True
        )
        output_prompt = output[0, :context_length]

        logResponse({"guess": response}, self.logfile)  # Log response

        # bool for skipping gradient update if llm response is invalid
        skip_update = False
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        try:
            res = json.loads(response)
            responses = res["response"][: self.num_guesses]
            random.shuffle(responses)  # shuffle for greedy
            for i in range(self.num_guesses):
                # Randomly choose pairs on the first iteration or if non-greedy
                if self.strategy == "random" or (
                    self.strategy == "greedy"
                    and len(self.best_guesses) < self.num_guesses
                ):
                    # Create response pairs to judge
                    pairs = list(itertools.combinations(responses, 2))
                    random.shuffle(pairs)
                    pi_guess, ref_guess = pairs[i]
                # If greedy, compare each generated word with previous top words
                elif self.strategy == "greedy":
                    pi_guess, ref_guess = responses[i], self.best_guesses[i]["word"]
                # If oracle, compare each generated word with the target word
                elif self.strategy == "oracle":
                    pi_guess, ref_guess = responses[i], self.target

                pi_tensor = self.ref_tokenizer(
                    json.dumps({"response": [pi_guess]}, indent=4),
                    return_tensors="pt",
                ).input_ids.view(-1)
                ref_tensor = self.ref_tokenizer(
                    json.dumps({"response": [ref_guess]}, indent=4),
                    return_tensors="pt",
                ).input_ids.view(-1)

                outputs = [
                    torch.cat((output_prompt, pi_tensor)),
                    torch.cat((output_prompt, ref_tensor)),
                ]
                max_length = max(len(outputs[0]), len(outputs[1]))

                # Create prompt + guess token tensors
                output = torch.full((2, max_length), self.processing_class.pad_token_id)
                output[0, : len(outputs[0])] = outputs[0]
                output[1, : len(outputs[1])] = outputs[1]
                prompt_ids = prompt_ids[0].repeat((output.shape[0], 1))
                prompt_mask = prompt_mask[0].repeat((output.shape[0], 1))
                num_examples = 1
                completion_ids = output[:, context_length:]
                completion_ids, completion_mask = truncate_right(
                    completion_ids,
                    self.processing_class.eos_token_id,
                    self.processing_class.pad_token_id,
                )
                contain_eos_token = torch.any(
                    completion_ids == self.processing_class.eos_token_id, dim=-1
                )
                prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
                prompt_completion_mask = torch.cat(
                    (prompt_mask, completion_mask), dim=1
                )

                # Get the logprobs of the completions from the model
                output = model(
                    prompt_completion_ids, attention_mask=prompt_completion_mask
                )
                # There is 1 offset, because the model predict the next token
                logits = output.logits[:, context_length - 1 : -1]
                # Turn logits into logprobs
                all_logprobs = F.log_softmax(logits, dim=-1)
                # Take the completion tokens logprob
                logprobs = torch.take_along_dim(
                    all_logprobs, completion_ids.unsqueeze(-1), dim=2
                ).squeeze(-1)
                del output, logits, all_logprobs  # free memory

                # Same for the reference model
                with torch.no_grad():
                    if self.ref_model is not None:
                        ref_output = self.ref_model(
                            prompt_completion_ids, attention_mask=prompt_completion_mask
                        )
                    else:  # peft case: we just need to disable the adapter
                        with self.model.disable_adapter():
                            ref_output = self.model(
                                prompt_completion_ids,
                                attention_mask=prompt_completion_mask,
                            )
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_all_logprobs = F.log_softmax(ref_logits, dim=-1)
                    ref_logprobs = torch.take_along_dim(
                        ref_all_logprobs, completion_ids.unsqueeze(-1), dim=2
                    ).squeeze(-1)
                    del ref_output, ref_logits, ref_all_logprobs  # free memory

                # Decode the completions, and format them if the input is conversational
                device = prompt_completion_ids.device
                completions_ids = prompt_completion_ids[:, context_length:]
                completions = self.processing_class.batch_decode(
                    completions_ids, skip_special_tokens=True
                )
                if is_conversational({"prompt": prompts[0]}):
                    completions = [
                        [{"role": "assistant", "content": completion}]
                        for completion in completions
                    ]

                # Get the reward from the reward model or judge
                if self.judge is not None:
                    # Once formatted, conversational data may contain special tokens (such as <|im_start|>) that are not
                    # directly understandable by the judge and could alter its judgment. To avoid this and make the judge
                    # independent of the model's chat template, we use the raw conversation data, and apply our own chat
                    # template to it.
                    if is_conversational({"prompt": prompts[0]}):
                        environment = jinja2.Environment()
                        template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
                        prompts = [
                            template.render(messages=prompt) for prompt in prompts
                        ]
                        # completions = [
                        #     template.render(messages=completion) for completion in completions
                        # ]
                        # print("COMPLETIONS", completions)
                        completions = [
                            completion[0]["content"].strip()
                            for completion in completions
                        ]

                    ranks_of_first_completion, best_sims = self.judge.judge(
                        prompts,
                        list(
                            zip(completions[:num_examples], completions[num_examples:])
                        ),
                    )

                    # Update Best guesses
                    try:
                        best_sims = self.best_guesses + best_sims
                        unique_guesses = {}
                        for item in best_sims:
                            unique_guesses[item["word"]] = item
                        self.best_guesses = sorted(
                            unique_guesses.values(), key=lambda x: x["sim"]
                        )[-self.num_guesses :]
                    except:
                        pass

                    # convert ranks to a True/False mask:
                    # when rank == 0, it means the first completion is the best
                    # when rank == 1, it means the second completion is the best
                    mask = torch.tensor(
                        [rank == 0 for rank in ranks_of_first_completion], device=device
                    )
                else:
                    # The reward model may not have the same chat template or tokenizer as the model, so we need to use the
                    # raw data (string), apply the chat template (if needed), and tokenize it with the reward processing class.
                    prompts = (
                        2 * prompts
                    )  # repeat the prompt: [prompt0, prompt1] -> [prompt0, prompt1, prompt0, prompt1]
                    if is_conversational({"prompt": prompts[0]}):
                        examples = [
                            {"prompt": p, "completion": c}
                            for p, c in zip(prompts, completions)
                        ]
                        examples = [
                            apply_chat_template(example, self.reward_processing_class)
                            for example in examples
                        ]
                        prompts = [example["prompt"] for example in examples]
                        completions = [example["completion"] for example in examples]

                    # Tokenize the prompts
                    prompts_ids = self.reward_processing_class(
                        prompts, padding=True, return_tensors="pt", padding_side="left"
                    )["input_ids"].to(device)
                    context_length = prompts_ids.shape[1]

                    # Tokenize the completions
                    completions_ids = self.reward_processing_class(
                        completions,
                        padding=True,
                        return_tensors="pt",
                        padding_side="right",
                    )["input_ids"].to(device)

                    # Concatenate the prompts and completions and get the reward
                    prompt_completion_ids = torch.cat(
                        (prompts_ids, completions_ids), dim=1
                    )
                    with torch.inference_mode():
                        _, scores, _ = get_reward(
                            self.reward_model,
                            prompt_completion_ids,
                            self.reward_processing_class.pad_token_id,
                            context_length,
                        )

                        # Filter completion. Ensure that the sample contains stop_token_id
                        # Completions not passing that filter will receive a lower score.
                        if self.args.missing_eos_penalty is not None:
                            scores[~contain_eos_token] -= self.args.missing_eos_penalty

                    # Split the scores in 2 (the prompts of the first half are the same as the second half)
                    first_half, second_half = scores.split(num_examples)

                    # Get the indices of the chosen and rejected examples
                    mask = first_half >= second_half

                num_examples_range = torch.arange(num_examples, device=device)
                chosen_indices = num_examples_range + (~mask * num_examples)
                rejected_indices = num_examples_range + (mask * num_examples)

                # Build tensor so that the first half is the chosen examples and the second half the rejected examples
                cr_indices = torch.cat(
                    (chosen_indices, rejected_indices), dim=0
                )  # cr = chosen and rejected
                cr_logprobs = logprobs[cr_indices]
                cr_ref_logprobs = ref_logprobs[cr_indices]

                # mask out the padding tokens
                padding_mask = ~completion_mask.bool()
                cr_padding_mask = padding_mask[cr_indices]

                cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
                cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)

                # Split the chosen and rejected examples
                chosen_logprobs_sum, rejected_logprobs_sum = torch.split(
                    cr_logprobs_sum, num_examples
                )
                chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(
                    cr_ref_logprobs_sum, num_examples
                )
                pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
                ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

                logits = pi_logratios - ref_logratios

                if self.args.loss_type == "sigmoid":
                    losses = -F.logsigmoid(self.beta * logits)
                elif self.args.loss_type == "ipo":
                    losses = (logits - 1 / (2 * self.beta)) ** 2
                else:
                    raise NotImplementedError(f"invalid loss type {self.loss_type}")

                loss = losses.mean()

                # Log everything
                if self.reward_model is not None:
                    scores_margin = scores[chosen_indices] - scores[rejected_indices]
                    self.stats["objective/scores_margin"].append(
                        self.accelerator.gather(scores_margin.mean()).mean().item()
                    )
                    self.stats["objective/scores"].append(
                        self.accelerator.gather(scores.mean()).mean().item()
                    )
                self.stats["val/contain_eos_token"].append(
                    contain_eos_token.float().mean().item()
                )
                self.stats["logps/chosen"].append(
                    self.accelerator.gather(chosen_logprobs_sum).mean().item()
                )
                self.stats["logps/rejected"].append(
                    self.accelerator.gather(rejected_logprobs_sum).mean().item()
                )

                kl = logprobs - ref_logprobs
                mean_kl = kl.sum(1).mean()
                self.stats["objective/kl"].append(
                    self.accelerator.gather(mean_kl).mean().item()
                )
                non_score_reward = (-self.beta * kl).sum(1)
                mean_non_score_reward = non_score_reward.mean()
                self.stats["objective/non_score_reward"].append(
                    self.accelerator.gather(mean_non_score_reward).mean().item()
                )
                if self.reward_model is not None:
                    rlhf_reward = scores + non_score_reward
                    self.stats["objective/rlhf_reward"].append(
                        self.accelerator.gather(rlhf_reward).mean().item()
                    )
                mean_entropy = -logprobs.sum(1).mean()
                self.stats["objective/entropy"].append(
                    self.accelerator.gather(mean_entropy).mean().item()
                )
                chosen_rewards = self.beta * (
                    chosen_logprobs_sum - chosen_ref_logprobs_sum
                )
                gathered_chosen_rewards = self.accelerator.gather(chosen_rewards)
                self.stats["rewards/chosen"].append(
                    gathered_chosen_rewards.mean().item()
                )
                rejected_rewards = self.beta * (
                    rejected_logprobs_sum - rejected_ref_logprobs_sum
                )
                gathered_rejected_rewards = self.accelerator.gather(rejected_rewards)
                self.stats["rewards/rejected"].append(
                    gathered_rejected_rewards.mean().item()
                )
                margin = gathered_chosen_rewards - gathered_rejected_rewards
                self.stats["rewards/margins"].append(margin.mean().item())
                accuracy = margin > 0
                self.stats["rewards/accuracies"].append(accuracy.float().mean().item())
                self.stats["beta"].append(self.beta)

                if (
                    self.args.torch_empty_cache_steps is not None
                    and self.state.global_step % self.args.torch_empty_cache_steps == 0
                ):
                    empty_cache()

                loss = loss / num_prefs
                kwargs = {}

                # For LOMO optimizers you need to explicitly use the learnign rate
                if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                    kwargs["learning_rate"] = self._get_learning_rate()

                if self.args.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training

                if self.use_apex:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss, **kwargs)

                total_loss += loss
        except:
            pass
        del inputs

        # return loss.detach() / self.args.gradient_accumulation_steps
        return total_loss.detach() / self.args.gradient_accumulation_steps

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time=None,
    ):
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.detach().item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
            logs["learning_rate"] = self._get_learning_rate()

            # Add our metrics
            for key, val in self.stats.items():
                logs[key] = sum(val) / max(len(val), 1)
            self.stats = {key: [] for key in self.stats}  # reset stats

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
                self.log(logs, start_time)
            else:  # transformers<=4.46
                self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(
                metrics=metrics, trial=trial
            )

            if self.args.save_strategy == "best":
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )
