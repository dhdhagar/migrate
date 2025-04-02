import json
import numpy as np
from typing import Any, Dict, Optional, Union
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
from transformers import is_apex_available, is_wandb_available
from transformers.training_args import OptimizerNames
from transformers.utils import is_peft_available, is_sagemaker_mp_enabled, logging
import itertools
from packaging import version
import prompts as prompts_getter
import arc_utils.utils as arc_utils
from .utils.DPO_utils import create_pairs

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


# Log LLM repsonse
def logResponse(response, logfile):
    with open(logfile, "r") as file:
        data = json.load(file)
        data["Completions"].append(response)
    with open(logfile, "w") as file:
        json.dump(data, file, indent=4)


def logRelatedCompletions(response, logfile):
    with open(logfile, "r") as file:
        data = json.load(file)
        data["Related"].append(response)
    with open(logfile, "w") as file:
        json.dump(data, file, indent=4)


partial_oracles = {
    "airbase": [["airplay", 0.59577], ["airplane", 0.7133261], ["airfield", 0.8343316]],
    "birthstone": [["topaz", 0.57368684], ["stonefaceless", 0.6203526], ["jewelstone", 0.7818818]],
    "cement": [["ceramics", 0.6023039], ["cinder", 0.7185499], ["concrete", 0.8148617]],
    "computer": [["parts", 0.6009335], ["machine", 0.717893], ["laptop", 0.8281281]],
    "filament": [["bedding", 0.5998772], ["webbing", 0.685358], ["filaments", 0.8125428]],
    "machetes": [["pickax", 0.6015184], ["cleaver", 0.7179322], ["machete", 0.8730302]],
    "meatloaf": [["roaster", 0.6004665], ["tenderloin", 0.7293914], ["loafmeat", 0.8380054]],
    "mob": [["group", 0.6105943], ["masses", 0.7245998], ["mobs", 0.8967455]],
    "polyethylene": [["stuff", 0.5985743], ["acetylene", 0.7112042], ["polyester", 0.8660064]],
    "skillet": [["dishcloth", 0.601084], ["kitchenware", 0.7172206], ["pan", 0.8994801]],
}


class TTT_DPOTrainer(OnlineDPOTrainer):

    def __init__(
        self,
        target,
        strategy,
        logfile,
        sample_related,
        task,
        validation_example,
        validation_interval,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target = target
        self.logfile = logfile
        self.strategy = strategy
        self.past_completions = {}
        self.iteration = 0
        self.sample_related = sample_related
        self.task = task
        self.strategy = "Greedy_Gold"

        self.validation_example = validation_example
        self.validation_interval = validation_interval

    def get_bb_score(self, completion1, completion2, verbose=True):
        score = 1 - arc_utils.hamming_distance(completion1, completion2)
        if verbose:
            print("ATTEMPT:\n", completion2)
            print("SOLUTION:\n", completion1)
            print("SCORE:", score)
        return score

    def sample_related_completions(self, model, chosen_completion, n):
        inputs = None
        if self.task == "semantle":
            inputs = {"prompt": prompts_getter.get_semantle_related_prompt(n, chosen_completion)}
        inputs = maybe_apply_chat_template(inputs, self.processing_class)
        inputs = [self.tokenize_row(inputs, self.model.config.is_encoder_decoder, self.processing_class)]
        inputs = self.data_collator(inputs)
        inputs = self._prepare_inputs(inputs)
        prompt_ids = inputs["prompt_input_ids"]
        prompt_mask = inputs["prompt_attention_mask"]
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            completion_ids = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
            )
        related_completions = self.ref_tokenizer.decode(
            completion_ids[0, prompt_ids.size(1) :], skip_special_tokens=True
        )
        try:
            related_completions = json.loads(related_completions)["response"]
            return related_completions
        except Exception as _:
            return None

    def run_validation(self, model):
        prompts = self.validation_example["dataset"]

        inputs = [{"prompt": prompt} for prompt in prompts]
        inputs = [maybe_apply_chat_template(x, self.processing_class) for x in inputs]
        inputs = [self.tokenize_row(x, model.config.is_encoder_decoder, self.processing_class) for x in inputs]
        inputs = self.data_collator(inputs)

        inputs = self._prepare_inputs(inputs)
        prompt_ids = inputs["prompt_input_ids"]
        prompt_mask = inputs["prompt_attention_mask"]
        with unwrap_model_for_generation(
            model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            output = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                do_sample=False,
                max_new_tokens=self.args.max_new_tokens,
            )

        completion_ids = output[:, prompt_ids.size(1) :]
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        del completion_ids

        # Aggregate results
        results = {}
        for completion in completions:
            parsed_completion = arc_utils.parse_response(completion)
            parsed_completion = completion if parsed_completion.size == 0 else parsed_completion
            # Get black-box score if completion is valid otherwise 0
            score = (
                self.judge.get_bb_score(self.validation_example["solution"], parsed_completion)
                if isinstance(parsed_completion, np.ndarray)
                else 0
            )
            parsed_completion = str(parsed_completion)
            # Track completions and their scores
            results[parsed_completion] = results.get(parsed_completion, {"score": score, "count": 0})
            results[parsed_completion]["count"] += 1

        # Do pass@2 majority voting -- also make sure majority is more than 1
        sorted_majority = sorted(results.items(), key=lambda x: x[1]["count"], reverse=True)
        if any(x[1]["score"] == 1.0 for x in sorted_majority[:2]) and sorted_majority[0][1]["count"] > 1:
            self.control.should_training_stop = True

        print("VALIDATION SOLUTION", self.validation_example["solution"])
        print("VALIDATION ATTEMPT", sorted_majority[0][0])
        print("VALIDATION SCORE", sorted_majority[0][1]["score"])

        wandb.log({"validation/majority_reward_max": max([x[1]["score"] for x in sorted_majority[:2]])})
        wandb.log({"validation/majority_reward_mean": np.mean([x[1]["score"] for x in sorted_majority[:2]])})
        self.judge.log_response(
            [{"completion": x[0], "score": x[1]["score"], "count": x[1]["count"]} for x in results.items()],
            validation=True,
        )

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        if self.iteration % self.validation_interval == 0:
            self.run_validation(model)
        self.iteration += 1
        model.train()

        prompts = inputs["prompt"]
        batch_size = len(prompts)

        if self.args.use_vllm:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_vllm(model, prompts)
        else:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate(model, prompts)

        solutions = inputs["solution"]
        if self.strategy != "Vanilla":
            completion_ids, completion_mask, prompt_ids, prompt_mask, solutions = create_pairs(
                self, completion_ids, prompt_ids, prompt_mask, inputs["solution"]
            )
            batch_size = len(completion_ids) // 2

        contain_eos_token = torch.any(completion_ids == self.processing_class.eos_token_id, dim=-1)

        logprobs = self._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        with torch.no_grad():
            if self.ref_model is not None:
                ref_logprobs = self._forward(self.ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask)
            else:  # peft case: we just need to disable the adapter
                with self.model.disable_adapter():
                    ref_logprobs = self._forward(self.model, prompt_ids, prompt_mask, completion_ids, completion_mask)

        # Decode the completions, and format them if the input is conversational
        device = logprobs.device
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational({"prompt": prompts[0]}):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Get the reward from the reward model or judge
        if self.judge is not None:
            # Once formatted, conversational data may contain special tokens (such as <|im_start|>) that are not
            # directly understandable by the judge and could alter its judgment. To avoid this and make the judge
            # independent of the model's chat template, we use the raw conversation data, and apply our own chat
            # template to it.
            if is_conversational({"prompt": prompts[0]}):
                environment = jinja2.Environment()
                template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
                prompts = [template.render(messages=prompt) for prompt in prompts]
                completions = [template.render(messages=completion) for completion in completions]

            ranks_of_first_completion = self.judge.judge(
                prompts, list(zip(completions[:batch_size], completions[batch_size:])), solutions=solutions
            )

            # convert ranks to a True/False mask:
            # when rank == 0, it means the first completion is the best
            # when rank == 1, it means the second completion is the best
            mask = torch.tensor([rank == 0 for rank in ranks_of_first_completion], device=device)
        else:
            # The reward model may not have the same chat template or tokenizer as the model, so we need to use the
            # raw data (string), apply the chat template (if needed), and tokenize it with the reward processing class.
            prompts = 2 * prompts  # repeat the prompt: [prompt0, prompt1] -> [prompt0, prompt1, prompt0, prompt1]
            if is_conversational({"prompt": prompts[0]}):
                examples = [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]
                examples = [apply_chat_template(example, self.reward_processing_class) for example in examples]
                prompts = [example["prompt"] for example in examples]
                completions = [example["completion"] for example in examples]

            # Tokenize the prompts
            prompts_ids = self.reward_processing_class(prompts, padding=True, return_tensors="pt", padding_side="left")[
                "input_ids"
            ].to(device)
            context_length = prompts_ids.shape[1]

            # Tokenize the completions
            completions_ids = self.reward_processing_class(
                completions, padding=True, return_tensors="pt", padding_side="right"
            )["input_ids"].to(device)

            # Concatenate the prompts and completions and get the reward
            prompt_completion_ids = torch.cat((prompts_ids, completions_ids), dim=1)
            with torch.inference_mode():
                _, scores, _ = get_reward(
                    self.reward_model, prompt_completion_ids, self.reward_processing_class.pad_token_id, context_length
                )

                # Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty

            # Split the scores in 2 (the prompts of the first half are the same as the second half)
            first_half, second_half = scores.split(batch_size)

            # Get the indices of the chosen and rejected examples
            mask = first_half >= second_half

        batch_range = torch.arange(batch_size, device=device)
        chosen_indices = batch_range + (~mask * batch_size)
        rejected_indices = batch_range + (mask * batch_size)

        # Build tensor so that the first half is the chosen examples and the second half the rejected examples
        cr_indices = torch.cat((chosen_indices, rejected_indices), dim=0)  # cr = chosen and rejected
        cr_logprobs = logprobs[cr_indices]
        cr_ref_logprobs = ref_logprobs[cr_indices]

        # mask out the padding tokens
        padding_mask = ~completion_mask.bool()
        cr_padding_mask = padding_mask[cr_indices]

        cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
        cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)

        # Split the chosen and rejected examples
        chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, batch_size)
        chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)
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
                self.accelerator.gather_for_metrics(scores_margin.mean()).mean().item()
            )
            self.stats["objective/scores"].append(self.accelerator.gather_for_metrics(scores.mean()).mean().item())
        self.stats["val/contain_eos_token"].append(contain_eos_token.float().mean().item())
        self.stats["logps/chosen"].append(self.accelerator.gather_for_metrics(chosen_logprobs_sum).mean().item())
        self.stats["logps/rejected"].append(self.accelerator.gather_for_metrics(rejected_logprobs_sum).mean().item())

        kl = logprobs - ref_logprobs
        mean_kl = kl.sum(1).mean()
        self.stats["objective/kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        non_score_reward = (-self.beta * kl).sum(1)
        mean_non_score_reward = non_score_reward.mean()
        self.stats["objective/non_score_reward"].append(
            self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
        )
        if self.reward_model is not None:
            rlhf_reward = scores + non_score_reward
            self.stats["objective/rlhf_reward"].append(self.accelerator.gather_for_metrics(rlhf_reward).mean().item())
        mean_entropy = -logprobs.sum(1).mean()
        self.stats["objective/entropy"].append(self.accelerator.gather_for_metrics(mean_entropy).mean().item())
        chosen_rewards = self.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
        gathered_chosen_rewards = self.accelerator.gather_for_metrics(chosen_rewards)
        self.stats["rewards/chosen"].append(gathered_chosen_rewards.mean().item())
        rejected_rewards = self.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
        gathered_rejected_rewards = self.accelerator.gather_for_metrics(rejected_rewards)
        self.stats["rewards/rejected"].append(gathered_rejected_rewards.mean().item())
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

        if not self.control.should_training_stop:
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

        return loss.detach() / self.args.gradient_accumulation_steps
