import torch
from trl.trainer.utils import truncate_right


def create_pairs(self, completion_ids, prompt_ids, prompt_mask, solutions):
    batch_size = len(completion_ids) // 2
    eos_token_id = self.processing_class.eos_token_id
    pad_token_id = self.processing_class.pad_token_id
    decoded_completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

    # Format:
    # p1_c1 = prompt1, completion1
    # completion_ids: [p1_c1, p2_c1, p2_c1, p2_c2] -> [(p1_c1, p1_c2), (p2_c1, p2_c2)]
    # solutions: [sol1, sol2]
    #
    # (Greedy) Pair every completion with the solution
    # [(p1_c1, sol1), (p1_c2, sol1), (p2_c1, sol2), (p2_c2, sol2)]
    if self.strategy == "Greedy_Gold":
        completion_ids = (
            self.processing_class(
                decoded_completions + [str(x) for x in solutions] + [str(x) for x in solutions],
                padding=True,
                return_tensors="pt",
            )
            .to(self.args.device)
            .input_ids
        )
        completion_ids, completion_mask = truncate_right(completion_ids, eos_token_id, pad_token_id)
        prompt_ids = torch.cat([prompt_ids, prompt_ids], dim=0)
        prompt_mask = torch.cat([prompt_mask, prompt_mask], dim=0)
        solutions = solutions * 2
    # (Greedy_Online) Pair every completion with the solution and with each other
    # [(p1_c1, p1_c2), (p2_c1, p2_c2), (p1_c1, sol1), (p1_c2, sol1), (p2_c1, sol2), (p2_c2, sol2)]
    elif self.strategy == "Greedy_Mixed":
        completion_ids = (
            self.processing_class(
                decoded_completions[:batch_size]
                + [str(x) for x in solutions]
                + [str(x) for x in solutions]
                + decoded_completions[batch_size:]
                + decoded_completions,
                padding=True,
                return_tensors="pt",
            )
            .to(self.args.device)
            .input_ids
        )
        completion_ids, completion_mask = truncate_right(completion_ids, eos_token_id, pad_token_id)
        prompt_ids = torch.cat([prompt_ids[:batch_size], prompt_ids, prompt_ids[batch_size:], prompt_ids], dim=0)
        prompt_mask = torch.cat([prompt_mask[:batch_size], prompt_mask, prompt_mask[batch_size:], prompt_mask], dim=0)
        solutions = solutions * 3

    return completion_ids, completion_mask, prompt_ids, prompt_mask, solutions
