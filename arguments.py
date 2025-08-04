import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="semantle", choices=["semantle", "arc", "molecule"])
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target", "-t", type=str, default="computer")
    parser.add_argument("--strategy", type=str, default="gold")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--opro_sampling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=10)
    parser.add_argument("--online_temperature", type=float, default=1.0)
    parser.add_argument("--online_max_completion_length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=int, default=0)
    parser.add_argument("--target_modules", type=list[str], default=["q_proj", "v_proj"])
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_permutations", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--readable_prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_training_size", type=int, default=50)
    parser.add_argument("--max_training_size", type=int, default=80)
    parser.add_argument("--max_validation_size", type=int, default=64)
    parser.add_argument("--max_test_size", type=int, default=64)
    parser.add_argument("--validation_interval", type=int, default=5)
    parser.add_argument("--use_vllm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grpo_weight", type=float, default=1.0)
    parser.add_argument("--nll_weight", type=float, default=0.0)
    parser.add_argument("--pro_loss_weight", type=float, default=0.0)
    parser.add_argument("--pro_loss_only_positive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_temperature", type=float, default=1.0)
    parser.add_argument("--use_train_temp_schedule", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb_prefix", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--save_datasets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inject_best_at_lowest_score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_early_stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--trl_seed", type=int, default=42)
    parser.add_argument("--warmstart_seed", type=int, default=42, choices=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--dataset_file", type=str, default="kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json"
    )
    parser.add_argument(
        "--dataset_solutions_file",
        type=str,
        default="kaggle/input/arc-prize-2024/arc-agi_evaluation_solutions.json",
    )
    parser.add_argument("--compute_entropy", action=argparse.BooleanOptionalAction, default=False)

    # Semantle-only arguments
    parser.add_argument("--num_guesses", type=int, default=10)
    parser.add_argument("--warmstart", type=float, default=0)
    # parser.add_argument("--encoder", type=str, default="chandar-lab/NeoBERT")
    parser.add_argument("--embedder", type=str, default="princeton-nlp/sup-simcse-roberta-large")
    parser.add_argument("--semantle_warmstart_file", type=str, default="warmstart/warmstart-20.json")

    # ARC-only arguments
    parser.add_argument("--use_barc_format", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_induction", action=argparse.BooleanOptionalAction, default=False)

    # Arguments for NS sampling
    parser.add_argument(
        "--migrate_sampling_strategy", type=str, choices=["none", "opro", "random_topk", "evo"], default="none"
    )
    parser.add_argument("--migrate_gamma", type=int, default=0)
    parser.add_argument("--migrate_alpha", type=int, default=0)
    parser.add_argument("--migrate_beta", type=int, default=0)
    parser.add_argument("--greedy_topk", type=int, default=1)
    parser.add_argument("--include_scores", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--evo_num_islands", type=int, default=4)
    parser.add_argument("--evo_migration_interval", type=int, default=40)
    parser.add_argument("--evo_migration_rate", type=float, default=0.05)
    parser.add_argument("--evo_archive_size", type=int, default=5)
    parser.add_argument("--evo_exploration_ratio", type=float, default=0.2)
    parser.add_argument("--evo_exploitation_ratio", type=float, default=0.8)
    parser.add_argument("--evo_exploration_topk", type=int, default=3)
    parser.add_argument("--evo_exploitation_topk", type=int, default=3)

    # Arguments for group construction
    parser.add_argument("--replace_ns_prompt", action=argparse.BooleanOptionalAction, default=True)

    # Arguments for final inference
    parser.add_argument("--only_inference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inf_batch_size", type=int, default=10)
    parser.add_argument("--inf_temperature", type=float, default=1.0)
    parser.add_argument("--inf_max_new_tokens", type=int, default=512)
    parser.add_argument("--inf_num_samples", type=int, default=64)

    args = parser.parse_args()
    return args
