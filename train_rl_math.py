#!/usr/bin/env python3
"""Train a math model with GRPO and verifiable answer rewards.

Two reward modes are supported:

1. `correctness`: reward is 1 iff the final answer is correct, otherwise 0.
2. `correctness_brevity`: incorrect answers still get 0, while correct answers
   receive an extra brevity bonus controlled by `--brevity-weight`.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from data_utils import (
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TRAIN_CONFIG,
    DEFAULT_TRAIN_DATASET,
    DEFAULT_TRAIN_SPLIT,
    load_math_records,
    records_to_training_dataset,
    write_json,
)
from math_utils import score_completion
from model_utils import (
    load_causal_lm_for_training,
    load_tokenizer,
    lora_config_from_args,
)


DEFAULT_OUTPUT_DIR = "results/rl_math"


class MathReward:
    """Callable reward object used by TRL's GRPOTrainer."""

    def __init__(
        self,
        tokenizer: Any,
        reward_mode: str,
        brevity_weight: float,
        brevity_length_cap: int,
        brevity_measure: str,
        use_math_verify: bool,
        log_every: int,
    ):
        self.tokenizer = tokenizer
        self.reward_mode = reward_mode
        self.brevity_weight = brevity_weight if reward_mode == "correctness_brevity" else 0.0
        self.brevity_length_cap = brevity_length_cap
        self.brevity_measure = brevity_measure
        self.use_math_verify = use_math_verify
        self.log_every = log_every
        self.calls = 0
        self.scored_completions = 0

    def __call__(self, completions, answer, **kwargs):
        rewards: list[float] = []
        diagnostics: list[dict[str, Any]] = []
        for completion, gold_answer in zip(completions, answer):
            diag = score_completion(
                completion=completion,
                gold_answer=str(gold_answer),
                tokenizer=self.tokenizer,
                use_math_verify=self.use_math_verify,
                brevity_length_cap=self.brevity_length_cap,
                brevity_weight=self.brevity_weight,
                brevity_measure=self.brevity_measure,
            )
            rewards.append(float(diag["reward"]))
            diagnostics.append(diag)

        self.calls += 1
        self.scored_completions += len(rewards)
        if self.log_every > 0 and self.calls % self.log_every == 0 and rewards:
            correct_rate = sum(1 for item in diagnostics if item["correct"]) / len(diagnostics)
            mean_reward = sum(rewards) / len(rewards)
            mean_reasoning = sum(item["reasoning_tokens"] for item in diagnostics) / len(diagnostics)
            mean_completion = sum(item["completion_tokens"] for item in diagnostics) / len(diagnostics)
            print(
                "[reward] "
                f"calls={self.calls} completions={self.scored_completions} "
                f"batch_correct={correct_rate:.3f} "
                f"batch_reward={mean_reward:.3f} "
                f"reasoning_tokens={mean_reasoning:.1f} "
                f"completion_tokens={mean_completion:.1f}",
                flush=True,
            )
        return rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a math model with GRPO on verifiable-answer rewards."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--train-dataset-name", default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--train-dataset-config", default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-adapter-dir", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--shuffle-train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)

    parser.add_argument(
        "--reward-mode",
        choices=["correctness", "correctness_brevity"],
        default="correctness",
    )
    parser.add_argument(
        "--brevity-weight",
        type=float,
        default=0.25,
        help="Lambda for the brevity bonus. Only used in correctness_brevity mode.",
    )
    parser.add_argument("--brevity-length-cap", type=int, default=256)
    parser.add_argument(
        "--brevity-measure",
        choices=["reasoning_tokens", "completion_tokens"],
        default="reasoning_tokens",
    )
    parser.add_argument("--disable-math-verify", action="store_true")
    parser.add_argument("--reward-log-every", type=int, default=20)

    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reward-test", action="store_true")
    return parser.parse_args()


def make_grpo_config(args: argparse.Namespace):
    """Create a GRPOConfig while tolerating small TRL version differences."""

    from trl import GRPOConfig

    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if effective_batch % args.num_generations != 0:
        raise ValueError(
            "per-device batch size * gradient accumulation steps must be divisible "
            "by --num-generations for GRPO."
        )

    config_values = {
        "output_dir": args.output_dir,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "report_to": "none",
        "use_vllm": False,
    }

    while True:
        try:
            return GRPOConfig(**config_values)
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if not match:
                raise
            bad_key = match.group(1)
            if bad_key not in config_values:
                raise
            print(f"TRL GRPOConfig rejected {bad_key!r}; retrying without it.", flush=True)
            config_values.pop(bad_key)


def write_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    """Persist enough metadata to reproduce the run."""

    payload = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "script": "train_rl_math.py",
        "args": vars(args),
        "reward": {
            "mode": args.reward_mode,
            "correctness": "1 if final answer verifies against gold answer else 0",
            "brevity_bonus": (
                "correct * brevity_weight * max(0, 1 - measured_tokens / "
                "brevity_length_cap)"
            ),
            "measured_length": args.brevity_measure,
        },
    }
    write_json(output_dir / "run_config.json", payload)


def load_training_data(args: argparse.Namespace, tokenizer: Any):
    """Load and prepare the RL prompt dataset."""

    records = load_math_records(
        dataset_name=args.train_dataset_name,
        dataset_config=args.train_dataset_config,
        split=args.train_split,
        limit=args.max_train_samples,
        jsonl_path=args.train_jsonl,
    )
    if args.shuffle_train:
        import random

        rng = random.Random(args.seed)
        rng.shuffle(records)
    dataset = records_to_training_dataset(
        records,
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
    )
    return records, dataset


def print_dataset_preview(records, dataset, tokenizer) -> None:
    """Print a compact preview before expensive training starts."""

    print(f"Prepared {len(records)} RL training problems.", flush=True)
    if not records:
        return
    first = records[0]
    prompt = dataset[0]["prompt"]
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    print(
        "First problem preview:\n"
        f"  id={first.id}\n"
        f"  source={first.source}\n"
        f"  gold_answer={first.gold_answer}\n"
        f"  prompt_tokens={prompt_tokens}\n"
        f"  problem={first.problem[:240].replace(chr(10), ' ')}",
        flush=True,
    )


def run_reward_test(args: argparse.Namespace) -> None:
    """Exercise the reward function without loading a model."""

    reward = MathReward(
        tokenizer=None,
        reward_mode=args.reward_mode,
        brevity_weight=args.brevity_weight,
        brevity_length_cap=args.brevity_length_cap,
        brevity_measure=args.brevity_measure,
        use_math_verify=not args.disable_math_verify,
        log_every=1,
    )
    completions = [
        "We compute 6 * 7 = 42. Therefore the final answer is \\boxed{42}.",
        "A long but wrong derivation. Final answer: \\boxed{43}.",
        "Just \\boxed{42}.",
    ]
    scores = reward(completions, answer=["42", "42", "42"])
    for score, completion in zip(scores, completions):
        print(f"reward={score:.4f} completion={completion}")


def run_dry_check(args: argparse.Namespace) -> None:
    """Load data and score a few synthetic completions without loading a model."""

    tokenizer = load_tokenizer(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    records, dataset = load_training_data(args, tokenizer)
    print_dataset_preview(records, dataset, tokenizer)
    reward = MathReward(
        tokenizer=tokenizer,
        reward_mode=args.reward_mode,
        brevity_weight=args.brevity_weight,
        brevity_length_cap=args.brevity_length_cap,
        brevity_measure=args.brevity_measure,
        use_math_verify=not args.disable_math_verify,
        log_every=1,
    )
    completions = [f"Final answer: \\boxed{{{record.gold_answer}}}" for record in records[:5]]
    scores = reward(completions, answer=[record.gold_answer for record in records[:5]])
    for record, score in zip(records[:5], scores):
        print(f"id={record.id} reward={score:.4f} answer={record.gold_answer}")


def main() -> None:
    args = parse_args()
    if args.reward_test:
        run_reward_test(args)
        return
    if args.dry_run:
        run_dry_check(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(args, output_dir)

    print(
        "Starting math RL training:\n"
        f"  model={args.model_name}\n"
        f"  train={args.train_jsonl or args.train_dataset_name}\n"
        f"  reward_mode={args.reward_mode}\n"
        f"  brevity_weight={args.brevity_weight if args.reward_mode == 'correctness_brevity' else 0.0}\n"
        f"  output_dir={output_dir}",
        flush=True,
    )

    tokenizer = load_tokenizer(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    records, train_dataset = load_training_data(args, tokenizer)
    print_dataset_preview(records, train_dataset, tokenizer)

    model = load_causal_lm_for_training(args)
    reward_func = MathReward(
        tokenizer=tokenizer,
        reward_mode=args.reward_mode,
        brevity_weight=args.brevity_weight,
        brevity_length_cap=args.brevity_length_cap,
        brevity_measure=args.brevity_measure,
        use_math_verify=not args.disable_math_verify,
        log_every=args.reward_log_every,
    )

    from trl import GRPOTrainer

    trainer_kwargs = {
        "model": model,
        "reward_funcs": reward_func,
        "args": make_grpo_config(args),
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
    }
    if args.initial_adapter_dir is None:
        trainer_kwargs["peft_config"] = lora_config_from_args(args)

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Saved adapter/model artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

