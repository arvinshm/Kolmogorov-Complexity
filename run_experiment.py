#!/usr/bin/env python3
"""Orchestrate the two-condition RL experiment from a single command."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from data_utils import (
    DEFAULT_BENCHMARK_DATASET,
    DEFAULT_BENCHMARK_SPLIT,
    DEFAULT_MODEL,
    DEFAULT_TRAIN_CONFIG,
    DEFAULT_TRAIN_DATASET,
    DEFAULT_TRAIN_SPLIT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run correctness-only RL, brevity RL, benchmark evals, and comparison."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--work-dir", default="results/two_condition_experiment")
    parser.add_argument("--train-dataset-name", default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--train-dataset-config", default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--benchmark-dataset-name", default=DEFAULT_BENCHMARK_DATASET)
    parser.add_argument("--benchmark-dataset-config", default=None)
    parser.add_argument("--benchmark-split", default=DEFAULT_BENCHMARK_SPLIT)
    parser.add_argument("--benchmark-jsonl", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--benchmark-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--brevity-weight", type=float, default=0.25)
    parser.add_argument("--brevity-length-cap", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--train-extra-args", default="")
    parser.add_argument("--eval-extra-args", default="")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument("--dry-print", action="store_true")
    return parser.parse_args()


def script_path(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def add_if_present(command: list[str], flag: str, value) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def run(command: list[str], dry_print: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"\n$ {printable}", flush=True)
    if not dry_print:
        subprocess.run(command, check=True)


def train_command(args: argparse.Namespace, reward_mode: str, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        script_path("train_rl_math.py"),
        "--model-name",
        args.model_name,
        "--output-dir",
        str(output_dir),
        "--train-dataset-name",
        args.train_dataset_name,
        "--train-split",
        args.train_split,
        "--seed",
        str(args.seed),
        "--reward-mode",
        reward_mode,
        "--max-steps",
        str(args.max_steps),
        "--num-generations",
        str(args.num_generations),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--per-device-train-batch-size",
        str(args.per_device_train_batch_size),
    ]
    if args.train_dataset_config is not None:
        command.extend(["--train-dataset-config", args.train_dataset_config])
    add_if_present(command, "--train-jsonl", args.train_jsonl)
    add_if_present(command, "--max-train-samples", args.max_train_samples)
    if reward_mode == "correctness_brevity":
        command.extend(
            [
                "--brevity-weight",
                str(args.brevity_weight),
                "--brevity-length-cap",
                str(args.brevity_length_cap),
            ]
        )
    command.extend(shlex.split(args.train_extra_args))
    return command


def eval_command(
    args: argparse.Namespace,
    adapter_dir: Path | None,
    output_dir: Path,
    run_name: str,
) -> list[str]:
    command = [
        sys.executable,
        script_path("evaluate_math.py"),
        "--model-name",
        args.model_name,
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
        "--dataset-name",
        args.benchmark_dataset_name,
        "--dataset-split",
        args.benchmark_split,
        "--seed",
        str(args.seed),
    ]
    if args.benchmark_dataset_config is not None:
        command.extend(["--dataset-config", args.benchmark_dataset_config])
    add_if_present(command, "--dataset-jsonl", args.benchmark_jsonl)
    add_if_present(command, "--limit", args.benchmark_limit)
    if adapter_dir is not None:
        command.extend(["--adapter-dir", str(adapter_dir)])
    command.extend(shlex.split(args.eval_extra_args))
    return command


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    train_correctness_dir = work_dir / "adapters" / "correctness"
    train_brevity_dir = work_dir / "adapters" / f"brevity_lambda_{args.brevity_weight:g}"
    eval_dir = work_dir / "eval"

    if not args.skip_base_eval:
        run(eval_command(args, None, eval_dir, "base"), args.dry_print)

    if not args.skip_training:
        run(train_command(args, "correctness", train_correctness_dir), args.dry_print)
        run(train_command(args, "correctness_brevity", train_brevity_dir), args.dry_print)

    run(eval_command(args, train_correctness_dir, eval_dir, "correctness"), args.dry_print)
    run(eval_command(args, train_brevity_dir, eval_dir, "brevity"), args.dry_print)

    comparison_json = work_dir / "comparison_brevity_vs_correctness.json"
    compare_command = [
        sys.executable,
        script_path("compare_runs.py"),
        "--a-predictions",
        str(eval_dir / "correctness" / "predictions.jsonl"),
        "--b-predictions",
        str(eval_dir / "brevity" / "predictions.jsonl"),
        "--a-name",
        "correctness",
        "--b-name",
        f"brevity_lambda_{args.brevity_weight:g}",
        "--output-json",
        str(comparison_json),
        "--seed",
        str(args.seed),
    ]
    run(compare_command, args.dry_print)


if __name__ == "__main__":
    main()

