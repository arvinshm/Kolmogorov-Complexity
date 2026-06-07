#!/usr/bin/env python3
"""Evaluate a base model or LoRA adapter on a math benchmark."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from data_utils import (
    DEFAULT_BENCHMARK_DATASET,
    DEFAULT_BENCHMARK_SPLIT,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    build_model_prompt,
    load_math_records,
    write_json,
    write_jsonl,
)
from math_utils import score_completion
from model_utils import get_model_device, load_causal_lm_for_eval, load_tokenizer


DEFAULT_OUTPUT_DIR = "results/eval_math"


@dataclass
class PredictionRow:
    """Serializable evaluation result for one completion."""

    id: str
    sample_index: int
    source: str
    subject: str | None
    level: int | str | None
    problem: str
    gold_answer: str
    predicted_answer: str
    normalized_gold: str
    normalized_prediction: str
    extraction_method: str
    correct: bool
    reward_if_used: float
    completion_tokens: int
    reasoning_tokens: int
    brevity_score: float
    raw_generation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model or adapter on MATH-500 or another math dataset."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--dataset-name", default=DEFAULT_BENCHMARK_DATASET)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default=DEFAULT_BENCHMARK_SPLIT)
    parser.add_argument("--dataset-jsonl", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples-per-problem", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--disable-math-verify", action="store_true")
    parser.add_argument("--brevity-weight", type=float, default=0.0)
    parser.add_argument("--brevity-length-cap", type=int, default=256)
    parser.add_argument(
        "--brevity-measure",
        choices=["reasoning_tokens", "completion_tokens"],
        default="reasoning_tokens",
    )

    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute a Wilson 95% confidence interval for a binomial proportion."""

    if total == 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * ((phat * (1 - phat) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize(rows: list[PredictionRow], args: argparse.Namespace) -> dict[str, Any]:
    """Aggregate prediction rows into benchmark metrics."""

    total_completions = len(rows)
    correct_completions = sum(row.correct for row in rows)
    problem_ids = sorted({row.id for row in rows})
    correct_problem_ids = sorted({row.id for row in rows if row.correct})
    completion_ci = wilson_interval(correct_completions, total_completions)
    prompt_ci = wilson_interval(len(correct_problem_ids), len(problem_ids))

    by_subject: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    by_level: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        subject_key = str(row.subject or "unknown")
        level_key = str(row.level or "unknown")
        by_subject[subject_key]["total"] += 1
        by_subject[subject_key]["correct"] += int(row.correct)
        by_level[level_key]["total"] += 1
        by_level[level_key]["correct"] += int(row.correct)

    def finish_group(group: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
        return {
            key: {
                "correct": value["correct"],
                "total": value["total"],
                "accuracy": value["correct"] / value["total"] if value["total"] else 0.0,
            }
            for key, value in sorted(group.items())
        }

    return {
        "model_name": args.model_name,
        "adapter_dir": args.adapter_dir,
        "dataset": args.dataset_jsonl or args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "num_problems": len(problem_ids),
        "num_completions": total_completions,
        "num_samples_per_problem": args.num_samples_per_problem,
        "correct_completions": correct_completions,
        "completion_accuracy": correct_completions / total_completions
        if total_completions
        else 0.0,
        "completion_accuracy_wilson_95": list(completion_ci),
        "problems_solved": len(correct_problem_ids),
        "pass_at_k": len(correct_problem_ids) / len(problem_ids) if problem_ids else 0.0,
        "pass_at_k_wilson_95": list(prompt_ci),
        "mean_completion_tokens": sum(row.completion_tokens for row in rows) / total_completions
        if total_completions
        else 0.0,
        "mean_reasoning_tokens": sum(row.reasoning_tokens for row in rows) / total_completions
        if total_completions
        else 0.0,
        "mean_correct_reasoning_tokens": (
            sum(row.reasoning_tokens for row in rows if row.correct) / correct_completions
            if correct_completions
            else None
        ),
        "by_subject": finish_group(by_subject),
        "by_level": finish_group(by_level),
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def evaluate(args: argparse.Namespace) -> tuple[list[PredictionRow], dict[str, Any]]:
    """Run generation and scoring."""

    import torch
    from transformers import set_seed

    set_seed(args.seed)

    records = load_math_records(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        limit=args.limit,
        jsonl_path=args.dataset_jsonl,
    )
    print(
        f"Loaded {len(records)} benchmark problems from "
        f"{args.dataset_jsonl or args.dataset_name}:{args.dataset_split}",
        flush=True,
    )

    tokenizer = load_tokenizer(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    model = load_causal_lm_for_eval(args)

    results: list[PredictionRow] = []
    prompts_done = 0
    for batch in batched(records, args.batch_size):
        prompts = [
            build_model_prompt(record.problem, tokenizer, args.system_prompt)
            for record in batch
        ]
        tokenized = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {
            key: value.to(get_model_device(model))
            for key, value in tokenized.items()
        }

        do_sample = args.temperature > 0.0 or args.num_samples_per_problem > 1
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": do_sample,
            "num_return_sequences": args.num_samples_per_problem,
        }
        if do_sample:
            generate_kwargs["temperature"] = args.temperature if args.temperature > 0 else 0.8
            generate_kwargs["top_p"] = args.top_p

        with torch.inference_mode():
            generated = model.generate(**tokenized, **generate_kwargs)

        prompt_width = tokenized["input_ids"].shape[1]
        new_token_ids = generated[:, prompt_width:]
        decoded = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

        for row_index, record in enumerate(batch):
            for sample_index in range(args.num_samples_per_problem):
                generation = decoded[
                    row_index * args.num_samples_per_problem + sample_index
                ]
                diag = score_completion(
                    completion=generation,
                    gold_answer=record.gold_answer,
                    tokenizer=tokenizer,
                    use_math_verify=not args.disable_math_verify,
                    brevity_length_cap=args.brevity_length_cap,
                    brevity_weight=args.brevity_weight,
                    brevity_measure=args.brevity_measure,
                )
                results.append(
                    PredictionRow(
                        id=record.id,
                        sample_index=sample_index,
                        source=record.source,
                        subject=record.subject,
                        level=record.level,
                        problem=record.problem,
                        gold_answer=record.gold_answer,
                        predicted_answer=diag["predicted_answer"],
                        normalized_gold=diag["normalized_gold"],
                        normalized_prediction=diag["normalized_prediction"],
                        extraction_method=diag["extraction_method"],
                        correct=bool(diag["correct"]),
                        reward_if_used=float(diag["reward"]),
                        completion_tokens=int(diag["completion_tokens"]),
                        reasoning_tokens=int(diag["reasoning_tokens"]),
                        brevity_score=float(diag["brevity_score"]),
                        raw_generation=generation,
                    )
                )

        prompts_done += len(batch)
        solved = len({row.id for row in results if row.correct})
        accuracy = sum(row.correct for row in results) / len(results)
        pass_at_k = solved / prompts_done if prompts_done else 0.0
        print(
            f"[{prompts_done:>4}/{len(records)} problems] "
            f"completion_accuracy={accuracy:.3f} "
            f"pass@{args.num_samples_per_problem}={pass_at_k:.3f} "
            f"solved={solved}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results, summarize(results, args)


def save_results(
    args: argparse.Namespace,
    rows: list[PredictionRow],
    summary: dict[str, Any],
) -> Path:
    """Write predictions, summary, and config."""

    run_name = args.run_name
    if run_name is None:
        run_name = "adapter" if args.adapter_dir else "base"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "predictions.jsonl", (asdict(row) for row in rows))
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "run_config.json", vars(args))
    print(f"Saved predictions to {output_dir / 'predictions.jsonl'}", flush=True)
    print(f"Saved summary to {output_dir / 'summary.json'}", flush=True)
    return output_dir


def main() -> None:
    args = parse_args()
    rows, summary = evaluate(args)
    save_results(args, rows, summary)
    print(
        "Final benchmark result: "
        f"completion_accuracy={100.0 * summary['completion_accuracy']:.2f}% "
        f"pass@{summary['num_samples_per_problem']}={100.0 * summary['pass_at_k']:.2f}% "
        f"mean_reasoning_tokens={summary['mean_reasoning_tokens']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
