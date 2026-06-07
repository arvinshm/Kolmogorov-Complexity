#!/usr/bin/env python3
"""Statistical comparison for two paired math benchmark evaluations."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from data_utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two paired prediction files with McNemar and bootstrap tests."
    )
    parser.add_argument("--a-predictions", required=True)
    parser.add_argument("--b-predictions", required=True)
    parser.add_argument("--a-name", default="A")
    parser.add_argument("--b-name", default="B")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def keyed_correct(rows: list[dict[str, Any]], sample_index: int) -> dict[str, dict[str, Any]]:
    """Keep exactly one completion per problem id for paired comparison."""

    keyed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if int(row.get("sample_index", 0)) != sample_index:
            continue
        key = str(row["id"])
        keyed[key] = row
    return keyed


def exact_mcnemar_pvalue(b_correct_a_wrong: int, a_correct_b_wrong: int) -> float:
    """Two-sided exact binomial McNemar p-value."""

    discordant = b_correct_a_wrong + a_correct_b_wrong
    if discordant == 0:
        return 1.0
    smaller = min(b_correct_a_wrong, a_correct_b_wrong)
    cdf = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * cdf)


def bootstrap_diff_ci(
    paired_diffs: list[int],
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Paired bootstrap CI for accuracy difference B - A."""

    if not paired_diffs:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(paired_diffs)
    estimates: list[float] = []
    for _ in range(iterations):
        total = 0
        for _ in range(n):
            total += paired_diffs[rng.randrange(n)]
        estimates.append(total / n)
    estimates.sort()
    lo = estimates[int(0.025 * (iterations - 1))]
    hi = estimates[int(0.975 * (iterations - 1))]
    return lo, hi


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    a_rows = keyed_correct(read_jsonl(args.a_predictions), args.sample_index)
    b_rows = keyed_correct(read_jsonl(args.b_predictions), args.sample_index)
    shared_ids = sorted(set(a_rows) & set(b_rows))
    if not shared_ids:
        raise ValueError("No shared problem ids found between prediction files.")

    a_correct = [bool(a_rows[key]["correct"]) for key in shared_ids]
    b_correct = [bool(b_rows[key]["correct"]) for key in shared_ids]
    paired_diffs = [int(b) - int(a) for a, b in zip(a_correct, b_correct)]

    both_correct = sum(a and b for a, b in zip(a_correct, b_correct))
    both_wrong = sum((not a) and (not b) for a, b in zip(a_correct, b_correct))
    a_only = sum(a and not b for a, b in zip(a_correct, b_correct))
    b_only = sum(b and not a for a, b in zip(a_correct, b_correct))
    n = len(shared_ids)
    a_accuracy = sum(a_correct) / n
    b_accuracy = sum(b_correct) / n
    diff = b_accuracy - a_accuracy
    ci_low, ci_high = bootstrap_diff_ci(
        paired_diffs,
        iterations=args.bootstrap_iters,
        seed=args.seed,
    )

    a_reasoning = [
        float(a_rows[key]["reasoning_tokens"])
        for key in shared_ids
        if a_rows[key].get("reasoning_tokens") is not None
    ]
    b_reasoning = [
        float(b_rows[key]["reasoning_tokens"])
        for key in shared_ids
        if b_rows[key].get("reasoning_tokens") is not None
    ]
    a_correct_reasoning = [
        float(a_rows[key]["reasoning_tokens"])
        for key in shared_ids
        if a_rows[key].get("reasoning_tokens") is not None and a_rows[key]["correct"]
    ]
    b_correct_reasoning = [
        float(b_rows[key]["reasoning_tokens"])
        for key in shared_ids
        if b_rows[key].get("reasoning_tokens") is not None and b_rows[key]["correct"]
    ]

    return {
        "a_name": args.a_name,
        "b_name": args.b_name,
        "sample_index": args.sample_index,
        "paired_problem_count": n,
        "a_accuracy": a_accuracy,
        "b_accuracy": b_accuracy,
        "accuracy_difference_b_minus_a": diff,
        "bootstrap_95_ci_b_minus_a": [ci_low, ci_high],
        "mcnemar": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "a_correct_b_wrong": a_only,
            "b_correct_a_wrong": b_only,
            "exact_two_sided_p": exact_mcnemar_pvalue(b_only, a_only),
        },
        "lengths": {
            "a_mean_reasoning_tokens": mean(a_reasoning),
            "b_mean_reasoning_tokens": mean(b_reasoning),
            "a_mean_correct_reasoning_tokens": mean(a_correct_reasoning),
            "b_mean_correct_reasoning_tokens": mean(b_correct_reasoning),
        },
    }


def main() -> None:
    args = parse_args()
    result = compare(args)
    output_json = args.output_json
    if output_json is None:
        output_json = str(Path(args.b_predictions).with_name("comparison_vs_a.json"))
    write_json(output_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved comparison to {output_json}")


if __name__ == "__main__":
    main()

