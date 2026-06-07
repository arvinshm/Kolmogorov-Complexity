#!/usr/bin/env python3
"""Dataset and prompt utilities for the math RL experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from math_utils import extract_final_answer, extract_gsm8k_final_answer


DEFAULT_MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
DEFAULT_TRAIN_DATASET = "openai/gsm8k"
DEFAULT_TRAIN_CONFIG = "main"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_BENCHMARK_DATASET = "HuggingFaceH4/MATH-500"
DEFAULT_BENCHMARK_SPLIT = "test"

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem with concise reasoning. "
    "End with exactly one final answer written as \\boxed{...}. The boxed "
    "answer is the only part that will be graded."
)


@dataclass(frozen=True)
class MathRecord:
    """A standardized math problem record."""

    id: str
    problem: str
    gold_answer: str
    source: str
    subject: str | None = None
    level: int | str | None = None
    solution: str | None = None


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read JSONL records from disk."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSONL records to disk."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write an indented JSON file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def row_id(row: dict[str, Any], index: int) -> str:
    """Return a stable problem identifier when a dataset has one."""

    for key in ["unique_id", "id", "problem_id", "question_id"]:
        if key in row and row[key] not in [None, ""]:
            return str(row[key])
    return str(index)


def coerce_level(value: Any) -> int | str | None:
    """Normalize benchmark level metadata."""

    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return int(digits)
    return text


def standardize_math_row(
    row: dict[str, Any],
    index: int,
    source: str,
) -> MathRecord:
    """Convert common math dataset schemas into a MathRecord."""

    problem = (
        row.get("problem")
        or row.get("question")
        or row.get("prompt")
        or row.get("input")
        or row.get("query")
    )
    if problem is None:
        raise ValueError(f"Could not find a problem/question field in row {index}.")

    raw_answer = row.get("gold_answer") or row.get("final_answer") or row.get("target")
    solution = row.get("solution")
    answer_field = row.get("answer")

    if raw_answer is None and isinstance(answer_field, str):
        if "####" in answer_field:
            raw_answer = extract_gsm8k_final_answer(answer_field)
            solution = solution or answer_field
        elif len(answer_field) <= 200:
            raw_answer = answer_field
        else:
            raw_answer = extract_final_answer(answer_field).answer
            solution = solution or answer_field

    if raw_answer is None and isinstance(solution, str):
        raw_answer = extract_final_answer(solution).answer

    if raw_answer is None:
        raise ValueError(f"Could not find a final answer field in row {index}.")

    return MathRecord(
        id=row_id(row, index),
        problem=str(problem),
        gold_answer=str(raw_answer).strip(),
        source=source,
        subject=str(row["subject"]) if row.get("subject") is not None else None,
        level=coerce_level(row.get("level")),
        solution=str(solution) if solution is not None else None,
    )


def load_math_records(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    limit: int | None = None,
    jsonl_path: str | None = None,
) -> list[MathRecord]:
    """Load math records from Hugging Face datasets or a local JSONL file."""

    if jsonl_path:
        rows = read_jsonl(jsonl_path, limit=limit)
        source = str(jsonl_path)
    else:
        from datasets import load_dataset

        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        rows = [dict(row) for row in dataset]
        source = f"{dataset_name}/{dataset_config or ''}:{split}"

    return [
        standardize_math_row(row, index=index, source=source)
        for index, row in enumerate(rows)
    ]


def build_user_prompt(problem: str) -> str:
    """Build the user-visible prompt for one math problem."""

    return (
        "Solve the following math problem. Use concise reasoning, then finish "
        "with exactly one boxed final answer.\n\n"
        f"Problem:\n{problem}"
    )


def build_model_prompt(problem: str, tokenizer: Any, system_prompt: str) -> str:
    """Apply the tokenizer chat template when available."""

    user_prompt = build_user_prompt(problem)
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{system_prompt}\n\n{user_prompt}\n\nAnswer:"


def records_to_training_dataset(
    records: list[MathRecord],
    tokenizer: Any,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    """Create a prompt dataset suitable for TRL GRPOTrainer."""

    from datasets import Dataset

    rows = [
        {
            "id": record.id,
            "prompt": build_model_prompt(record.problem, tokenizer, system_prompt),
            "problem": record.problem,
            "answer": record.gold_answer,
            "source": record.source,
            "subject": record.subject,
            "level": record.level,
        }
        for record in records
    ]
    return Dataset.from_list(rows)

