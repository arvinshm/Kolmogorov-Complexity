#!/usr/bin/env python3
"""Shared answer extraction, verification, and length utilities.

The experiment uses verifiable final answers as the RL signal. This module is
deliberately conservative: it prefers `math-verify` when available, and falls
back to exact normalized strings plus simple numeric equivalence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


BOXED_MARKER = "\\boxed"


@dataclass(frozen=True)
class ExtractedAnswer:
    """The answer pulled from a model completion or dataset solution."""

    answer: str
    method: str


def completion_to_text(completion: Any) -> str:
    """Convert TRL/HF completion payloads into plain text."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    if isinstance(completion, list):
        pieces: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                pieces.append(str(item.get("content", "")))
            else:
                pieces.append(str(item))
        return "".join(pieces)
    return str(completion)


def extract_braced_content(text: str, start_index: int) -> str | None:
    """Return the balanced content inside `{...}` starting at `start_index`."""

    if start_index >= len(text) or text[start_index] != "{":
        return None
    depth = 0
    pieces: list[str] = []
    for pos in range(start_index, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
            if depth > 1:
                pieces.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(pieces).strip()
            if depth < 0:
                return None
            pieces.append(char)
        else:
            if depth >= 1:
                pieces.append(char)
    return None


def extract_last_boxed(text: str) -> str:
    """Extract the last `\\boxed{...}` expression from `text`."""

    start = text.rfind(BOXED_MARKER)
    while start != -1:
        brace_index = start + len(BOXED_MARKER)
        while brace_index < len(text) and text[brace_index].isspace():
            brace_index += 1
        content = extract_braced_content(text, brace_index)
        if content is not None:
            return content
        start = text.rfind(BOXED_MARKER, 0, start)
    return ""


def extract_gsm8k_final_answer(answer_text: str) -> str:
    """Extract the final GSM8K answer after the `####` delimiter."""

    marker = "####"
    if marker in answer_text:
        return answer_text.rsplit(marker, 1)[-1].strip()
    return extract_final_answer(answer_text).answer


def extract_final_answer(text: str) -> ExtractedAnswer:
    """Extract a final answer from model or dataset text."""

    value = str(text).strip()
    boxed = extract_last_boxed(value)
    if boxed:
        return ExtractedAnswer(boxed, "boxed")

    if "####" in value:
        return ExtractedAnswer(value.rsplit("####", 1)[-1].strip(), "gsm8k_hash")

    patterns = [
        r"final answer\s*(?:is|:)\s*(.+)",
        r"answer\s*(?:is|:)\s*(.+)",
        r"therefore\s*,?\s*(.+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, value, flags=re.IGNORECASE)
        if matches:
            candidate = matches[-1].strip()
            candidate = re.split(r"[\n\r]", candidate)[0].strip()
            return ExtractedAnswer(candidate, "answer_header")

    nonempty_lines = [line.strip() for line in value.splitlines() if line.strip()]
    if nonempty_lines:
        last_line = nonempty_lines[-1].rstrip(".")
        return ExtractedAnswer(last_line, "last_line")
    return ExtractedAnswer("", "empty")


def strip_outer_curly(text: str) -> str:
    """Strip redundant outer `{...}` wrappers."""

    value = text.strip()
    changed = True
    while changed and value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        changed = bool(inner)
        if changed:
            value = inner
    return value


def normalize_answer(text: str) -> str:
    """Canonicalize presentation-only answer differences."""

    value = str(text).strip()
    value = value.replace("\n", " ")
    value = value.replace("$", "")
    value = value.replace("\\left", "")
    value = value.replace("\\right", "")
    value = value.replace("\\!", "")
    value = value.replace("\\,", "")
    value = value.replace("\\;", "")
    value = value.replace("\\:", "")
    value = value.replace("\\quad", " ")
    value = value.replace("\\qquad", " ")
    value = value.replace("\\tfrac", "\\frac")
    value = value.replace("\\dfrac", "\\frac")
    value = re.sub(r"\\text\s*{([^{}]*)}", r"\1", value)
    value = re.sub(r"\\mathrm\s*{([^{}]*)}", r"\1", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = value.rstrip(".")
    value = strip_outer_curly(value)
    value = value.replace(" ", "")
    return value


def latex_frac_to_plain(value: str) -> str:
    """Convert simple `\\frac{a}{b}` strings into `a/b`."""

    pattern = re.compile(r"\\frac\s*{([^{}]+)}\s*{([^{}]+)}")
    while True:
        match = pattern.search(value)
        if not match:
            return value
        replacement = f"({match.group(1)})/({match.group(2)})"
        value = value[: match.start()] + replacement + value[match.end() :]


def parse_numeric_answer(text: str) -> Fraction | None:
    """Parse a simple numeric answer into a Fraction if possible."""

    value = normalize_answer(text)
    value = latex_frac_to_plain(value)
    value = value.replace(",", "")
    value = value.replace("\\%", "%")
    percent = value.endswith("%")
    if percent:
        value = value[:-1]
    if not value:
        return None

    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if percent:
        parsed /= 100
    return parsed


def verify_with_math_verify(prediction: str, gold: str) -> bool | None:
    """Try Hugging Face math-verify, returning None if unavailable or unsure."""

    try:
        from math_verify import parse, verify
    except Exception:
        return None

    prediction_candidates = [
        prediction,
        f"\\boxed{{{prediction}}}",
        f"Answer: {prediction}",
    ]
    gold_candidates = [
        gold,
        f"\\boxed{{{gold}}}",
        f"Answer: {gold}",
    ]
    for gold_text in gold_candidates:
        for pred_text in prediction_candidates:
            try:
                parsed_gold = parse(gold_text)
                parsed_prediction = parse(pred_text)
                if parsed_gold and parsed_prediction and bool(
                    verify(parsed_gold, parsed_prediction)
                ):
                    return True
            except Exception:
                continue
    return None


def answers_equivalent(prediction: str, gold: str, use_math_verify: bool = True) -> bool:
    """Return True if the predicted and gold final answers match."""

    prediction = str(prediction).strip()
    gold = str(gold).strip()
    if not prediction or not gold:
        return False

    if use_math_verify:
        verified = verify_with_math_verify(prediction, gold)
        if verified is True:
            return True

    if normalize_answer(prediction) == normalize_answer(gold):
        return True

    pred_numeric = parse_numeric_answer(prediction)
    gold_numeric = parse_numeric_answer(gold)
    return pred_numeric is not None and pred_numeric == gold_numeric


def reasoning_text_for_length(completion: str) -> str:
    """Return the generated reasoning span used for brevity rewards.

    The final boxed answer is not the interesting part of this experiment, so
    this trims everything from the last answer marker onward when possible.
    """

    text = completion_to_text(completion)
    answer_positions = [
        pos
        for pos in [
            text.rfind(BOXED_MARKER),
            text.lower().rfind("final answer"),
            text.lower().rfind("answer:"),
        ]
        if pos > 0
    ]
    if answer_positions:
        return text[: min(answer_positions)].strip()
    return text.strip()


def count_tokens(text: str, tokenizer: Any | None = None) -> int:
    """Count tokens with the model tokenizer, falling back to whitespace."""

    value = str(text)
    if tokenizer is None:
        return len(value.split())
    try:
        return len(tokenizer(value, add_special_tokens=False)["input_ids"])
    except Exception:
        return len(value.split())


def brevity_score(length: int, length_cap: int) -> float:
    """Map a nonnegative length to a [0, 1] brevity score."""

    if length_cap <= 0:
        raise ValueError("length_cap must be positive.")
    return max(0.0, 1.0 - (max(0, length) / float(length_cap)))


def score_completion(
    completion: Any,
    gold_answer: str,
    tokenizer: Any | None = None,
    use_math_verify: bool = True,
    brevity_length_cap: int = 256,
    brevity_weight: float = 0.0,
    brevity_measure: str = "reasoning_tokens",
) -> dict[str, Any]:
    """Score one completion and return reward plus diagnostics."""

    text = completion_to_text(completion)
    extracted = extract_final_answer(text)
    correct = answers_equivalent(
        extracted.answer,
        gold_answer,
        use_math_verify=use_math_verify,
    )

    completion_tokens = count_tokens(text, tokenizer)
    reasoning_tokens = count_tokens(reasoning_text_for_length(text), tokenizer)
    if brevity_measure == "completion_tokens":
        measured_length = completion_tokens
    elif brevity_measure == "reasoning_tokens":
        measured_length = reasoning_tokens
    else:
        raise ValueError(f"Unknown brevity_measure: {brevity_measure}")

    brevity = brevity_score(measured_length, brevity_length_cap)
    reward = 0.0
    if correct:
        reward = 1.0 + max(0.0, brevity_weight) * brevity

    return {
        "reward": reward,
        "correct": correct,
        "predicted_answer": extracted.answer,
        "extraction_method": extracted.method,
        "normalized_prediction": normalize_answer(extracted.answer),
        "normalized_gold": normalize_answer(gold_answer),
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "brevity_score": brevity,
        "measured_length": measured_length,
        "raw_completion": text,
    }

