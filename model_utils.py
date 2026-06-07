#!/usr/bin/env python3
"""Model loading helpers shared by training and evaluation scripts."""

from __future__ import annotations

from typing import Any


def version_tuple(version: str) -> tuple[int, ...]:
    """Convert a version string like `0.10.0` into comparable integers."""

    pieces: list[int] = []
    for part in version.split("."):
        digits = "".join(char for char in part if char.isdigit())
        if not digits:
            break
        pieces.append(int(digits))
    return tuple(pieces)


def disable_incompatible_torchao() -> None:
    """Avoid optional PEFT torchao dispatcher crashes on old torchao builds."""

    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return

    try:
        torchao_version = version("torchao")
    except PackageNotFoundError:
        return

    if version_tuple(torchao_version) > (0, 16, 0):
        return

    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.torchao as peft_lora_torchao
    except Exception:
        return

    peft_import_utils.is_torchao_available = lambda: False
    peft_lora_torchao.is_torchao_available = lambda: False
    print(
        f"Disabled optional PEFT torchao dispatcher because torchao "
        f"{torchao_version} is too old for this PEFT version.",
        flush=True,
    )


def resolve_dtype(name: str):
    """Return a torch dtype or the string `auto`."""

    if name == "auto":
        return "auto"
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_tokenizer(model_name: str, trust_remote_code: bool = False, padding_side: str = "left"):
    """Load a tokenizer with a valid pad token."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def model_kwargs_from_args(args: Any, for_training: bool) -> dict[str, Any]:
    """Build common AutoModelForCausalLM kwargs from an argparse namespace."""

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": resolve_dtype(args.torch_dtype),
    }
    if args.load_in_4bit:
        import torch
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif not for_training:
        model_kwargs["device_map"] = "auto"
    return model_kwargs


def lora_config_from_args(args: Any):
    """Create the LoRA configuration used by both RL conditions."""

    disable_incompatible_torchao()

    from peft import LoraConfig, TaskType

    target_modules = [
        item.strip()
        for item in str(args.lora_target_modules).split(",")
        if item.strip()
    ]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


def load_causal_lm_for_training(args: Any):
    """Load the policy model for GRPO training."""

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs_from_args(args, for_training=True),
    )
    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if args.initial_adapter_dir:
        from peft import PeftModel

        disable_incompatible_torchao()
        print(f"Loading initial trainable adapter: {args.initial_adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(
            model,
            args.initial_adapter_dir,
            is_trainable=True,
        )
    return model


def load_causal_lm_for_eval(args: Any):
    """Load a base model plus an optional PEFT adapter for evaluation."""

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs_from_args(args, for_training=False),
    )
    if args.adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    return model


def get_model_device(model: Any):
    """Return the device where input tensors should be placed."""

    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device

