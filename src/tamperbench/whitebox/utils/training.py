"""Shared pieces for the LoRA and full fine-tuning defenses: data files, models, saving."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from tamperbench.whitebox.utils.activations import Encoded, encode

ALL_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Rows of a JSON-lines file."""
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encode_conversations(
    tokenizer: PreTrainedTokenizerBase, pairs: Sequence[tuple[str, str]], max_length: int, min_prompt_tokens: int = 1
) -> list[tuple[Encoded, str]]:
    """Encode (prompt, response) pairs, dropping any that leave no response tokens within `max_length`.

    Returns:
        (encoding, prompt) for each kept pair, in input order.
    """
    kept = []
    for prompt, response in pairs:
        e = encode(tokenizer, prompt, response, max_length)
        if e is not None and e.prompt_length >= min_prompt_tokens and len(e.input_ids) > e.prompt_length:
            kept.append((e, prompt))
    if not kept:
        raise ValueError("no conversation fits max_length with a non-empty response")
    return kept


def pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """Return the pad id, falling back to EOS (padded positions are masked everywhere)."""
    return cast(int, tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)


def load_model(source: str) -> PreTrainedModel:
    """Load a causal LM in bfloat16 on the first GPU."""
    return AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.bfloat16, device_map={"": 0})


def lora_model(
    source: str,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str] = ALL_PROJECTIONS,
    layers_to_transform: Sequence[int] | None = None,
) -> PeftModel:
    """Load the base model with LoRA adapters and gradient checkpointing."""
    model = cast(
        PeftModel,
        get_peft_model(
            load_model(source),
            LoraConfig(
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=list(target_modules),
                layers_to_transform=None if layers_to_transform is None else list(layers_to_transform),
                bias="none",
            ),
        ),
    )
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # pyright: ignore[reportCallIssue]
    return model


def response_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over positions whose label is not -100."""
    return F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100)


def save_model(model: PreTrainedModel | PeftModel, source: str, output: Path) -> None:
    """Merge any LoRA adapters, save the weights, and copy the input checkpoint's tokenizer."""
    merged = cast(PreTrainedModel, model.merge_and_unload()) if isinstance(model, PeftModel) else model  # pyright: ignore[reportCallIssue]
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output))
    AutoTokenizer.from_pretrained(source).save_pretrained(str(output))
