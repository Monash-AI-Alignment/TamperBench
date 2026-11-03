"""Chat encoding and residual-stream activations at the post-instruction positions.

Difference-in-means abliteration (Arditi et al. 2024; Heretic) reads the residual stream
at the tokens the chat template appends after the user's message, the last of which
predicts the first response token. These are the positions the moment penalty matches
and the probes inspect. Layer indices are `hidden_states` indices: 0 is the embedding
output and l is the residual stream after l blocks (Heretic's convention).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

_SENTINEL = "\u2063MOMENT-GUARD-SENTINEL\u2063"


def chat_text(tokenizer: PreTrainedTokenizerBase, prompt: str, response: str | None = None) -> str:
    """Render one user turn with the chat template, ending in the generation prompt or a response."""
    messages = [{"role": "user", "content": prompt}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return cast(str, tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=response is None))


def tokenize(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """Token ids of chat-template text (the template already holds any BOS token)."""
    return cast(list[int], tokenizer(text, add_special_tokens=False)["input_ids"])


def post_instruction_length(tokenizer: PreTrainedTokenizerBase) -> int:
    """Count the tokens the chat template adds after the user's message, up to the response."""
    text = chat_text(tokenizer, _SENTINEL)
    end = text.index(_SENTINEL) + len(_SENTINEL)
    return len(tokenize(tokenizer, text)) - len(tokenize(tokenizer, text[:end]))


@dataclass(frozen=True)
class Encoded:
    """One tokenized conversation.

    Attributes:
        input_ids: Prompt (with generation prompt) followed by the response, if any.
        prompt_length: Tokens up to and including the generation prompt.
    """

    input_ids: list[int]
    prompt_length: int


def encode(tokenizer: PreTrainedTokenizerBase, prompt: str, response: str | None, max_length: int) -> Encoded | None:
    """Tokenize a prompt (and response), or None if the prompt alone exceeds `max_length`.

    The response is truncated to fit. Returns None if the template does not render the
    prompt-with-generation-prompt as a token prefix of the full conversation.
    """
    prompt_ids = tokenize(tokenizer, chat_text(tokenizer, prompt))
    if len(prompt_ids) >= max_length:
        return None
    if response is None:
        return Encoded(prompt_ids, len(prompt_ids))
    full_ids = tokenize(tokenizer, chat_text(tokenizer, prompt, response))
    if full_ids[: len(prompt_ids)] != prompt_ids:
        return None
    return Encoded(full_ids[:max_length], len(prompt_ids))


def collate(examples: Sequence[Encoded], pad_token_id: int, num_positions: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch.

    Returns:
        input_ids, attention_mask; `labels` (response tokens, -100 elsewhere);
        `response_logit_mask` (positions whose logits predict a response token);
        `positions` (batch, num_positions): the last `num_positions` prompt tokens.
    """
    width = max(len(e.input_ids) for e in examples)
    batch = len(examples)
    input_ids = torch.full((batch, width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, width), dtype=torch.long)
    labels = torch.full((batch, width), -100, dtype=torch.long)
    response_logit_mask = torch.zeros((batch, width), dtype=torch.bool)
    positions = torch.empty((batch, num_positions), dtype=torch.long)
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row, :length] = torch.tensor(example.input_ids)
        attention_mask[row, :length] = 1
        labels[row, example.prompt_length : length] = input_ids[row, example.prompt_length : length]
        response_logit_mask[row, example.prompt_length - 1 : length - 1] = True
        positions[row] = torch.arange(example.prompt_length - num_positions, example.prompt_length)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "response_logit_mask": response_logit_mask,
        "positions": positions,
    }


def gather(hidden_states: Sequence[torch.Tensor], layers: Sequence[int], positions: torch.Tensor) -> torch.Tensor:
    """Residual streams at `positions` for `layers`, shape (layers, num_positions, batch, d)."""
    rows = torch.arange(positions.shape[0], device=positions.device)[:, None]
    return torch.stack([hidden_states[layer][rows, positions].transpose(0, 1) for layer in layers])


@torch.no_grad()
def collect(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    layers: Sequence[int],
    num_positions: int,
    batch_size: int = 16,
    max_length: int = 512,
) -> torch.Tensor:
    """Post-instruction activations of `prompts` (prompts too long are dropped).

    Returns:
        bfloat16 tensor on the CPU, shape (layers, num_positions, kept prompts, d).
    """
    pad = cast(int, tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    encoded = [e for e in (encode(tokenizer, p, None, max_length) for p in prompts) if e is not None]
    device = next(model.parameters()).device
    chunks = []
    for start in range(0, len(encoded), batch_size):
        batch = collate(encoded[start : start + batch_size], pad, num_positions)
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            output_hidden_states=True,
        )
        chunks.append(gather(outputs.hidden_states, layers, batch["positions"].to(device)).to("cpu", torch.bfloat16))
    return torch.cat(chunks, dim=2)
