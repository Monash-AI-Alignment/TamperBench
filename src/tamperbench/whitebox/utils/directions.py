"""Refusal directions and in-forward ablation hooks, shared by direction-based defenses.

Layer conventions follow `hidden_states`: index l is the residual stream after l decoder
blocks (0 is the embedding output), so residual index l is the output of block l - 1.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from tamperbench.whitebox.utils.activations import collect


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The model's list of decoder blocks (works through PEFT wrappers)."""
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) and type(module[0]).__name__.endswith("DecoderLayer"):
            return module
    raise ValueError("no ModuleList of *DecoderLayer blocks found")


def _project_out(hidden: torch.Tensor, direction: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    unit = direction.to(hidden.dtype)
    return hidden - scale * (hidden @ unit).unsqueeze(-1) * unit


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


@contextmanager
def residual_ablation(
    model: nn.Module, directions: dict[int, torch.Tensor], mask: torch.Tensor | None = None
) -> Iterator[None]:
    """Remove unit `directions[l]` from residual stream l (l >= 1) during forward passes.

    Args:
        model: Decoder-only model.
        directions: Residual index -> unit vector of size d.
        mask: Optional per-example weight, shape (batch,); 0 leaves that example untouched.
    """
    layers = decoder_layers(model)
    handles = []
    for index, direction in directions.items():
        if index < 1:
            raise ValueError("residual index 0 is the embedding output; ablate from index 1")

        def hook(_module: nn.Module, _inputs: Any, output: Any, direction: torch.Tensor = direction) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            scale = 1.0 if mask is None else mask.to(hidden.dtype).view(-1, *([1] * (hidden.dim() - 1)))
            return _replace_hidden(output, _project_out(hidden, direction.to(hidden.device), scale))

        handles.append(layers[index - 1].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def write_ablation(model: nn.Module, block: int, direction: torch.Tensor, alpha: float = 1.0) -> Iterator[None]:
    """Remove `alpha` times `direction` from what decoder block `block` writes to the residual stream.

    Equivalent to orthogonalising the block's attention output and MLP down projections,
    W <- (I - alpha r r^T) W, as in single-layer abliteration (Kuo et al. 2026; alpha = 2
    reflects the component instead of removing it).
    """
    layer = decoder_layers(model)[block]
    # Under PEFT the matching module is the LoRA wrapper, whose output includes the adapter.
    targets = [m for name, m in layer.named_modules() if name.endswith(("o_proj", "down_proj"))]

    def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        return _project_out(output, direction.to(output.device), alpha)

    handles = [module.register_forward_hook(hook) for module in targets]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def mean_difference_directions(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    harmful_prompts: Sequence[str],
    harmless_prompts: Sequence[str],
    layers: Sequence[int],
    max_length: int = 512,
) -> dict[int, torch.Tensor]:
    """Return unit differences of mean last-prompt-token residual streams (harmful minus harmless), per layer."""
    harmful = collect(model, tokenizer, harmful_prompts, layers, 1, max_length=max_length)[:, 0].float().mean(1)
    harmless = collect(model, tokenizer, harmless_prompts, layers, 1, max_length=max_length)[:, 0].float().mean(1)
    device = next(model.parameters()).device
    return {
        layer: torch.nn.functional.normalize(harmful[i] - harmless[i], dim=0).to(device)
        for i, layer in enumerate(layers)
    }
