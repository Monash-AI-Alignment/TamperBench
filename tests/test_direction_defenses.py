"""Ablation hooks used by ReFAT and ART, and the Circuit Breakers rerouting loss."""

from typing import cast

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from tamperbench.whitebox.defenses.circuit_breakers.circuit_breakers import rerouting_losses
from tamperbench.whitebox.utils.directions import decoder_layers, residual_ablation, write_ablation


def _tiny_llama(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=50,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    return LlamaForCausalLM(config).eval()


def _unit(d: int, seed: int) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.randn(d, generator=torch.Generator().manual_seed(seed)), dim=0)


def test_residual_ablation_removes_the_direction_only_for_masked_examples() -> None:
    """ReFAT trains some examples under ablation: those rows lose the direction, others are untouched.

    Checked on what block 2 receives: transformers records `hidden_states` with its own
    hooks, before ours rewrite the block output, so the recorded states are not ablated.
    """
    model = _tiny_llama()
    ids = torch.randint(0, 50, (2, 5), generator=torch.Generator().manual_seed(1))
    direction = _unit(16, 2)
    received: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, args: tuple, kwargs: dict) -> None:  # pyright: ignore[reportMissingTypeArgument]
        received.append(kwargs["hidden_states"] if "hidden_states" in kwargs else args[0])

    handle = decoder_layers(model)[2].register_forward_pre_hook(capture, with_kwargs=True)
    with torch.no_grad():
        plain = model(input_ids=ids).logits
        with residual_ablation(model, {2: direction}, mask=torch.tensor([1.0, 0.0])):
            ablated = model(input_ids=ids).logits
    handle.remove()
    before, after = received
    assert after[0].matmul(direction).abs().max() < 1e-5
    assert before[0].matmul(direction).abs().max() > 1e-3
    torch.testing.assert_close(after[1], before[1])
    torch.testing.assert_close(ablated[1], plain[1])
    assert not torch.allclose(ablated[0], plain[0])


def test_write_ablation_equals_orthogonalised_weights() -> None:
    """The hook matches editing o_proj and down_proj to (I - alpha r r^T) W, including alpha = 2."""
    for alpha in (1.0, 2.0):
        model, edited = _tiny_llama(), _tiny_llama()
        direction = _unit(16, 3)
        projector = torch.eye(16) - alpha * torch.outer(direction, direction)
        block = cast(LlamaDecoderLayer, decoder_layers(edited)[1])
        with torch.no_grad():
            block.self_attn.o_proj.weight.copy_(projector @ block.self_attn.o_proj.weight)
            block.mlp.down_proj.weight.copy_(projector @ block.mlp.down_proj.weight)
            ids = torch.randint(0, 50, (2, 6), generator=torch.Generator().manual_seed(4))
            with write_ablation(model, 1, direction, alpha):
                hooked = model(input_ids=ids).logits
            torch.testing.assert_close(hooked, edited(input_ids=ids).logits)
            assert not torch.allclose(hooked, model(input_ids=ids).logits)


def test_rerouting_loss_is_one_when_unchanged_and_zero_when_orthogonal() -> None:
    """Circuit breakers start at cosine 1 (loss 1) and reach 0 once harmful states are orthogonal."""
    generator = torch.Generator().manual_seed(5)
    states = torch.randn(2, 3, 4, 8, generator=generator)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
    retain, rerouting = rerouting_losses(states, states, mask, states, states, mask)
    assert retain == 0
    torch.testing.assert_close(rerouting, torch.tensor(1.0))

    a = torch.zeros(1, 1, 2, 4)
    a[..., 0] = 1.0
    b = torch.zeros(1, 1, 2, 4)
    b[..., 1] = 1.0
    _, orthogonal = rerouting_losses(a, a, torch.ones(1, 2), a, b, torch.ones(1, 2))
    assert orthogonal == 0
