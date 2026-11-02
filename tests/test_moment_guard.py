"""Behaviour of the moment-guard defense: the kernel estimator, its bias, what each m blinds, and token alignment."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from tamperbench.whitebox.defenses.moment_guard.activations import collate, encode, post_instruction_length
from tamperbench.whitebox.defenses.moment_guard.moments import MomentBank, moment_penalty, polynomial_mmd2


def _tensor_power(x: torch.Tensor, degree: int) -> torch.Tensor:
    """Each row's j-fold outer product x^{(x)j}, flattened."""
    out = x
    for _ in range(degree - 1):
        out = torch.einsum("na,nb->nab", out, x).flatten(1)
    return out


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_estimators_equal_explicit_moment_tensor_formulas(degree: int) -> None:
    """The kernel trick reproduces both estimates built from explicit moment tensors."""
    generator = torch.Generator().manual_seed(degree)
    x = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    y = torch.randn(5, 3, generator=generator, dtype=torch.float64) + 0.5
    tx, ty = _tensor_power(x, degree), _tensor_power(y, degree)
    sx, sy = tx.sum(0), ty.sum(0)
    xx = (sx @ sx - (tx * tx).sum()) / (6 * 5)
    yy = (sy @ sy - (ty * ty).sum()) / (5 * 4)
    xy = (sx @ sy) / (6 * 5)
    torch.testing.assert_close(polynomial_mmd2(x, y, degree, unbiased=True), xx + yy - 2 * xy)
    torch.testing.assert_close(polynomial_mmd2(x, y, degree, unbiased=False), (sx / 6 - sy / 5).square().sum())


def test_unbiased_estimate_ignores_spread_but_training_penalty_stays_bounded() -> None:
    """Unbiased: ~0 for equal means however wide the spread. Training penalty: never below 0.

    Spreading a class (x1 = -x2, large norm) drives the unbiased estimate to minus
    infinity, which a model being trained will find (the first lambda = 10 smoke runs did).
    """
    generator = torch.Generator().manual_seed(0)
    unbiased = [
        polynomial_mmd2(
            3.0 * torch.randn(16, 8, generator=generator, dtype=torch.float64),
            torch.randn(16, 8, generator=generator, dtype=torch.float64),
            1,
            unbiased=True,
        )
        for _ in range(300)
    ]
    assert torch.stack(unbiased).mean().abs() < 0.3

    direction = torch.zeros(4, dtype=torch.float64)
    direction[0] = 1e3
    spread = torch.stack([direction, -direction])
    harmless = torch.randn(2, 4, generator=generator, dtype=torch.float64)
    assert polynomial_mmd2(spread, harmless, 1, unbiased=True) < -1e5
    penalty, _ = moment_penalty(spread.view(1, 1, 2, 4), harmless.view(1, 1, 2, 4), m=3)
    assert penalty >= 0


def test_m_selects_which_moments_must_match() -> None:
    """Harmful at +-a*r, harmless at 0: means agree (m=2 sees nothing), variances differ (m=3 does)."""
    generator = torch.Generator().manual_seed(1)
    n, d = 400, 6
    r = torch.zeros(d, dtype=torch.float64)
    r[0] = 1.0
    signs = torch.randint(0, 2, (n,), generator=generator).double() * 2 - 1
    noise = 0.1 * torch.randn(2, n, d, generator=generator, dtype=torch.float64)
    harmful = (3.0 * signs[:, None] * r + noise[0]).view(1, 1, n, d)
    harmless = noise[1].view(1, 1, n, d)
    first, _ = moment_penalty(harmful, harmless, m=2)
    second, parts = moment_penalty(harmful, harmless, m=3)
    assert first.abs() < 0.2
    assert parts[2] > 50 * first.abs().clamp_min(1e-3)
    torch.testing.assert_close(second, parts[1] + parts[2])


def test_bank_samples_enter_the_estimate_but_receive_no_gradient() -> None:
    """Banked activations widen the estimate; gradients flow only to the current batch."""
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(4, 3, generator=generator, requires_grad=True)
    y = torch.randn(4, 3, generator=generator)
    x_old = torch.randn(8, 3, generator=generator, requires_grad=True)
    y_old = torch.randn(8, 3, generator=generator)
    value = polynomial_mmd2(x, y, 2, x_old, y_old, unbiased=False)
    torch.testing.assert_close(value, polynomial_mmd2(torch.cat([x, x_old]), torch.cat([y, y_old]), 2, unbiased=False))
    value.backward()
    assert x.grad is not None and x_old.grad is None

    bank = MomentBank(size=5)
    for step in range(3):
        bank.push(torch.full((1, 1, 2, 3), float(step)), torch.full((1, 1, 2, 3), float(step)))
    assert bank.harmful is not None and bank.harmful.shape[2] == 5
    assert bank.harmful[0, 0, :, 0].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0][-5:]


def test_batch_marks_post_instruction_positions_and_response_targets() -> None:
    """Positions are the template tokens after the user turn; CE and KL masks cover exactly the response."""
    words = ["[UNK]", "<", ">", "</", "user", "assistant", "a", "b", "c", "d", "e"]
    backend = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    template = (
        "{% for m in messages %}<{{ m['role'] }}> {{ m['content'] }} </{{ m['role'] }}> {% endfor %}"
        "{% if add_generation_prompt %}<assistant> {% endif %}"
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", chat_template=template)
    ids = {w: i for i, w in enumerate(words)}

    assert post_instruction_length(tokenizer) == 6  # "</ user > < assistant >"
    short = encode(tokenizer, "a", "c d", max_length=64)
    long = encode(tokenizer, "a b", "e", max_length=64)
    assert short is not None and long is not None
    batch = collate([short, long], pad_token_id=0, num_positions=6)

    for row, example in enumerate([short, long]):
        prompt_tail = example.input_ids[example.prompt_length - 6 : example.prompt_length]
        assert batch["input_ids"][row, batch["positions"][row]].tolist() == prompt_tail
        assert prompt_tail[-3:] == [ids["<"], ids["assistant"], ids[">"]]
        targets = batch["labels"][row][batch["labels"][row] != -100].tolist()
        assert targets == example.input_ids[example.prompt_length :]
        predicts_response = batch["labels"][row, 1:] != -100
        assert torch.equal(batch["response_logit_mask"][row, :-1], predicts_response)
    assert encode(tokenizer, "a b c d e", None, max_length=8) is None
