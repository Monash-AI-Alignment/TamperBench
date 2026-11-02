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
def test_estimator_equals_explicit_moment_tensor_formula(degree: int) -> None:
    """The kernel trick reproduces the unbiased estimate built from explicit moment tensors."""
    generator = torch.Generator().manual_seed(degree)
    x = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    y = torch.randn(5, 3, generator=generator, dtype=torch.float64) + 0.5
    tx, ty = _tensor_power(x, degree), _tensor_power(y, degree)
    sx, sy = tx.sum(0), ty.sum(0)
    xx = (sx @ sx - (tx * tx).sum()) / (6 * 5)
    yy = (sy @ sy - (ty * ty).sum()) / (5 * 4)
    xy = (sx @ sy) / (6 * 5)
    torch.testing.assert_close(polynomial_mmd2(x, y, degree), xx + yy - 2 * xy)


def test_estimator_is_not_inflated_by_within_class_spread() -> None:
    """Equal means, wide spread: the plug-in estimate is about tr(Sigma)/n, the unbiased one about 0.

    A plug-in penalty would push the model to shrink each class instead of matching them.
    """
    generator = torch.Generator().manual_seed(0)
    unbiased, plug_in = [], []
    for _ in range(300):
        x = 3.0 * torch.randn(16, 8, generator=generator, dtype=torch.float64)
        y = torch.randn(16, 8, generator=generator, dtype=torch.float64)
        unbiased.append(polynomial_mmd2(x, y, 1))
        plug_in.append((x.mean(0) - y.mean(0)).square().sum())
    assert torch.stack(plug_in).mean() > 4.0  # (9 + 1) * 8 / 16 = 5
    assert torch.stack(unbiased).mean().abs() < 0.3


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
    value = polynomial_mmd2(x, y, 2, x_old, y_old)
    torch.testing.assert_close(value, polynomial_mmd2(torch.cat([x, x_old]), torch.cat([y, y_old]), 2))
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
