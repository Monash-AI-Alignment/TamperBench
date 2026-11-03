"""Circuit Breakers via Representation Rerouting (Zou et al., 2024, arXiv 2406.04313).

Port of `compute_loss` in GraySwanAI/circuit-breakers `src/lorra_circuit_breaker.py`:

    loss = c_retain * mean ||h_lora - h_orig||_2   over retain tokens, layers 0..max(target)
         + c_cb     * sum ReLU(cos(h_lora, h_orig)) / #tokens   over harmful tokens, target layers

with c_retain = alpha * t / 2T and c_cb = alpha * (1 - t / 2T). The original
representations come from the same model with the LoRA adapters disabled. As in the
original, padded positions are zero in both retain tensors and still count in the mean.
"""

import json
import random
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch import nn
from transformers import AutoTokenizer
from typing_extensions import override

from tamperbench.whitebox.defenses.circuit_breakers.config import CircuitBreakersConfig
from tamperbench.whitebox.defenses.defense import AlignmentDefense
from tamperbench.whitebox.defenses.registry import register_defense
from tamperbench.whitebox.utils.activations import Encoded, collate
from tamperbench.whitebox.utils.names import DefenseName
from tamperbench.whitebox.utils.ops import run_in_isolation
from tamperbench.whitebox.utils.training import encode_conversations, lora_model, pad_token_id, read_jsonl, save_model

LOG_FILE = "circuit_breakers_log.jsonl"


@register_defense(DefenseName.CIRCUIT_BREAKERS, CircuitBreakersConfig)
class CircuitBreakers(AlignmentDefense[CircuitBreakersConfig]):
    """Representation Rerouting with LoRA."""

    @override
    def _run_defense(self) -> Path:
        return run_in_isolation(
            target=run_circuit_breakers, args=(self.defense_config,), error_context="Circuit Breakers training"
        )


def rerouting_losses(
    lora_retain: torch.Tensor,
    orig_retain: torch.Tensor,
    retain_mask: torch.Tensor,
    lora_harmful: torch.Tensor,
    orig_harmful: torch.Tensor,
    harmful_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain and rerouting losses from stacked hidden states (layers, batch, tokens, d) and masks (batch, tokens)."""
    retain_weight = retain_mask[None, :, :, None].to(lora_retain.dtype)
    retain = torch.norm(
        lora_retain * retain_weight - orig_retain * retain_weight, dim=-1, p=2, dtype=torch.float
    ).nanmean()
    harmful_weight = harmful_mask[None, :, :].expand(lora_harmful.shape[0], -1, -1).float()
    cosine = (F.normalize(lora_harmful.float(), dim=-1) * F.normalize(orig_harmful.float(), dim=-1)).sum(-1)
    rerouting = (torch.relu(cosine) * harmful_weight).sum() / harmful_weight.sum()
    return retain, rerouting


def train(
    model: PeftModel,
    retain: list[Encoded],
    harmful: list[Encoded],
    config: CircuitBreakersConfig,
    pad: int,
    log_path: Path,
) -> None:
    """Run the rerouting loop for `max_steps` steps."""
    device = next(model.parameters()).device
    rng = random.Random(config.seed)
    # Residual streams 0 .. max(target) + 1: the original trains a copy truncated after the
    # last target block, so its retain loss covers every hidden state that copy produces.
    retain_count = max(config.target_layers) + 2
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate, weight_decay=0.0
    )
    with log_path.open("w") as log:
        for step in range(config.max_steps):
            progress = step / (2 * config.max_steps)
            retain_coeff, rerouting_coeff = config.alpha * progress, config.alpha * (1 - progress)
            retain_batch = collate(rng.sample(retain, config.batch_size), pad, 1)
            harmful_batch = collate(rng.sample(harmful, config.batch_size), pad, 1)
            retain_inputs = {
                "input_ids": retain_batch["input_ids"].to(device),
                "attention_mask": retain_batch["attention_mask"].to(device),
            }
            harmful_inputs = {
                "input_ids": harmful_batch["input_ids"].to(device),
                "attention_mask": harmful_batch["attention_mask"].to(device),
            }
            # The decoder stack without the LM head: the loss only reads hidden states.
            backbone = cast(nn.Module, model.get_base_model().model)
            with torch.no_grad(), model.disable_adapter():
                model.eval()
                orig_retain = torch.stack(
                    backbone(**retain_inputs, output_hidden_states=True).hidden_states[:retain_count]
                )
                orig_states = backbone(**harmful_inputs, output_hidden_states=True).hidden_states
                orig_harmful = torch.stack([orig_states[i] for i in config.target_layers])
            model.train()
            lora_retain = torch.stack(backbone(**retain_inputs, output_hidden_states=True).hidden_states[:retain_count])
            harmful_states = backbone(**harmful_inputs, output_hidden_states=True).hidden_states
            lora_harmful = torch.stack([harmful_states[i] for i in config.target_layers])
            retain_loss, rerouting_loss = rerouting_losses(
                lora_retain,
                orig_retain,
                retain_inputs["attention_mask"],
                lora_harmful,
                orig_harmful,
                harmful_inputs["attention_mask"],
            )
            loss = retain_coeff * retain_loss + rerouting_coeff * rerouting_loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            if step % config.log_interval == 0 or step == config.max_steps - 1:
                record = {
                    "step": step,
                    "retain_loss": retain_loss.item(),
                    "rerouting_loss": rerouting_loss.item(),
                    "retain_coeff": retain_coeff,
                    "rerouting_coeff": rerouting_coeff,
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(record, flush=True)


def run_circuit_breakers(config: CircuitBreakersConfig) -> Path:
    """Train Circuit Breakers, merge the LoRA weights and save the model with its log."""
    if not torch.cuda.is_available():
        raise RuntimeError("Circuit Breakers training requires CUDA")
    torch.manual_seed(config.seed)
    source = str(config.input_checkpoint_path)
    output = Path(config.output_checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(source)

    def load(path: str, max_length: int) -> list[Encoded]:
        rows = [(r["prompt"], r["response"]) for r in read_jsonl(path)]
        return [e for e, _ in encode_conversations(tokenizer, rows, max_length)]

    retain = load(config.retain_data_path, config.retain_max_length)
    harmful = load(config.harmful_data_path, config.harmful_max_length)
    print(f"{len(retain)} retain and {len(harmful)} harmful conversations", flush=True)
    model = lora_model(
        source,
        config.lora_r,
        config.lora_alpha,
        config.lora_dropout,
        layers_to_transform=range(max(config.target_layers) + 1),
    )
    output.mkdir(parents=True, exist_ok=True)
    train(model, retain, harmful, config, pad_token_id(tokenizer), output / LOG_FILE)
    save_model(model, source, output)
    return output
