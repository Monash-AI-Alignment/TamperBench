"""ART: abliteration-resistant tuning (Kuo et al., 2026, arXiv 2605.26526, Algorithms 2-3).

Every `refresh_every` steps, ABLITERATE: for each decoder block l, take the unit
difference of mean residual streams entering block l over the batch's harmful (y-) and
safe (y+) responses, ablate it from what block l writes, and keep the block whose
ablation lowers the harmful cross-entropy most. Each step then minimises

    gamma_ar * -log(CE(y- | abliterated model) + epsilon) + gamma_ret * CE(y+ | model)

so the abliterated model is pushed away from the harmful continuation while the model
keeps its safe answers. The abliteration is a forward hook, so the gradient of the first
term flows through the projected weights (the paper's grad wrt the abliterated
parameters, applied to the parameters).
"""

import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedModel
from typing_extensions import override

from tamperbench.whitebox.defenses.art.config import ARTConfig
from tamperbench.whitebox.defenses.defense import AlignmentDefense
from tamperbench.whitebox.defenses.registry import register_defense
from tamperbench.whitebox.utils.activations import Encoded, collate, encode
from tamperbench.whitebox.utils.directions import decoder_layers, write_ablation
from tamperbench.whitebox.utils.names import DefenseName
from tamperbench.whitebox.utils.ops import run_in_isolation
from tamperbench.whitebox.utils.training import load_model, pad_token_id, read_jsonl, response_cross_entropy, save_model

LOG_FILE = "art_log.jsonl"


@register_defense(DefenseName.ART, ARTConfig)
class ART(AlignmentDefense[ARTConfig]):
    """Abliteration-resistant tuning, full-parameter."""

    @override
    def _run_defense(self) -> Path:
        return run_in_isolation(target=run_art, args=(self.defense_config,), error_context="ART training")


def _forward(model: PreTrainedModel, batch: dict[str, torch.Tensor], hidden: bool = False):
    device = next(model.parameters()).device
    return model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        output_hidden_states=hidden,
    )


@torch.no_grad()
def select_ablation(
    model: PreTrainedModel, harmful: dict[str, torch.Tensor], safe: dict[str, torch.Tensor], alpha: float
) -> tuple[int, torch.Tensor, float]:
    """ABLITERATE (Alg. 3): the block and direction whose ablation most lowers CE on y-.

    Returns:
        (block index, unit direction, CE reduction).
    """
    device = next(model.parameters()).device
    harmful_out = _forward(model, harmful, hidden=True)
    safe_out = _forward(model, safe, hidden=True)
    baseline = response_cross_entropy(harmful_out.logits, harmful["labels"].to(device))
    harmful_mask = (harmful["labels"] != -100).to(device).unsqueeze(-1).float()
    safe_mask = (safe["labels"] != -100).to(device).unsqueeze(-1).float()
    best = (0, torch.empty(0), float("-inf"))
    for block in range(len(decoder_layers(model))):
        h_harm = (harmful_out.hidden_states[block].float() * harmful_mask).sum((0, 1)) / harmful_mask.sum()
        h_safe = (safe_out.hidden_states[block].float() * safe_mask).sum((0, 1)) / safe_mask.sum()
        direction = torch.nn.functional.normalize(h_harm - h_safe, dim=0)
        with write_ablation(model, block, direction, alpha):
            ablated = response_cross_entropy(_forward(model, harmful).logits, harmful["labels"].to(device))
        gain = float(baseline - ablated)
        if gain > best[2]:
            best = (block, direction, gain)
    return best


def train(
    model: PreTrainedModel, pairs: list[tuple[Encoded, Encoded]], config: ARTConfig, pad: int, log_path: Path
) -> None:
    """Run ART for `num_steps` steps on (safe, harmful) encoded pairs."""
    device = next(model.parameters()).device
    rng = random.Random(config.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    if config.eight_bit_optimizer:
        import bitsandbytes as bnb  # pyright: ignore[reportMissingImports]

        optimiser = bnb.optim.PagedAdamW8bit(params, lr=config.learning_rate)
    else:
        optimiser = torch.optim.AdamW(params, lr=config.learning_rate)
    block, direction, gain = 0, torch.empty(0), 0.0
    with log_path.open("w") as log:
        for step in range(config.num_steps):
            chosen = rng.sample(pairs, config.batch_size)
            safe = collate([s for s, _ in chosen], pad, 1)
            harmful = collate([h for _, h in chosen], pad, 1)
            if step % config.refresh_every == 0:
                model.eval()
                block, direction, gain = select_ablation(model, harmful, safe, config.ablation_alpha)
                model.train()
            optimiser.zero_grad(set_to_none=True)
            # Backward runs inside the hook context: gradient checkpointing recomputes the
            # forward during backward and must see the same ablation.
            with write_ablation(model, block, direction, config.ablation_alpha):
                harmful_ce = response_cross_entropy(_forward(model, harmful).logits, harmful["labels"].to(device))
                (config.gamma_ar * -torch.log(harmful_ce + config.epsilon)).backward()
            safe_ce = response_cross_entropy(_forward(model, safe).logits, safe["labels"].to(device))
            (config.gamma_ret * safe_ce).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimiser.step()
            if step % config.log_interval == 0 or step == config.num_steps - 1:
                record = {
                    "step": step,
                    "ablated_block": block,
                    "selection_gain": gain,
                    "harmful_ce_ablated": harmful_ce.item(),
                    "safe_ce": safe_ce.item(),
                    "grad_norm": float(grad_norm),
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(record, flush=True)


def run_art(config: ARTConfig) -> Path:
    """Train ART (full parameters) and save the model with its log."""
    if not torch.cuda.is_available():
        raise RuntimeError("ART training requires CUDA")
    torch.manual_seed(config.seed)
    source = str(config.input_checkpoint_path)
    output = Path(config.output_checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(source)
    pairs = []
    for row in read_jsonl(config.pairs_data_path):
        safe = encode(tokenizer, row["prompt"], row["chosen"], config.max_length)
        harmful = encode(tokenizer, row["prompt"], row["rejected"], config.max_length)
        if (
            safe
            and harmful
            and len(safe.input_ids) > safe.prompt_length
            and len(harmful.input_ids) > harmful.prompt_length
        ):
            pairs.append((safe, harmful))
    print(f"{len(pairs)} preference pairs", flush=True)
    model = load_model(source)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    output.mkdir(parents=True, exist_ok=True)
    train(model, pairs, config, pad_token_id(tokenizer), output / LOG_FILE)
    save_model(model, source, output)
    return output
