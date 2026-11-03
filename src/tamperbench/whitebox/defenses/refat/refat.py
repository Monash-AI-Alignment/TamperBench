"""ReFAT: refusal training while the refusal feature is ablated (Yu et al., ICLR 2025, arXiv 2409.20089).

Loss (eq. 9-11): on refusal examples, cross-entropy of the refusal with the layerwise
refusal feature (difference of mean last-token activations, harmful minus harmless)
projected out of the residual stream with probability p_RFA, else without; plus plain
cross-entropy on utility examples. The feature is recomputed from the current model
every `refresh_every` steps.
"""

import json
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing_extensions import override

from tamperbench.whitebox.defenses.defense import AlignmentDefense
from tamperbench.whitebox.defenses.refat.config import ReFATConfig
from tamperbench.whitebox.defenses.registry import register_defense
from tamperbench.whitebox.utils.activations import Encoded, collate
from tamperbench.whitebox.utils.directions import mean_difference_directions, residual_ablation
from tamperbench.whitebox.utils.names import DefenseName
from tamperbench.whitebox.utils.ops import run_in_isolation
from tamperbench.whitebox.utils.training import (
    encode_conversations,
    lora_model,
    pad_token_id,
    read_jsonl,
    response_cross_entropy,
    save_model,
)

LOG_FILE = "refat_log.jsonl"


@register_defense(DefenseName.REFAT, ReFATConfig)
class ReFAT(AlignmentDefense[ReFATConfig]):
    """Refusal Feature Adversarial Training with LoRA."""

    @override
    def _run_defense(self) -> Path:
        return run_in_isolation(target=run_refat, args=(self.defense_config,), error_context="ReFAT training")


def train(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    refusals: list[tuple[Encoded, str]],
    utility: list[tuple[Encoded, str]],
    config: ReFATConfig,
    log_path: Path,
) -> None:
    """Run the ReFAT loop; each list pairs encoded conversations with their prompts."""
    device = next(model.parameters()).device
    rng = random.Random(config.seed)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    half = config.batch_size // 2
    pad = pad_token_id(tokenizer)
    steps_per_epoch = min(len(refusals), len(utility)) // half
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)
    directions: dict[int, torch.Tensor] = {}
    step = 0
    with log_path.open("w") as log:
        for _ in range(config.epochs):
            rng.shuffle(refusals)
            rng.shuffle(utility)
            for i in range(steps_per_epoch):
                if step % config.refresh_every == 0:
                    model.eval()
                    directions = mean_difference_directions(
                        model,
                        tokenizer,
                        [p for _, p in rng.sample(refusals, config.feature_samples)],
                        [p for _, p in rng.sample(utility, config.feature_samples)],
                        config.rfa_layers,
                        config.max_length,
                    )
                    model.train()
                refusal_batch = collate([e for e, _ in refusals[i * half : (i + 1) * half]], pad, 1)
                utility_batch = collate([e for e, _ in utility[i * half : (i + 1) * half]], pad, 1)
                ablate = (torch.rand(half, generator=generator, device=device) < config.rfa_probability).float()
                optimiser.zero_grad(set_to_none=True)
                # Backward runs inside the hook context: gradient checkpointing recomputes the
                # forward during backward and must see the same ablation.
                with residual_ablation(model, directions, mask=ablate):
                    refusal_ce = response_cross_entropy(
                        model(
                            input_ids=refusal_batch["input_ids"].to(device),
                            attention_mask=refusal_batch["attention_mask"].to(device),
                        ).logits,
                        refusal_batch["labels"].to(device),
                    )
                    refusal_ce.backward()
                utility_ce = response_cross_entropy(
                    model(
                        input_ids=utility_batch["input_ids"].to(device),
                        attention_mask=utility_batch["attention_mask"].to(device),
                    ).logits,
                    utility_batch["labels"].to(device),
                )
                utility_ce.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimiser.step()
                if step % config.log_interval == 0:
                    record = {
                        "step": step,
                        "refusal_ce": refusal_ce.item(),
                        "utility_ce": utility_ce.item(),
                        "ablated_fraction": ablate.mean().item(),
                        "grad_norm": float(grad_norm),
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(record, flush=True)
                step += 1


def run_refat(config: ReFATConfig) -> Path:
    """Train ReFAT, merge the LoRA weights and save the model with its log."""
    if not torch.cuda.is_available():
        raise RuntimeError("ReFAT training requires CUDA")
    torch.manual_seed(config.seed)
    source = str(config.input_checkpoint_path)
    output = Path(config.output_checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(source)

    def load(path: str) -> list[tuple[Encoded, str]]:
        return encode_conversations(
            tokenizer, [(r["prompt"], r["response"]) for r in read_jsonl(path)], config.max_length
        )

    refusals, utility = load(config.refusal_data_path), load(config.utility_data_path)
    print(f"{len(refusals)} refusal and {len(utility)} utility conversations", flush=True)
    model = lora_model(source, config.lora_r, config.lora_alpha, config.lora_dropout)
    output.mkdir(parents=True, exist_ok=True)
    train(model, tokenizer, refusals, utility, config, output / LOG_FILE)
    save_model(model, source, output)
    return output
