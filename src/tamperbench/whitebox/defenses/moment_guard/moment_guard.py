"""Moment-guarded refusal training.

Refusal is trained into a model while the first m-1 moments of its post-instruction
activations on harmful and harmless prompts are matched across a band of layers, so the
statistics that difference-in-means abliteration (m >= 2) or covariance-based attacks
(m >= 3) use to locate refusal carry no signal there. See `MomentGuardConfig` for the
objective and `moments.py` for the estimator.
"""

import gc
import json
import math
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from typing_extensions import override

from tamperbench.whitebox.defenses.defense import AlignmentDefense
from tamperbench.whitebox.defenses.moment_guard.activations import (
    Encoded,
    collate,
    collect,
    encode,
    gather,
    post_instruction_length,
)
from tamperbench.whitebox.defenses.moment_guard.config import MomentGuardConfig
from tamperbench.whitebox.defenses.moment_guard.moments import MomentBank, moment_penalty
from tamperbench.whitebox.defenses.registry import register_defense
from tamperbench.whitebox.utils.names import DefenseName
from tamperbench.whitebox.utils.ops import run_in_isolation

LOG_FILE = "moment_guard_log.jsonl"
CALIBRATION_FILE = "moment_guard_calibration.pt"


@register_defense(DefenseName.MOMENT_GUARD, MomentGuardConfig)
class MomentGuard(AlignmentDefense[MomentGuardConfig]):
    """Refusal training with a polynomial-kernel moment-matching penalty, via LoRA."""

    @override
    def _run_defense(self) -> Path:
        return run_in_isolation(
            target=run_moment_guard,
            args=(self.defense_config,),
            error_context="moment-guard defense training",
        )


def _read_pairs(path: str) -> list[tuple[str, str]]:
    with open(path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [(row["prompt"], row["response"]) for row in rows]


def _encode_all(
    tokenizer: PreTrainedTokenizerBase, pairs: list[tuple[str, str]], max_length: int, num_positions: int
) -> list[Encoded]:
    encoded = [encode(tokenizer, prompt, response, max_length) for prompt, response in pairs]
    kept = [
        e for e in encoded if e is not None and e.prompt_length >= num_positions and len(e.input_ids) > e.prompt_length
    ]
    if not kept:
        raise ValueError("no conversation fits max_length with a non-empty response")
    return kept


def calibrate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: tuple[list[str], list[str]],
    config: MomentGuardConfig,
    num_positions: int,
) -> dict[str, torch.Tensor | None]:
    """Per-layer scale (mean activation norm) and degree->=2 subspace, from the input model.

    Activations are divided by the scale so kernel values are O(1) whatever the layer's norm.
    The subspace is the top `subspace_rank` principal components of the pooled, scaled
    activations of both classes.
    """
    layers = list(config.moment_layers)
    activations = torch.cat(
        [collect(model, tokenizer, p, layers, num_positions, max_length=config.max_length) for p in prompts], dim=2
    ).float()
    scales = activations.norm(dim=-1).mean(dim=(1, 2))
    bases = None
    if config.subspace_rank:
        pooled = (activations / scales[:, None, None, None]).flatten(1, 2)
        centred = pooled - pooled.mean(dim=1, keepdim=True)
        bases = torch.stack(
            [torch.linalg.svd(layer, full_matrices=False).Vh[: config.subspace_rank].T for layer in centred]
        )
    return {"scales": scales, "bases": bases}


def train(
    model: PeftModel,
    refusals: list[Encoded],
    harmless: list[Encoded],
    config: MomentGuardConfig,
    calibration: dict[str, torch.Tensor | None],
    pad_token_id: int,
    num_positions: int,
    log_path: Path,
) -> None:
    """Optimise refusal CE + KL to the input model + lambda * D_m with AdamW on the LoRA weights."""
    device = next(model.parameters()).device
    layers = list(config.moment_layers)
    scales = cast(torch.Tensor, calibration["scales"]).to(device)[:, None, None, None]
    bases = calibration["bases"]
    bases = None if bases is None else bases.to(device)
    generator = torch.Generator().manual_seed(config.seed)
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)
    warmup = max(1, config.warmup_steps)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: min(1.0, (step + 1) / warmup)
        * 0.5
        * (1 + math.cos(math.pi * min(1.0, step / max(1, config.num_steps - 1)))),
    )
    bank = MomentBank(config.bank_size) if config.bank_size else None
    model.train()

    def batch(data: list[Encoded]) -> dict[str, torch.Tensor]:
        rows = torch.randint(len(data), (config.batch_size,), generator=generator).tolist()
        return {k: v.to(device) for k, v in collate([data[i] for i in rows], pad_token_id, num_positions).items()}

    with log_path.open("w") as log:
        for step in range(config.num_steps):
            harmful_batch, harmless_batch = batch(refusals), batch(harmless)

            harmful_out = model(
                input_ids=harmful_batch["input_ids"],
                attention_mask=harmful_batch["attention_mask"],
                output_hidden_states=True,
            )
            refusal_ce = F.cross_entropy(
                harmful_out.logits[:, :-1].flatten(0, 1).float(),
                harmful_batch["labels"][:, 1:].flatten(),
                ignore_index=-100,
            )

            harmless_out = model(
                input_ids=harmless_batch["input_ids"],
                attention_mask=harmless_batch["attention_mask"],
                output_hidden_states=True,
            )
            with torch.no_grad(), model.disable_adapter():
                reference_logits = model(
                    input_ids=harmless_batch["input_ids"], attention_mask=harmless_batch["attention_mask"]
                ).logits
            mask = harmless_batch["response_logit_mask"]
            kl = F.kl_div(
                F.log_softmax(harmless_out.logits[mask].float(), dim=-1),
                F.log_softmax(reference_logits[mask].float(), dim=-1),
                log_target=True,
                reduction="batchmean",
            )

            harmful_acts = gather(harmful_out.hidden_states, layers, harmful_batch["positions"]).float() / scales
            harmless_acts = gather(harmless_out.hidden_states, layers, harmless_batch["positions"]).float() / scales
            with torch.set_grad_enabled(config.moment_weight > 0):
                penalty, parts = moment_penalty(
                    harmful_acts,
                    harmless_acts,
                    config.m,
                    bases,
                    None if bank is None else bank.harmful,
                    None if bank is None else bank.harmless,
                    whiten_shrinkage=config.whiten_shrinkage,
                )

            loss = refusal_ce + config.kl_weight * kl
            if config.moment_weight > 0:
                loss = loss + config.moment_weight * penalty
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimiser.step()
            schedule.step()
            if bank is not None:
                bank.push(harmful_acts, harmless_acts)

            if step % config.log_interval == 0 or step == config.num_steps - 1:
                record = {
                    "step": step,
                    "loss": loss.item(),
                    "refusal_ce": refusal_ce.item(),
                    "kl": kl.item(),
                    "moment_penalty": penalty.item(),
                    **{f"mmd2_degree_{degree}": value.item() for degree, value in parts.items()},
                    "grad_norm": float(grad_norm),
                    "learning_rate": schedule.get_last_lr()[0],
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(record, flush=True)


def run_moment_guard(config: MomentGuardConfig) -> Path:
    """Train, merge the LoRA weights and save the defended model with its training log."""
    output = Path(config.output_checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError("moment-guard training requires CUDA")
    torch.manual_seed(config.seed)

    source = str(config.input_checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(source)
    pad_token_id = cast(int, tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    num_positions = config.num_positions or post_instruction_length(tokenizer)
    refusals = _encode_all(tokenizer, _read_pairs(config.refusal_data_path), config.max_length, num_positions)
    harmless = _encode_all(tokenizer, _read_pairs(config.harmless_data_path), config.max_length, num_positions)
    print(f"{len(refusals)} refusal and {len(harmless)} harmless conversations; {num_positions} positions", flush=True)

    base = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.bfloat16, device_map={"": 0})
    base.eval()
    calibration = calibrate(
        base,
        tokenizer,
        (
            [p for p, _ in _read_pairs(config.refusal_data_path)[: config.calibration_samples]],
            [p for p, _ in _read_pairs(config.harmless_data_path)[: config.calibration_samples]],
        ),
        config,
        num_positions,
    )

    model = cast(
        PeftModel,
        get_peft_model(
            base,
            LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.lora_target_modules),
                bias="none",
            ),
        ),
    )
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # pyright: ignore[reportCallIssue]

    output.mkdir(parents=True, exist_ok=True)
    train(model, refusals, harmless, config, calibration, pad_token_id, num_positions, output / LOG_FILE)

    merged = cast(PreTrainedModel, model.merge_and_unload())  # pyright: ignore[reportCallIssue]
    merged.save_pretrained(str(output))
    AutoTokenizer.from_pretrained(source).save_pretrained(str(output))
    torch.save(
        {**calibration, "layers": list(config.moment_layers), "num_positions": num_positions}, output / CALIBRATION_FILE
    )

    del model, merged, base
    gc.collect()
    torch.cuda.empty_cache()
    return output
