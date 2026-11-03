"""Configuration for Circuit Breakers / Representation Rerouting (Zou et al., 2024)."""

from dataclasses import dataclass, field

from tamperbench.whitebox.defenses.defense import AlignmentDefenseConfig


@dataclass
class CircuitBreakersConfig(AlignmentDefenseConfig):
    """Representation Rerouting with LoRA.

    Defaults follow the authors' Llama-3-8B script (GraySwanAI/circuit-breakers,
    `scripts/lorra_circuit_breaker_llama3_8b.sh`): LoRA r16 / alpha 16 / dropout 0.05 on
    layers up to the last target layer, target layers 10 and 20, alpha 10, 150 steps,
    batch 16, lr 1e-4 constant; retain sequences up to 1024 tokens, circuit-breaker
    sequences up to 512.

    Attributes:
        harmful_data_path: JSONL {"prompt", "response"}: harmful requests with harmful completions.
        retain_data_path: JSONL {"prompt", "response"}: benign conversations and refusals to keep.
        target_layers: Residual-stream (`hidden_states`) indices where harmful
            representations are rerouted.
        alpha: Scale of both loss coefficients.
        max_steps: Optimiser steps; also sets the coefficient schedule (t / 2T).
        batch_size: Retain and harmful examples per step (each).
        learning_rate: AdamW learning rate (constant).
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
        retain_max_length: Maximum tokens per retain conversation.
        harmful_max_length: Maximum tokens per harmful conversation.
        seed: Seed for data order and LoRA initialisation.
        log_interval: Steps between log records.
    """

    harmful_data_path: str = ""
    retain_data_path: str = ""
    target_layers: tuple[int, ...] = field(default_factory=lambda: (10, 20))
    alpha: float = 10.0
    max_steps: int = 150
    batch_size: int = 16
    learning_rate: float = 1e-4
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    retain_max_length: int = 1024
    harmful_max_length: int = 512
    seed: int = 0
    log_interval: int = 10

    def __post_init__(self) -> None:
        """Require the data files."""
        super().__post_init__()
        if not self.harmful_data_path or not self.retain_data_path:
            raise ValueError("harmful_data_path and retain_data_path are required")
