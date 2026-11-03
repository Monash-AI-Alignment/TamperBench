"""Configuration for ReFAT (Yu et al., ICLR 2025)."""

from dataclasses import dataclass, field

from tamperbench.whitebox.defenses.defense import AlignmentDefenseConfig


@dataclass
class ReFATConfig(AlignmentDefenseConfig):
    """Refusal Feature Adversarial Training.

    Defaults are the paper's Llama-3-8B-Instruct settings (Table 4 and section 5): LoRA
    r128 / alpha 32 on all linear layers, lr 2e-5, batch 32, 1 epoch, max length 512;
    refusal features recomputed every 4 steps from 32 + 32 instructions; ablation at
    residual layers 8-32 on refusal examples with probability 0.5.

    Attributes:
        refusal_data_path: JSONL {"prompt", "response"}: harmful requests with refusals.
        utility_data_path: JSONL {"prompt", "response"}: harmless requests with answers.
        rfa_layers: Residual-stream (`hidden_states`) indices where the refusal feature is ablated.
        rfa_probability: Chance each refusal example is trained under ablation (p_RFA).
        refresh_every: Steps between recomputing the refusal features.
        feature_samples: Harmful and harmless instructions used per recomputation.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
        learning_rate: AdamW learning rate (constant).
        batch_size: Examples per step, half refusal and half utility.
        epochs: Passes over the paired data.
        max_length: Maximum tokens per conversation.
        seed: Seed for data order, ablation draws and LoRA initialisation.
        log_interval: Steps between log records.
    """

    refusal_data_path: str = ""
    utility_data_path: str = ""
    rfa_layers: tuple[int, ...] = field(default_factory=lambda: tuple(range(8, 33)))
    rfa_probability: float = 0.5
    refresh_every: int = 4
    feature_samples: int = 32
    lora_r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 1
    max_length: int = 512
    seed: int = 0
    log_interval: int = 10

    def __post_init__(self) -> None:
        """Require the data files and an even batch."""
        super().__post_init__()
        if not self.refusal_data_path or not self.utility_data_path:
            raise ValueError("refusal_data_path and utility_data_path are required")
        if self.batch_size % 2:
            raise ValueError("batch_size must be even (half refusal, half utility)")
