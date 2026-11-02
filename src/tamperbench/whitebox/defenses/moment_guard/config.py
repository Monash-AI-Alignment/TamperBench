"""Configuration for the moment-guard defense."""

from dataclasses import dataclass, field

from tamperbench.whitebox.defenses.defense import AlignmentDefenseConfig


@dataclass
class MomentGuardConfig(AlignmentDefenseConfig):
    """Configuration for moment-guarded refusal training.

    Trains refusal into `input_checkpoint_path` (typically an abliterated model) with LoRA,
    minimising

        CE(refusals | harmful prompts) + kl_weight * KL(input model || trained) on harmless
        conversations + moment_weight * D_m,

    where D_m sums, over `moment_layers`, the unbiased MMD^2 between harmful and harmless
    post-instruction activations under the kernels (x^T y)^j for j = 1..m-1. The KL
    reference is the input model itself (LoRA adapters disabled), so no second copy is
    loaded.

    Attributes:
        refusal_data_path: JSONL of {"prompt", "response"}: harmful prompts with refusals.
        harmless_data_path: JSONL of {"prompt", "response"}: harmless conversations.
        moment_layers: Residual-stream (`hidden_states`) indices whose moments are matched.
        m: Lowest moment allowed to differ; degrees 1..m-1 are matched (m=2: means).
        moment_weight: lambda. 0 trains the same refusal objective without the penalty (control).
        subspace_rank: Top principal components (per layer, of the input model's activations)
            in which degrees >= 2 are matched; 0 uses the full space.
        bank_size: Earlier activations kept per class, layer and position to widen the
            moment estimates; 0 uses the current batch only.
        whiten_shrinkage: If > 0, the degree-1 term is the mean gap under the shrunk
            inverse within-class covariance (see `moments.whitened_mean_gap`), which also
            closes gaps along low-variance directions that linear probes exploit; 0 uses
            the Euclidean gap.
        num_positions: Post-instruction tokens matched; None uses every token the chat
            template appends after the user message.
        kl_weight: Weight of the KL term.
        calibration_samples: Prompts per class used to fix per-layer scales and subspaces.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
        lora_target_modules: Modules given LoRA adapters.
        learning_rate: AdamW learning rate.
        batch_size: Conversations per class per step.
        num_steps: Optimiser steps.
        max_length: Maximum tokens per conversation (prompts longer than this are dropped).
        seed: Seed for data order and LoRA initialisation.
        log_interval: Steps between log records.
    """

    refusal_data_path: str = ""
    harmless_data_path: str = ""
    moment_layers: tuple[int, ...] = field(default_factory=lambda: tuple(range(8, 25)))
    m: int = 2
    moment_weight: float = 1.0
    subspace_rank: int = 64
    bank_size: int = 256
    whiten_shrinkage: float = 0.0
    num_positions: int | None = None
    kl_weight: float = 1.0
    calibration_samples: int = 256
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_steps: int = 1000
    max_length: int = 512
    seed: int = 0
    log_interval: int = 10

    def __post_init__(self) -> None:
        """Validate paths and the moment order."""
        super().__post_init__()
        if not self.refusal_data_path or not self.harmless_data_path:
            raise ValueError("refusal_data_path and harmless_data_path are required")
        if self.m < 2:
            raise ValueError(f"m must be >= 2, got {self.m}; use moment_weight=0 for the control")
