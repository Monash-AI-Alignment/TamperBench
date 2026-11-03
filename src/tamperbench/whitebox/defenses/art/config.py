"""Configuration for ART, abliteration-resistant tuning (Kuo et al., 2026)."""

from dataclasses import dataclass

from tamperbench.whitebox.defenses.defense import AlignmentDefenseConfig


@dataclass
class ARTConfig(AlignmentDefenseConfig):
    """Abliteration-resistant tuning.

    Defaults follow arXiv 2605.26526 Algorithms 2-3 and appendix A.4 for Llama-8B:
    gamma_ar 1, gamma_ret 2, lr 2e-5, batch 4, 1,000 steps, layer re-selected every 10
    steps; full-parameter training. The paper does not give epsilon or the ablation
    strength; defaults are 1e-3 and 1.0.

    Attributes:
        pairs_data_path: JSONL {"prompt", "chosen", "rejected"}: y+ (safe) and y- (harmful) responses.
        gamma_ar: Weight of the abliteration-resistance term.
        gamma_ret: Weight of the retain (chosen-response) term.
        epsilon: Added inside the log of the harmful cross-entropy.
        ablation_alpha: Strength of the simulated abliteration.
        refresh_every: Steps between re-selecting the ablated layer and its direction.
        learning_rate: Optimiser learning rate (constant).
        batch_size: Pairs per step.
        num_steps: Optimiser steps.
        max_length: Maximum tokens per conversation.
        eight_bit_optimizer: Use bitsandbytes paged 8-bit AdamW so an 8B model fits one 80 GB GPU.
        seed: Seed for data order.
        log_interval: Steps between log records.
    """

    pairs_data_path: str = ""
    gamma_ar: float = 1.0
    gamma_ret: float = 2.0
    epsilon: float = 1e-3
    ablation_alpha: float = 1.0
    refresh_every: int = 10
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_steps: int = 1000
    max_length: int = 512
    eight_bit_optimizer: bool = True
    seed: int = 0
    log_interval: int = 10

    def __post_init__(self) -> None:
        """Require the pairs file."""
        super().__post_init__()
        if not self.pairs_data_path:
            raise ValueError("pairs_data_path is required")
