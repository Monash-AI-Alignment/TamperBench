"""ReFAT: refusal feature adversarial training (Yu et al., ICLR 2025)."""

from tamperbench.whitebox.defenses.refat.config import ReFATConfig
from tamperbench.whitebox.defenses.refat.refat import ReFAT

__all__ = ["ReFAT", "ReFATConfig"]
