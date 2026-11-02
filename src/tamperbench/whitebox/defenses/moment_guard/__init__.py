"""Moment-guarded refusal training (polynomial-kernel MMD between harmful and harmless activations)."""

from tamperbench.whitebox.defenses.moment_guard.config import MomentGuardConfig
from tamperbench.whitebox.defenses.moment_guard.moment_guard import MomentGuard

__all__ = ["MomentGuard", "MomentGuardConfig"]
