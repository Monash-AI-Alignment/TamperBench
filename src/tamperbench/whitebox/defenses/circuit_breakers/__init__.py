"""Circuit Breakers via Representation Rerouting (Zou et al., 2024)."""

from tamperbench.whitebox.defenses.circuit_breakers.circuit_breakers import CircuitBreakers
from tamperbench.whitebox.defenses.circuit_breakers.config import CircuitBreakersConfig

__all__ = ["CircuitBreakers", "CircuitBreakersConfig"]
