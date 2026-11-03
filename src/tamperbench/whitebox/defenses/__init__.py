"""Alignment-stage defenses against tampering supported by tamperbench."""

from tamperbench.whitebox.defenses.art import ART, ARTConfig
from tamperbench.whitebox.defenses.booster import Booster, BoosterConfig
from tamperbench.whitebox.defenses.circuit_breakers import CircuitBreakers, CircuitBreakersConfig
from tamperbench.whitebox.defenses.crl import CRL, CRLConfig
from tamperbench.whitebox.defenses.ctrl import CTRL, CTRLConfig
from tamperbench.whitebox.defenses.defense import (
    AlignmentDefense,
    AlignmentDefenseConfig,
)
from tamperbench.whitebox.defenses.moment_guard import MomentGuard, MomentGuardConfig
from tamperbench.whitebox.defenses.refat import ReFAT, ReFATConfig
from tamperbench.whitebox.defenses.registry import DEFENSES_REGISTRY
from tamperbench.whitebox.defenses.rsn_tune import RSNTune, RSNTuneConfig
from tamperbench.whitebox.defenses.sdd.sdd import SDD, SDDConfig
from tamperbench.whitebox.defenses.t_vaccine.t_vaccine import (
    TVaccineConfig,
    TVaccineDefense,
)
from tamperbench.whitebox.defenses.t_vaccine.tar import (
    TARTVaccineConfig,
    TARTVaccineDefense,
)
from tamperbench.whitebox.defenses.tar.tar import (
    TARConfig,
    TARDefense,
)

__all__: list[str] = [
    "ART",
    "CRL",
    "CTRL",
    "DEFENSES_REGISTRY",
    "SDD",
    "ARTConfig",
    "AlignmentDefense",
    "AlignmentDefenseConfig",
    "Booster",
    "BoosterConfig",
    "CRLConfig",
    "CTRLConfig",
    "CircuitBreakers",
    "CircuitBreakersConfig",
    "MomentGuard",
    "MomentGuardConfig",
    "RSNTune",
    "RSNTuneConfig",
    "ReFAT",
    "ReFATConfig",
    "SDDConfig",
    "TARConfig",
    "TARDefense",
    "TARTVaccineConfig",
    "TARTVaccineDefense",
    "TVaccineConfig",
    "TVaccineDefense",
]
