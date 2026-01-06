from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional


# Your propensity signature:
#   D: {"U": DU, "V": DV, ...}  (integers or floats)
#   C: {"U": CU, "V": CV, ...}  (floats)
#   rates: arbitrary dict of rate constants
#   h: SSA compartment size
PropensityFn = Callable[[Dict[str, float], Dict[str, float], Dict[str, float], float], float]


@dataclass(frozen=True)
class HybridReaction:
    """
    A single hybrid reaction channel operating within one SSA compartment.

    state_change convention:
      - 'D_<species>' means SSA discrete particle change in that compartment
      - 'C_<species>' means PDE mass change of ±1 particle in that compartment slice
        (engine will implement as ±(1/h) concentration over that slice)
    """
    label: str
    reactants: Dict[str, int]
    products: Dict[str, int]
    propensity: PropensityFn
    state_change: Dict[str, int]
    description: Optional[str] = None
