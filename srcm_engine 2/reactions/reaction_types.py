from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

# D: {"U": DU, "V": DV, ...}
# C: {"U": CU, "V": CV, ...}  (NOTE: in your engine this is PDE *mass per compartment*)
# rates: dict of rate constants
# h: SSA compartment size
PropensityFn = Callable[[Dict[str, float], Dict[str, float], Dict[str, float], float], float]


@dataclass(frozen=True)
class HybridReaction:
    """
    A single hybrid reaction channel operating within one SSA compartment.

    state_change convention:
      - 'D_<species>' means SSA discrete particle change in that compartment
      - 'C_<species>' means PDE particle-mass change (in units of particles)
        Engine implements this as +/- (1/h) concentration over that compartment's PDE slice.

    IMPORTANT:
      If state_change contains any 'C_<sp>' with delta < 0, then the reaction consumes
      continuous mass and must be gated by sufficient PDE concentration/mass.
    """
    label: str
    reactants: Dict[str, int]
    products: Dict[str, int]
    propensity: PropensityFn
    state_change: Dict[str, int]
    description: Optional[str] = None

    # Derived metadata (auto-filled)
    consumes_continuous: bool = field(init=False)
    produces_continuous: bool = field(init=False)
    consumed_species: Tuple[str, ...] = field(init=False)
    produced_species: Tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        # ---- validate keys ----
        consumed = []
        produced = []

        for key, delta in self.state_change.items():
            if not isinstance(key, str) or "_" not in key:
                raise ValueError(f"{self.label}: invalid state_change key '{key}' (expected 'D_<sp>' or 'C_<sp>')")

            prefix, sp = key.split("_", 1)
            if prefix not in ("D", "C"):
                raise ValueError(f"{self.label}: unknown prefix '{prefix}' in state_change key '{key}'")

            if not isinstance(delta, int):
                raise TypeError(f"{self.label}: state_change['{key}'] must be int (got {type(delta).__name__})")

            if prefix == "C":
                if delta < 0:
                    consumed.append(sp)
                elif delta > 0:
                    produced.append(sp)

        object.__setattr__(self, "consumed_species", tuple(sorted(set(consumed))))
        object.__setattr__(self, "produced_species", tuple(sorted(set(produced))))
        object.__setattr__(self, "consumes_continuous", len(consumed) > 0)
        object.__setattr__(self, "produces_continuous", len(produced) > 0)
