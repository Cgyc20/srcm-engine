from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from .reaction_types import HybridReaction, PropensityFn


@dataclass
class HybridReactionSystem:
    """
    Container for species and hybrid reactions.

    This is intentionally lightweight:
    - It stores the reactions
    - It helps build local D/C dict-views for propensity evaluation
    """
    species: List[str]

    def __post_init__(self):
        if len(self.species) == 0:
            raise ValueError("species must be a non-empty list")
        if len(set(self.species)) != len(self.species):
            raise ValueError("species must be unique")
        self.species_index = {s: i for i, s in enumerate(self.species)}

        self.pure_reactions: list[dict] = []
        self.hybrid_reactions: list[HybridReaction] = []

    @property
    def n_species(self) -> int:
        return len(self.species)

    def add_reaction(self, reactants: Dict[str, int], products: Dict[str, int], rate: float):
        """
        Optional: store macroscopic reactions for reference/documentation.
        (Engine doesn't need these if you provide PDE reaction terms separately.)
        """
        self.pure_reactions.append({"reactants": reactants, "products": products, "rate": float(rate)})

    def add_hybrid_reaction(
        self,
        reactants: Dict[str, int],
        products: Dict[str, int],
        propensity: PropensityFn,
        state_change: Dict[str, int],
        label: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        if label is None:
            label = f"HR{len(self.hybrid_reactions) + 1}"

        # light validation: all species tokens must look like D_U or C_V etc.
        for token_dict in (reactants, products, state_change):
            for key in token_dict.keys():
                if "_" not in key:
                    raise ValueError(f"Key '{key}' must contain '_' (e.g. 'D_U', 'C_V')")
                prefix, sp = key.split("_", 1)
                if prefix not in ("D", "C"):
                    raise ValueError(f"Key '{key}' must start with 'D_' or 'C_'")
                if sp not in self.species:
                    raise ValueError(f"Unknown species '{sp}' in key '{key}'. Known: {self.species}")

        self.hybrid_reactions.append(
            HybridReaction(
                label=label,
                reactants=reactants,
                products=products,
                propensity=propensity,
                state_change=state_change,
                description=description,
            )
        )

    def local_DC_views(self, ssa_counts_compartment: Dict[str, int], pde_mass_compartment: Dict[str, float]):
        """
        Return D and C dicts in the exact form your lambdas expect.

        Example
        -------
        D = {"U": 12, "V": 3}
        C = {"U": 8.4, "V": 0.1}
        """
        return ssa_counts_compartment, pde_mass_compartment
