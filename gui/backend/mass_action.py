import numpy as np
from typing import Callable, Dict, List, Tuple


def build_mass_action_terms(
    species: List[str],
    reactions: List[Dict],
) -> Callable:
    """
    Returns f(*S_arrays, r_dict) -> tuple(dS_arrays)
    - species: ["A", "B", ...]
    - reactions: list of {"reactants": {"A":1}, "products": {"B":1}, "rate_name": "alpha"}

    Uses mass-action: v = k * Π S_i^{stoich}
    Then dS += (products - reactants) * v
    """
    idx = {s: i for i, s in enumerate(species)}

    def terms(*args) -> Tuple[np.ndarray, ...]:
        *S, r = args  # last arg is rate dict
        dS = [np.zeros_like(S[i], dtype=float) for i in range(len(species))]

        for rxn in reactions:
            reactants = rxn["reactants"]
            products = rxn["products"]
            kname = rxn["rate_name"]
            k = float(r[kname])

            v = k
            for sp, sto in reactants.items():
                v = v * (S[idx[sp]] ** int(sto))

            for sp, sto in reactants.items():
                dS[idx[sp]] -= int(sto) * v
            for sp, sto in products.items():
                dS[idx[sp]] += int(sto) * v

        return tuple(dS)

    return terms
