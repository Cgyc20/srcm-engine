from typing import Dict

from srcm_engine.core import HybridModel
from backend.mass_action import build_mass_action_terms


def build_model_from_spec(spec: Dict) -> HybridModel:
    m = HybridModel(species=spec["species"])

    d = spec["domain"]
    m.domain(
        L=float(d["L"]),
        K=int(d["K"]),
        pde_multiple=int(d["pde_multiple"]),
        boundary=str(d["boundary"]),
    )

    # diffusion expects keyword args per species
    m.diffusion(**{k: float(v) for k, v in spec["diffusion"].items()})

    c = spec["conversion"]
    m.conversion(threshold=int(c["threshold"]), rate=float(c["rate"]))

    rxns = spec["reactions"]
    m.reaction_terms(build_mass_action_terms(spec["species"], rxns))

    for rxn in rxns:
        m.add_reaction(rxn["reactants"], rxn["products"], rate_name=rxn["rate_name"])

    m.build(rates={k: float(v) for k, v in spec["rates"].items()})
    return m
