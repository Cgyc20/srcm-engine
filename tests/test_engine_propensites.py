import numpy as np
from srcm_engine import Domain, ConversionParams
from srcm_engine.reactions import HybridReactionSystem
from srcm_engine.core.engine import SRCMEngine
from srcm_engine.state import HybridState
from srcm_engine.conversion import combined_mass


def test_engine_build_propensity_vector_shape():
    domain = Domain(length=1.0, n_ssa=4, pde_multiple=2, boundary="periodic")
    conversion = ConversionParams(threshold=5, rate=1.0)

    reactions = HybridReactionSystem(species=["U", "V"])
    reactions.add_hybrid_reaction(
        reactants={"D_U": 1},
        products={"D_U": 1},
        propensity=lambda D, C, r, h: 0.0,
        state_change={"D_U": 0},
        label="dummy"
    )

    def pde_terms(C, rates):
        return np.zeros_like(C)

    engine = SRCMEngine(
        reactions=reactions,
        pde_reaction_terms=pde_terms,
        diffusion_rates={"U": 0.1, "V": 0.2},
        domain=domain,
        conversion=conversion,
        reaction_rates={}
    )

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)
    state = HybridState(ssa=ssa, pde=pde)
    state.assert_consistent(domain)

    comb, pde_mass = combined_mass(state.ssa, state.pde, domain.pde_multiple, domain.dx)
    exceeds = conversion.exceeds_threshold_mask(comb)
    sufficient = conversion.sufficient_pde_mass_mask(state.pde, domain.pde_multiple, domain.h)

    a = engine.build_propensity_vector(
        state=state,
        pde_mass=pde_mass,
        exceeds_mask=exceeds,
        sufficient_mask=sufficient,
        out=None
    )

    n_species = 2
    n_hybrid = 1
    expected_len = (3 * n_species + n_hybrid) * domain.K
    assert a.shape == (expected_len,)
    assert np.all(a >= 0)
