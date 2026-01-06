import numpy as np
from srcm_engine import Domain, ConversionParams
from srcm_engine.reactions import HybridReactionSystem
from srcm_engine.core.engine import SRCMEngine


def test_run_records_correct_shapes():
    domain = Domain(length=1.0, n_ssa=5, pde_multiple=3, boundary="periodic")
    conversion = ConversionParams(threshold=10, rate=0.0)  # turn off conversion for simplicity

    reactions = HybridReactionSystem(species=["U", "V"])
    # no hybrid reactions
    def pde_terms(C, rates):
        return np.zeros_like(C)

    engine = SRCMEngine(
        reactions=reactions,
        pde_reaction_terms=pde_terms,
        diffusion_rates={"U": 0.0, "V": 0.0},
        domain=domain,
        conversion=conversion,
        reaction_rates={}
    )

    init_ssa = np.zeros((2, domain.K), dtype=int)
    init_pde = np.zeros((2, domain.n_pde), dtype=float)

    res = engine.run(init_ssa, init_pde, time=1.0, dt=0.2, seed=0)

    assert res.ssa.shape == (2, domain.K, 6)        # 0..1 in steps of 0.2 => 6 points
    assert res.pde.shape == (2, domain.n_pde, 6)
    assert res.time.shape == (6,)
    assert np.allclose(res.time, np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]))


def test_combined_grid_shape():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=4, boundary="periodic")
    conversion = ConversionParams(threshold=10, rate=0.0)

    reactions = HybridReactionSystem(species=["U"])
    def pde_terms(C, rates):
        return np.zeros_like(C)

    engine = SRCMEngine(
        reactions=reactions,
        pde_reaction_terms=pde_terms,
        diffusion_rates={"U": 0.0},
        domain=domain,
        conversion=conversion,
        reaction_rates={}
    )

    init_ssa = np.zeros((1, domain.K), dtype=int)
    init_ssa[0, 0] = 1
    init_pde = np.zeros((1, domain.n_pde), dtype=float)

    res = engine.run(init_ssa, init_pde, time=0.2, dt=0.2, seed=0)
    comb = res.combined()

    assert comb.shape == (1, domain.n_pde, 2)
    # in compartment 0, SSA contributes 1/h concentration
    assert np.allclose(comb[0, 0:4, 0], 1.0 / domain.h)
