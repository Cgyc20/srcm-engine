import numpy as np
from srcm_engine import Domain, ConversionParams
from srcm_engine.reactions import HybridReactionSystem
from srcm_engine.core.engine import SRCMEngine


def test_run_repeats_mean_matches_single_when_deterministic():
    domain = Domain(length=1.0, n_ssa=3, pde_multiple=2, boundary="periodic")
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
    init_pde = np.zeros((1, domain.n_pde), dtype=float)

    single = engine.run(init_ssa, init_pde, time=1.0, dt=0.5, seed=0)
    mean = engine.run_repeats(init_ssa, init_pde, time=1.0, dt=0.5, repeats=5, seed=0)

    assert np.allclose(mean.ssa, single.ssa)
    assert np.allclose(mean.pde, single.pde)
    assert np.allclose(mean.time, single.time)
