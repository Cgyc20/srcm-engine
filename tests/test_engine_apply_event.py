import numpy as np
from srcm_engine import Domain, ConversionParams
from srcm_engine.reactions import HybridReactionSystem
from srcm_engine.core.engine import SRCMEngine
from srcm_engine.state import HybridState
from srcm_engine.conversion import pde_mass_per_compartment


def make_engine(domain):
    conversion = ConversionParams(threshold=5, rate=1.0)

    reactions = HybridReactionSystem(species=["U", "V"])
    # One hybrid reaction: D_U += 1
    reactions.add_hybrid_reaction(
        reactants={"D_U": 2},
        products={"D_U": 3},
        propensity=lambda D, C, r, h: 0.0,
        state_change={"D_U": +1},
        label="inc_U"
    )

    def pde_terms(C, rates):
        return np.zeros_like(C)

    return SRCMEngine(
        reactions=reactions,
        pde_reaction_terms=pde_terms,
        diffusion_rates={"U": 0.0, "V": 0.0},
        domain=domain,
        conversion=conversion,
        reaction_rates={}
    )


def test_apply_cd_event_updates_ssa_and_pde_mass():
    domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")  # h=1
    engine = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # put exactly 1 particle worth of mass of U in compartment 0
    # since h=1, adding +1 mass is +1/h = 1 conc across 4 cells
    pde[0, 0:4] = 1.0

    state = HybridState(ssa, pde)
    rng = np.random.default_rng(0)

    # CD block for species U is block = n_species + 0 = 2
    # flat idx = block*K + comp
    n_species = 2
    block = n_species + 0
    idx = block * domain.K + 0

    # apply
    engine.apply_event(idx, state, rng, pde_mass=None)

    assert state.ssa[0, 0] == 1  # U gained a discrete particle

    mass = pde_mass_per_compartment(state.pde, domain.pde_multiple, domain.dx)
    assert np.allclose(mass[0, 0], 0.0)  # removed one unit of PDE mass


def test_apply_dc_event_updates_ssa_and_pde_mass():
    domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")  # h=1
    engine = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    ssa[1, 1] = 1  # one V particle in compartment 1
    pde = np.zeros((2, domain.n_pde), dtype=float)

    state = HybridState(ssa, pde)
    rng = np.random.default_rng(0)

    # DC block for species V is block = 2*n_species + 1 = 5
    n_species = 2
    block = 2 * n_species + 1
    idx = block * domain.K + 1

    engine.apply_event(idx, state, rng, pde_mass=None)

    assert state.ssa[1, 1] == 0

    mass = pde_mass_per_compartment(state.pde, domain.pde_multiple, domain.dx)
    assert np.allclose(mass[1, 1], 1.0)  # added one unit of PDE mass


def test_apply_hybrid_reaction_event_updates_ssa():
    domain = Domain(length=1.0, n_ssa=3, pde_multiple=2, boundary="periodic")
    engine = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)
    state = HybridState(ssa, pde)
    rng = np.random.default_rng(0)

    # hybrid block base = 3*n_species = 6
    # rxn_idx=0 => block=6
    n_species = 2
    block = 3 * n_species + 0
    idx = block * domain.K + 2  # compartment 2

    engine.apply_event(idx, state, rng, pde_mass=None)

    assert state.ssa[0, 2] == 1  # U increased
