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


# March 16th 2026:
# Commenting the following test out, because we have changed the logic to the new probabalistic approach!



# def test_apply_cd_event_updates_ssa_and_pde_mass():
#     domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")  # h=1
#     engine = make_engine(domain)

#     ssa = np.zeros((2, domain.K), dtype=int)
#     pde = np.zeros((2, domain.n_pde), dtype=float)

#     # put exactly 1 particle worth of mass of U in compartment 0
#     # since h=1, adding +1 mass is +1/h = 1 conc across 4 cells
#     pde[0, 0:4] = 1.0

#     state = HybridState(ssa, pde)
#     rng = np.random.default_rng(0)

#     # CD block for species U is block = n_species + 0 = 2
#     # flat idx = block*K + comp
#     n_species = 2
#     block = n_species + 0
#     idx = block * domain.K + 0

#     # apply
#     engine.apply_event(idx, state, rng, pde_mass=None)

#     assert state.ssa[0, 0] == 1  # U gained a discrete particle

#     mass = pde_mass_per_compartment(state.pde, domain.pde_multiple, domain.dx)
#     assert np.allclose(mass[0, 0], 0.0)  # removed one unit of PDE mass

# March 16th, new test.
def test_apply_cd_event_subunit_mass_adds_particle_when_rng_succeeds():
    domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")
    engine = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # compartment mass Y_k = 0.8 in compartment 0 for species U
    pde[0, 0:4] = 0.8 
    # h = 1.0, dx = h/4 = 0.25
    # PDE_mass in k=1 = 3.2*0.25 = 0.8! less than one particles worth.

    state = HybridState(ssa, pde)
    rng  = np.random.default_rng(0)

    n_species = 2
    block = n_species + 0
    idx = block * domain.K + 0

    engine.apply_event(idx, state, rng, pde_mass=None)

    mass = pde_mass_per_compartment(state.pde, domain.pde_multiple, domain.dx)
    assert np.allclose(mass[0, 0], 0.0)
    assert state.ssa[0, 0] == 1

def test_apply_cd_event_subunit_mass_does_not_add_particle_when_rng_fails():
    domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")
    engine = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # compartment mass Y_k = 0.2 in compartment 0 for species U
    pde[0, 0:4] = 0.2

    state = HybridState(ssa, pde)
    rng = np.random.default_rng(0)  # first draw ~0.6369 > 0.2

    n_species = 2
    block = n_species + 0
    idx = block * domain.K + 0

    engine.apply_event(idx, state, rng, pde_mass=None)

    mass = pde_mass_per_compartment(state.pde, domain.pde_multiple, domain.dx)
    assert np.allclose(mass[0, 0], 0.0)
    assert state.ssa[0, 0] == 0


def test_apply_cd_event_redistributes_when_uniform_removal_would_fail():
    domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")
    engine = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # total mass = 1.25 in compartment 0, but first cell is below 1/h = 1
    pde[0, 0:4] = np.array([0.5, 1.5, 1.5, 1.5], dtype=float)

    state = HybridState(ssa, pde.copy())
    rng = np.random.default_rng(0)

    n_species = 2
    block = n_species + 0
    idx = block * domain.K + 0

    engine.apply_event(idx, state, rng, pde_mass=None)
    print(f"The pde state is: {state.pde[0:4]}")
    
    assert state.ssa[0, 0] == 1 #Assert that the first species will gain a particle.
    
    assert np.allclose(state.pde[0, 0:4], np.array([0.0, 1/3, 1/3, 1/3]))


def test_apply_cd_event_uniform_removal_exact_profile():
    domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")
    engine = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # h = 1, so uniform removal is subtracting 1 from each fine cell
    pde[0, 0:4] = np.array([1.2, 1.2, 1.2, 1.2], dtype=float)

    state = HybridState(ssa, pde.copy())
    rng = np.random.default_rng(0)

    n_species = 2
    block = n_species + 0
    idx = block * domain.K + 0

    engine.apply_event(idx, state, rng, pde_mass=None)

    assert state.ssa[0, 0] == 1
    assert np.allclose(state.pde[0, 0:4], np.array([0.2, 0.2, 0.2, 0.2]))


def test_apply_cd_event_equal_redistribution_exact_profile():
    domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")
    engine = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # first cell is too small, remaining mass removed equally from the last three
    pde[0, 0:4] = np.array([0.5, 1.5, 1.5, 1.5], dtype=float)

    state = HybridState(ssa, pde.copy())
    rng = np.random.default_rng(0)

    n_species = 2
    block = n_species + 0
    idx = block * domain.K + 0

    engine.apply_event(idx, state, rng, pde_mass=None)

    assert state.ssa[0, 0] == 1
    assert np.allclose(state.pde[0, 0:4], np.array([0.0, 1/3, 1/3, 1/3]))


def test_apply_cd_event_equal_redistribution_two_small_cells_exact_profile():
    domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")
    engine = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # masses in cells: 0.05, 0.05, 0.6, 1.3 -> total mass 2.0
    # remove 1.0 mass:
    # first two cells empty immediately, remaining 0.9 mass shared equally over last two
    # concentration decrement on last two = (0.9/2)/0.25 = 1.8
    # final slice = [0, 0, 0.6, 1.3] - [0, 0, 1.8, 1.8] is impossible if interpreted this way
    # so choose a cleaner profile:
    pde[0, 0:4] = np.array([0.2, 0.2, 2.2, 2.2], dtype=float)

    state = HybridState(ssa, pde.copy())
    rng = np.random.default_rng(0)

    n_species = 2
    block = n_species + 0
    idx = block * domain.K + 0

    engine.apply_event(idx, state, rng, pde_mass=None)

    # total initial mass = (0.2 + 0.2 + 2.2 + 2.2)*0.25 = 1.2
    # equal-share rule empties first two cells, then removes remaining 0.9 mass equally from last two
    # final last two concentrations are 0.4 each
    assert state.ssa[0, 0] == 1
    assert np.allclose(state.pde[0, 0:4], np.array([0.0, 0.0, 0.4, 0.4]))


def test_apply_cd_event_subunit_mass_exactly_one_is_threshold_case():
    domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")
    engine = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # exactly one particle mass, but not uniformly removable
    pde[0, 0:4] = np.array([0.5, 0.5, 1.5, 1.5], dtype=float)

    # Total mass = 4.0, 4.0 * 0.25 = 1.0
    # So it should end with 0 mass everywhere
    state = HybridState(ssa, pde.copy())
    rng = np.random.default_rng(0)

    n_species = 2
    block = n_species + 0
    idx = block * domain.K + 0

    engine.apply_event(idx, state, rng, pde_mass=None)

    # equal-share redistribution should remove everything exactly
    assert state.ssa[0, 0] == 1
    assert np.allclose(state.pde[0, 0:4], np.array([0.0, 0.0, 0.0, 0.0]))


def test_apply_cd_event_only_updates_target_compartment():
    domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")
    engine = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    pde[0, 0:4] = np.array([0.5, 1.5, 1.5, 1.5], dtype=float)
    pde[0, 4:8] = np.array([9.0, 9.0, 9.0, 9.0], dtype=float)
    pde[1, :] = 7.0

    state = HybridState(ssa, pde.copy())
    rng = np.random.default_rng(0)

    n_species = 2
    block = n_species + 0
    idx = block * domain.K + 0

    engine.apply_event(idx, state, rng, pde_mass=None)

    assert state.ssa[0, 0] == 1
    assert np.allclose(state.pde[0, 0:4], np.array([0.0, 1/3, 1/3, 1/3]))
    assert np.allclose(state.pde[0, 4:8], np.array([9.0, 9.0, 9.0, 9.0]))
    assert np.allclose(state.pde[1, :], np.full(domain.n_pde, 7.0))


def test_apply_cd_event_on_species_v_does_not_change_species_u():
    domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")
    engine = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    pde[0, 4:8] = np.array([3.0, 3.0, 3.0, 3.0], dtype=float)  # species U untouched
    pde[1, 4:8] = np.array([0.5, 1.5, 1.5, 1.5], dtype=float)  # species V updated

    state = HybridState(ssa, pde.copy())
    rng = np.random.default_rng(0)

    n_species = 2
    block = n_species + 1  # species V
    idx = block * domain.K + 1

    engine.apply_event(idx, state, rng, pde_mass=None)

    assert state.ssa[1, 1] == 1
    assert np.allclose(state.pde[1, 4:8], np.array([0.0, 1/3, 1/3, 1/3]))
    assert np.allclose(state.pde[0, 4:8], np.array([3.0, 3.0, 3.0, 3.0]))


def test_apply_cd_event_subunit_mass_clears_only_target_compartment_when_rng_fails():
    domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")
    engine = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    pde[0, 0:4] = np.array([0.2, 0.2, 0.2, 0.2], dtype=float)  # Y_k = 0.2
    pde[0, 4:8] = np.array([5.0, 5.0, 5.0, 5.0], dtype=float)

    state = HybridState(ssa, pde.copy())
    rng = np.random.default_rng(0)  # first draw > 0.2, so no discrete add

    n_species = 2
    block = n_species + 0
    idx = block * domain.K + 0

    engine.apply_event(idx, state, rng, pde_mass=None)

    assert state.ssa[0, 0] == 0 # We have no addition of particle 
    assert np.allclose(state.pde[0, 0:4], np.array([0.0, 0.0, 0.0, 0.0]))
    assert np.allclose(state.pde[0, 4:8], np.array([5.0, 5.0, 5.0, 5.0]))


def test_apply_cd_event_preserves_nonnegativity_after_equal_redistribution():
    domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")
    engine = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    pde[0, 0:4] = np.array([0.01, 0.7, 1.64, 1.65], dtype=float)

    state = HybridState(ssa, pde.copy())
    rng = np.random.default_rng(0)

    n_species = 2
    block = n_species + 0
    idx = block * domain.K + 0

    engine.apply_event(idx, state, rng, pde_mass=None)

    assert state.ssa[0, 0] == 1
    assert np.all(state.pde[0, 0:4] >= -1e-12)


def test_apply_cd_event_removes_exactly_one_particle_mass_in_equal_redistribution_case():
    domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")
    engine = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    pde[0, 0:4] = np.array([0.5, 1.5, 1.5, 1.5], dtype=float)

    state = HybridState(ssa, pde.copy())
    rng = np.random.default_rng(0)

    n_species = 2
    block = n_species + 0
    idx = block * domain.K + 0

    mass_before = pde_mass_per_compartment(state.pde, domain.pde_multiple, domain.dx)
    engine.apply_event(idx, state, rng, pde_mass=None)
    mass_after = pde_mass_per_compartment(state.pde, domain.pde_multiple, domain.dx)

    assert np.allclose(mass_before[0, 0] - mass_after[0, 0], 1.0)




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

def test_apply_cd_event_subunit_mass_two_consecutive_draws_give_different_outcomes():
    domain = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")
    engine = make_engine(domain)

    rng = np.random.default_rng(0)

    n_species = 2
    block = n_species + 0
    idx = block * domain.K + 0

    # First state: Y_k = 0.8, first RNG draw is ~0.6369, so discrete particle is added
    ssa1 = np.zeros((2, domain.K), dtype=int)
    pde1 = np.zeros((2, domain.n_pde), dtype=float)
    pde1[0, 0:4] = 0.8
    state1 = HybridState(ssa1, pde1)

    engine.apply_event(idx, state1, rng, pde_mass=None)

    assert state1.ssa[0, 0] == 1
    assert np.allclose(state1.pde[0, 0:4], np.zeros(4))

    # Second state: Y_k = 0.2, second RNG draw is ~0.2698, so no discrete particle is added
    ssa2 = np.zeros((2, domain.K), dtype=int)
    pde2 = np.zeros((2, domain.n_pde), dtype=float)
    pde2[0, 0:4] = 0.2
    state2 = HybridState(ssa2, pde2)

    engine.apply_event(idx, state2, rng, pde_mass=None)

    assert state2.ssa[0, 0] == 0
    assert np.allclose(state2.pde[0, 0:4], np.zeros(4))

    