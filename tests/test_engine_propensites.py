# import numpy as np
# from srcm_engine import Domain, ConversionParams
# from srcm_engine.reactions import HybridReactionSystem
# from srcm_engine.core.engine import SRCMEngine
# from srcm_engine.state import HybridState
# from srcm_engine.conversion import combined_mass


# def test_engine_build_propensity_vector_shape():
#     domain = Domain(length=1.0, n_ssa=4, pde_multiple=2, boundary="periodic")
#     conversion = ConversionParams(threshold=5, rate=1.0)

#     reactions = HybridReactionSystem(species=["U", "V"])
#     reactions.add_hybrid_reaction(
#         reactants={"D_U": 1},
#         products={"D_U": 1},
#         propensity=lambda D, C, r, h: 0.0,
#         state_change={"D_U": 0},
#         label="dummy"
#     )

#     def pde_terms(C, rates):
#         return np.zeros_like(C)

#     engine = SRCMEngine(
#         reactions=reactions,
#         pde_reaction_terms=pde_terms,
#         diffusion_rates={"U": 0.1, "V": 0.2},
#         domain=domain,
#         conversion=conversion,
#         reaction_rates={}
#     )

#     ssa = np.zeros((2, domain.K), dtype=int)
#     pde = np.zeros((2, domain.n_pde), dtype=float)
#     state = HybridState(ssa=ssa, pde=pde)
#     state.assert_consistent(domain)

#     comb, pde_mass = combined_mass(state.ssa, state.pde, domain.pde_multiple, domain.dx)
#     exceeds = conversion.exceeds_threshold_mask(comb)
#     sufficient = conversion.sufficient_pde_mass_mask(state.pde, domain.pde_multiple, domain.h)

#     a = engine.build_propensity_vector(
#         state=state,
#         pde_mass=pde_mass,
#         exceeds_mask=exceeds,
#         sufficient_mask=sufficient,
#         out=None
#     )

#     n_species = 2
#     n_hybrid = 1
#     expected_len = (3 * n_species + n_hybrid) * domain.K
#     assert a.shape == (expected_len,)
#     assert np.all(a >= 0)


import numpy as np
from srcm_engine import Domain, ConversionParams
from srcm_engine.reactions import HybridReactionSystem
from srcm_engine.core.engine import SRCMEngine
from srcm_engine.state import HybridState
from srcm_engine.conversion import combined_mass


def make_engine(domain):
    conversion = ConversionParams(
        DC_threshold=5.0,
        CD_threshold=3.0,
        rate=2.0,
    )

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
    return engine, conversion


def test_engine_build_propensity_vector_shape():
    domain = Domain(length=1.0, n_ssa=4, pde_multiple=2, boundary="periodic")
    engine, conversion = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)
    state = HybridState(ssa=ssa, pde=pde)
    state.assert_consistent(domain)

    comb, pde_mass = combined_mass(state.ssa, state.pde, domain.pde_multiple, domain.dx)
    DC_mask = conversion.DC_mask(comb)
    CD_mask = conversion.CD_mask(comb)
    sufficient = conversion.sufficient_pde_mass_mask(state.pde, domain.pde_multiple, domain.h)

    a = engine.build_propensity_vector(
        state=state,
        pde_mass=pde_mass,
        DC_mask=DC_mask,
        CD_mask=CD_mask,
        sufficient_mask=sufficient,
        out=None,
    )

    n_species = 2
    n_hybrid = 1
    expected_len = (3 * n_species + n_hybrid) * domain.K
    assert a.shape == (expected_len,)
    assert np.all(a >= 0)


def test_cd_propensity_uses_CD_mask_and_pde_mass():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, conversion = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # domain.h = 0.5, dx = 0.25, each SSA compartment has 2 PDE cells
    # Species U:
    # compartment 0 mass = (1 + 1) * 0.25 = 0.5  -> below CD_threshold=3, allowed
    # compartment 1 mass = (10 + 10) * 0.25 = 5.0 -> not below CD_threshold, not allowed
    pde[0, 0:2] = np.array([1.0, 1.0])
    pde[0, 2:4] = np.array([10.0, 10.0])

    state = HybridState(ssa=ssa, pde=pde)

    comb, pde_mass = combined_mass(state.ssa, state.pde, domain.pde_multiple, domain.dx)
    DC_mask = conversion.DC_mask(comb)
    CD_mask = conversion.CD_mask(comb)
    sufficient = conversion.sufficient_pde_mass_mask(state.pde, domain.pde_multiple, domain.h)

    a = engine.build_propensity_vector(
        state=state,
        pde_mass=pde_mass,
        DC_mask=DC_mask,
        CD_mask=CD_mask,
        sufficient_mask=sufficient,
    )

    n_species = 2
    K = domain.K

    # CD block for species U
    block = n_species + 0
    start = block * K
    end = start + K

    # gamma = 2.0, so propensity = 2 * mass where CD_mask == 1
    expected = np.array([2.0 * 0.5, 0.0])
    assert np.allclose(a[start:end], expected)


def test_dc_propensity_uses_DC_mask_and_ssa_count():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, conversion = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # Species U discrete counts
    ssa[0, 0] = 4
    ssa[0, 1] = 7

    # Add PDE mass so combined mass is:
    # compartment 0: 4 + 0.5 = 4.5 -> not above DC_threshold=5
    # compartment 1: 7 + 0.0 = 7.0 -> above DC_threshold=5
    pde[0, 0:2] = np.array([1.0, 1.0])  # mass 0.5
    pde[0, 2:4] = np.array([0.0, 0.0])  # mass 0.0

    state = HybridState(ssa=ssa, pde=pde)

    comb, pde_mass = combined_mass(state.ssa, state.pde, domain.pde_multiple, domain.dx)
    DC_mask = conversion.DC_mask(comb)
    CD_mask = conversion.CD_mask(comb)
    sufficient = conversion.sufficient_pde_mass_mask(state.pde, domain.pde_multiple, domain.h)

    a = engine.build_propensity_vector(
        state=state,
        pde_mass=pde_mass,
        DC_mask=DC_mask,
        CD_mask=CD_mask,
        sufficient_mask=sufficient,
    )

    n_species = 2
    K = domain.K

    # DC block for species U
    block = 2 * n_species + 0
    start = block * K
    end = start + K

    # gamma = 2.0, only compartment 1 allowed
    expected = np.array([0.0, 2.0 * 7.0])
    assert np.allclose(a[start:end], expected)


def test_hysteresis_deadband_gives_zero_cd_and_dc_propensities():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, conversion = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # Put species U combined mass strictly between CD=3 and DC=5 in both compartments
    # compartment 0: 4
    # compartment 1: 4
    ssa[0, :] = np.array([4, 4])

    state = HybridState(ssa=ssa, pde=pde)

    comb, pde_mass = combined_mass(state.ssa, state.pde, domain.pde_multiple, domain.dx)
    DC_mask = conversion.DC_mask(comb)
    CD_mask = conversion.CD_mask(comb)
    sufficient = conversion.sufficient_pde_mass_mask(state.pde, domain.pde_multiple, domain.h)

    a = engine.build_propensity_vector(
        state=state,
        pde_mass=pde_mass,
        DC_mask=DC_mask,
        CD_mask=CD_mask,
        sufficient_mask=sufficient,
    )

    n_species = 2
    K = domain.K

    cd_block = n_species + 0
    dc_block = 2 * n_species + 0

    cd_vals = a[cd_block * K:(cd_block + 1) * K]
    dc_vals = a[dc_block * K:(dc_block + 1) * K]

    assert np.allclose(cd_vals, 0.0)
    assert np.allclose(dc_vals, 0.0)


def test_negative_pde_mass_does_not_create_negative_cd_propensity():
    domain = Domain(length=1.0, n_ssa=2, pde_multiple=2, boundary="periodic")
    engine, conversion = make_engine(domain)

    ssa = np.zeros((2, domain.K), dtype=int)
    pde = np.zeros((2, domain.n_pde), dtype=float)

    # Species U compartment 0 has negative integrated PDE mass
    pde[0, 0:2] = np.array([-1.0, -1.0])

    state = HybridState(ssa=ssa, pde=pde)

    comb, pde_mass = combined_mass(state.ssa, state.pde, domain.pde_multiple, domain.dx)
    DC_mask = conversion.DC_mask(comb)
    CD_mask = conversion.CD_mask(comb)
    sufficient = conversion.sufficient_pde_mass_mask(state.pde, domain.pde_multiple, domain.h)

    a = engine.build_propensity_vector(
        state=state,
        pde_mass=pde_mass,
        DC_mask=DC_mask,
        CD_mask=CD_mask,
        sufficient_mask=sufficient,
    )

    n_species = 2
    K = domain.K

    cd_block = n_species + 0
    cd_vals = a[cd_block * K:(cd_block + 1) * K]

    assert np.all(cd_vals >= 0.0)

    