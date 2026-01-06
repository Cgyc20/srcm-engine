"""
Two-species hybrid diffusion-reaction simulation
================================================

System:
    A ⇌ B
    A -> B  with rate α
    B -> A  with rate β

Spatially resolved with diffusion (PDE + SSA hybrid).

Author: Charlie Cameron
Date: January 2026
"""

import numpy as np
from srcm_engine import (
    Domain,
    ConversionParams,
    HybridReactionSystem,
    SRCMEngine,
)
from srcm_engine.results.io import save_npz
from srcm_engine.animation_util import animate_results, AnimationConfig, plot_mass_time_series

# ============================================================================
# PDE REACTION TERMS (macroscopic)
# ============================================================================
def ab_pde_terms(C: np.ndarray, rates: dict) -> np.ndarray:
    """
    f(u,v) = βv − αu
    g(u,v) = αu − βv
    """
    U, V = C[0], C[1]
    alpha = float(rates["alpha"])
    beta = float(rates["beta"])

    dU = beta * V - alpha * U
    dV = alpha * U - beta * V
    return np.array([dU, dV])


# ============================================================================
# BUILD ENGINE
# ============================================================================
def build_engine():
    # --- parameters ---
    L = 10.0
    K = 40
    pde_multiple = 8
    boundary = "zero-flux"

    total_time = 30.0
    dt = 0.006

    # Diffusion coefficient
    D = 0.1 

    # Reaction rates
    alpha = 0.01
    beta = 0.01

    # Conversion parameters
    threshold = 4
    conversion_rate = 1.0

    # --- domain + conversion ---
    domain = Domain(length=L, n_ssa=K, pde_multiple=pde_multiple, boundary=boundary)
    conversion = ConversionParams(threshold=threshold, rate=conversion_rate)

    # --- hybrid reaction system ---
    reactions = HybridReactionSystem(species=["A", "B"])

    # A -> B
    reactions.add_hybrid_reaction(
        reactants={"D_A": 1},
        products={"D_B": 1},
        propensity=lambda D, C, r, h: r["alpha"] * D["A"],
        state_change={"D_A": -1, "D_B": +1},
        label="R1_A_to_B",
        description="A -> B (SSA local)",
    )

    # B -> A
    reactions.add_hybrid_reaction(
        reactants={"D_B": 1},
        products={"D_A": 1},
        propensity=lambda D, C, r, h: r["beta"] * D["B"],
        state_change={"D_B": -1, "D_A": +1},
        label="R2_B_to_A",
        description="B -> A (SSA local)",
    )

    # --- engine ---
    engine = SRCMEngine(
        reactions=reactions,
        pde_reaction_terms=ab_pde_terms,
        diffusion_rates={"A": D, "B": D},
        domain=domain,
        conversion=conversion,
        reaction_rates={"alpha": alpha, "beta": beta},
    )

    # Metadata for saving
    meta = {
        "model": "A⇌B",
        "alpha": alpha,
        "beta": beta,
        "D": D,
        "threshold": threshold,
        "conversion_rate": conversion_rate,
        "L": L,
        "K": K,
        "pde_multiple": pde_multiple,
        "boundary": boundary,
        "total_time": total_time,
        "dt": dt,
    }

    return engine, meta


# ============================================================================
# INITIAL CONDITIONS
# ============================================================================
def initial_conditions(domain: Domain):
    """Half-domain initialisation: A left quarter, B right quarter."""
    K = domain.K
    init_ssa = np.zeros((2, K), dtype=int)
    init_pde = np.zeros((2, domain.n_pde), dtype=float)

    initial_particle_mass = 10
    init_ssa[0, : K // 4] = initial_particle_mass  # A in left quarter
    init_ssa[1, 3 * K // 4 :] = initial_particle_mass  # B in right quarter

    return init_ssa, init_pde


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    engine, meta = build_engine()
    init_ssa, init_pde = initial_conditions(engine.domain)

    print("\nRunning hybrid A⇌B simulation...")
    res = engine.run_repeats(
        initial_ssa=init_ssa,
        initial_pde=init_pde,
        time=meta["total_time"],
        dt=meta["dt"],
        repeats=100,
        seed=1,
    )
    print("Simulation complete.")

    # Save all results in one portable NPZ
    save_path = "data/ab_switch_mean.npz"
    save_npz(res, save_path, meta=meta)
    print(f"\nSaved results + metadata to: {save_path}")

    # Animate
    cfg = AnimationConfig(
        stride=20,
        interval_ms=25,
        threshold_particles=meta["threshold"],
        title="Hybrid Simulation: A ⇌ B",
        mass_plot_mode="per_species",
    )
    animate_results(res, cfg=cfg)

    # Plot static mass time series
    plot_mass_time_series(res, plot_mode="per_species")


if __name__ == "__main__":
    main()