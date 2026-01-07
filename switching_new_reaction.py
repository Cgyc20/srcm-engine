"""
Two-species hybrid diffusion-reaction simulation (user API)
==========================================================

System:
    A ⇌ B
    A -> B  with rate alpha
    B -> A  with rate beta

Spatially resolved with diffusion (PDE + SSA hybrid).

Author: Charlie Cameron
Date: January 2026
"""

import numpy as np

from srcm_engine.core import HybridModel
from srcm_engine.results.io import save_npz
from srcm_engine.animation_util import (
    animate_results,
    AnimationConfig,
    plot_mass_time_series,
)


# ============================================================================
# BUILD MODEL (USER API)
# ============================================================================
def build_model():
    # --- parameters ---
    L = 10.0
    K = 40
    pde_multiple = 8
    boundary = "zero-flux"

    total_time = 30.0
    dt = 0.006

    # Diffusion coefficient (same for A and B here)
    D = 0.1

    # Reaction rates
    alpha = 0.01
    beta = 0.01

    # Conversion parameters
    threshold = 4
    conversion_rate = 1.0

    # --- model ---
    m = HybridModel(species=["A", "B"])

    m.domain(L=L, K=K, pde_multiple=pde_multiple, boundary=boundary)
    m.diffusion(A=D, B=D)
    m.conversion(threshold=threshold, rate=conversion_rate)

    # PDE reaction terms, but in the user-friendly signature:
    #   lambda A, B, r: (dA, dB)
    m.reaction_terms(
        lambda A, B, r: (
            r["beta"] * B - r["alpha"] * A,   # dA
            r["alpha"] * A - r["beta"] * B,   # dB
        )
    )

    # Macroscopic reactions → auto-decomposed hybrid reactions

        # Macroscopic reactions → auto-decomposed hybrid reactions
    m.add_reaction({"A": 1}, {"B": 1}, rate_name="alpha")
    m.add_reaction({"B": 1}, {"A": 1}, rate_name="beta")

    # Build engine with named rates
    m.build(rates={"alpha": alpha, "beta": beta})


    # Metadata for saving
    meta = {
        "model": "A⇌B",
        "alpha": float(alpha),
        "beta": float(beta),
        "D": float(D),
        "threshold": int(threshold),
        "conversion_rate": float(conversion_rate),
        "L": float(L),
        "K": int(K),
        "pde_multiple": int(pde_multiple),
        "boundary": str(boundary),
        "total_time": float(total_time),
        "dt": float(dt),
        "hybrid_labels": m.hybrid_labels(),
    }

    return m, meta


# ============================================================================
# INITIAL CONDITIONS (USER PROVIDES NUMPY ARRAYS)
# ============================================================================
def initial_conditions(K: int, n_pde: int):
    """Half-domain initialisation: A left quarter, B right quarter."""
    init_ssa = np.zeros((2, K), dtype=int)
    init_pde = np.zeros((2, n_pde), dtype=float)

    initial_particle_mass = 10
    init_ssa[0, : K // 4] = initial_particle_mass          # A in left quarter
    init_ssa[1, 3 * K // 4 :] = initial_particle_mass      # B in right quarter

    return init_ssa, init_pde


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    m, meta = build_model()
    m.describe_reactions()
    K = m.domain_obj.K
    n_pde = m.domain_obj.n_pde
    init_ssa, init_pde = initial_conditions(K, n_pde)

    print("\nRunning hybrid A⇌B simulation...")
    res = m.run_repeats(
        init_ssa,
        init_pde,
        time=meta["total_time"],
        dt=meta["dt"],
        repeats=1000,
        seed=1,
        parallel=True,
        n_jobs=-1,
        progress=True,
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
