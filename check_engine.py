"""
End-to-end SI-style hybrid model using the new SRCMEngine package.

This mirrors the old hardcoded SI_hybrid_model + main script.

Run:
  python check_si_engine.py
"""

import numpy as np

from srcm_engine import (
    Domain,
    ConversionParams,
    HybridReactionSystem,
    SRCMEngine,
    save_results,
    load_results,
)


# ============================================================================
# PDE reaction terms (macroscopic)
# ============================================================================
def si_pde_terms(C: np.ndarray, rates: dict) -> np.ndarray:
    """
    SI-style macroscopic PDE reaction terms matching the old RHS:

      dU = + beta * U*V - alpha * U
      dV = - beta * U*V          (+ mu term was removed in your code)

    C shape: (2, Npde)
    """
    U, V = C[0], C[1]
    alpha = rates["alpha"]
    beta = rates["beta"]

    dU = beta * U * V - alpha * U
    dV = -beta * U * V

    return np.array([dU, dV])


def main():
    # ============================================================================
    # SIMULATION PARAMETERS - MATCHING YOUR OLD SCRIPT
    # ============================================================================
    bar_alpha = 0.01
    bar_beta = 0.1
    bar_mu = 0.01
    bar_D = 0.001
    omega = 50

    total_time = 100.0
    dt = 0.01
    threshold = 20
    conversion_rate = 10.0

    L = 10.0
    K = 18
    pde_multiple = 8

    boundary = "zero-flux"

    # -------------------------------------------------------------
    # Rescale (same as your old script)
    # -------------------------------------------------------------
    alpha = bar_alpha
    beta = bar_beta / omega
    mu = bar_mu * omega
    D = bar_D * (L**2)

    print(f"System size ω = {omega}")
    print(f"Rescaled rates: alpha={alpha}, beta={beta}, mu={mu}, D={D}")

    # ============================================================================
    # DOMAIN + CONVERSION
    # ============================================================================
    domain = Domain(length=L, n_ssa=K, pde_multiple=pde_multiple, boundary=boundary)
    conversion = ConversionParams(threshold=threshold, rate=conversion_rate)

    # ============================================================================
    # REACTIONS: we’ll define only the HYBRID channels the engine uses
    # ============================================================================
    reactions = HybridReactionSystem(species=["U", "V"])

    # Optional “pure” reactions for documentation (not used by engine)
    reactions.add_reaction({"U": 1}, {}, rate=alpha)          # U -> 0
    reactions.add_reaction({"U": 1, "V": 1}, {"U": 2}, rate=beta)  # U+V -> 2U
    reactions.add_reaction({}, {"V": 1}, rate=mu)             # 0 -> V  (SSA-only below)

    # --- Hybrid channels (match your old propensity_calculation + event updates) ---
    # Blocks 6..10 in your old model become hybrid reactions here.

    # R1: D_U + D_V -> 2 D_U
    reactions.add_hybrid_reaction(
        reactants={"D_U": 1, "D_V": 1},
        products={"D_U": 2},
        propensity=lambda D, C, r, h: r["beta"] * D["U"] * D["V"] / h,
        state_change={"D_U": +1, "D_V": -1},
        label="R1_DU_DV_to_2DU",
        description="D^U + D^V -> 2 D^U",
    )

    # R2: C_U + D_V -> 2 C_U   (PDE U gains 1 particle mass, SSA V loses 1)
    reactions.add_hybrid_reaction(
        reactants={"C_U": 1, "D_V": 1},
        products={"C_U": 2},
        propensity=lambda D, C, r, h: r["beta"] * C["U"] * D["V"] / h,
        state_change={"C_U": +1, "D_V": -1},
        label="R2_CU_DV_to_2CU",
        description="C^U + D^V -> 2 C^U  (PDE U +1 particle mass, SSA V -1)",
    )

    # R3: D_U + C_V -> 2 D_U   (SSA U +1, PDE V -1 particle mass)
    reactions.add_hybrid_reaction(
        reactants={"D_U": 1, "C_V": 1},
        products={"D_U": 2},
        propensity=lambda D, C, r, h: r["beta"] * D["U"] * C["V"] / h,
        state_change={"D_U": +1, "C_V": -1},
        label="R3_DU_CV_to_2DU",
        description="D^U + C^V -> 2 D^U  (SSA U +1, PDE V -1 particle mass)",
    )

    # R4: D_U -> 0
    reactions.add_hybrid_reaction(
        reactants={"D_U": 1},
        products={},
        propensity=lambda D, C, r, h: r["alpha"] * D["U"],
        state_change={"D_U": -1},
        label="R4_DU_to_0",
        description="D^U -> 0",
    )

    # R5: 0 -> D_V   (your old propensity: mu * h)
    reactions.add_hybrid_reaction(
        reactants={},
        products={"D_V": 1},
        propensity=lambda D, C, r, h: r["mu"] * h,
        state_change={"D_V": +1},
        label="R5_0_to_DV",
        description="0 -> D^V  (SSA birth in compartment at rate mu*h)",
    )

    # ============================================================================
    # ENGINE
    # ============================================================================
    engine = SRCMEngine(
        reactions=reactions,
        pde_reaction_terms=si_pde_terms,
        diffusion_rates={"U": D, "V": D},  # your old model used one diffusion rate for both
        domain=domain,
        conversion=conversion,
        reaction_rates={"alpha": alpha, "beta": beta, "mu": mu},
    )

    # ============================================================================
    # INITIAL CONDITIONS (same layout as before)
    # ============================================================================
    initial_particle_mass = 50

    init_ssa = np.zeros((2, domain.K), dtype=int)
    init_ssa[0, : domain.K // 4] = initial_particle_mass
    init_ssa[1, 3 * domain.K // 4 :] = initial_particle_mass

    # your old code started PDE as zeros
    init_pde = np.zeros((2, domain.n_pde), dtype=float)

    print("\nInitial totals:")
    print("  U particles:", int(init_ssa[0].sum()))
    print("  V particles:", int(init_ssa[1].sum()))
    print("  Total:", int(init_ssa.sum()))
    print("Running...")

    # ============================================================================
    # RUN REPEATS + SAVE
    # ============================================================================
    repeats = 50
    res_mean = engine.run_repeats(
        initial_ssa=init_ssa,
        initial_pde=init_pde,
        time=total_time,
        dt=dt,
        repeats=repeats,
        seed=1,
    )

    print("Done.")
    print("Shapes:")
    print("  ssa:", res_mean.ssa.shape)
    print("  pde:", res_mean.pde.shape)
    print("  combined:", res_mean.combined().shape)

    out_prefix = "data/si_hybrid_mean"
    save_results(res_mean, out_prefix)
    loaded = load_results(out_prefix)
    print("Saved + loaded OK:", loaded.ssa.shape, loaded.pde.shape)

    print("\nAll good ✅")

   

    res = engine.run_repeats(
        initial_ssa=init_ssa,
        initial_pde=init_pde,
        time=50.0,
        dt=0.01,
        repeats=5,
        seed=1,
    )

    from srcm_engine.animation_util import animate_results, AnimationConfig
    cfg = AnimationConfig(
        stride=20,
        interval_ms=20,
        threshold_particles=engine.conversion.threshold,
        title="SRCM – Turing hybrid",
    )
    animate_results(res, cfg=cfg)


if __name__ == "__main__":
    main()
