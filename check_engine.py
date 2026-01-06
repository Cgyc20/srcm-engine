"""
End-to-end SI-style hybrid model using the SRCMEngine package.
Saves a single self-contained NPZ with full metadata.

Run:
  python check_si_engine.py
"""

import numpy as np

from srcm_engine import (
    Domain,
    ConversionParams,
    HybridReactionSystem,
    SRCMEngine,
)
from srcm_engine.results.io import save_npz, load_npz  # <-- ensure you import the updated functions

# Optional plotting (only if you want to preview immediately)
from srcm_engine.animation_util import AnimationConfig, animate_results, plot_mass_time_series


# ============================================================================
# PDE reaction terms (macroscopic)
# ============================================================================
def si_pde_terms(C: np.ndarray, rates: dict) -> np.ndarray:
    """
    dU = + beta*U*V - alpha*U
    dV = - beta*U*V
    """
    U, V = C[0], C[1]
    alpha = float(rates["alpha"])
    beta = float(rates["beta"])
    dU = beta * U * V - alpha * U
    dV = -beta * U * V
    return np.array([dU, dV])


def build_engine():
    # --- parameters ---
    bar_alpha = 0.01
    bar_beta = 0.1
    bar_mu = 0.01
    bar_D = 0.001
    omega = 50

    total_time = 100.0
    dt = 0.01
    threshold_particles = 20
    conversion_rate = 10.0

    L = 10.0
    K = 18
    pde_multiple = 8
    boundary = "zero-flux"

    # rescale
    alpha = bar_alpha
    beta = bar_beta / omega
    mu = bar_mu * omega
    D = bar_D * (L**2)

    domain = Domain(length=L, n_ssa=K, pde_multiple=pde_multiple, boundary=boundary)
    conversion = ConversionParams(threshold=threshold_particles, rate=conversion_rate)

    # --- reactions ---
    reactions = HybridReactionSystem(species=["U", "V"])

    # Hybrid channels
    reactions.add_hybrid_reaction(
        reactants={"D_U": 1, "D_V": 1},
        products={"D_U": 2},
        propensity=lambda Dloc, Cloc, r, h: r["beta"] * Dloc["U"] * Dloc["V"] / h,
        state_change={"D_U": +1, "D_V": -1},
        label="R1",
        description="D^U + D^V -> 2D^U",
    )

    reactions.add_hybrid_reaction(
        reactants={"C_U": 1, "D_V": 1},
        products={"C_U": 2},
        propensity=lambda Dloc, Cloc, r, h: r["beta"] * Cloc["U"] * Dloc["V"] / h,
        state_change={"C_U": +1, "D_V": -1},
        label="R2",
        description="C^U + D^V -> 2C^U",
    )

    reactions.add_hybrid_reaction(
        reactants={"D_U": 1, "C_V": 1},
        products={"D_U": 2},
        propensity=lambda Dloc, Cloc, r, h: r["beta"] * Dloc["U"] * Cloc["V"] / h,
        state_change={"D_U": +1, "C_V": -1},
        label="R3",
        description="D^U + C^V -> 2D^U",
    )

    reactions.add_hybrid_reaction(
        reactants={"D_U": 1},
        products={},
        propensity=lambda Dloc, Cloc, r, h: r["alpha"] * Dloc["U"],
        state_change={"D_U": -1},
        label="R4",
        description="D^U -> 0",
    )

    reactions.add_hybrid_reaction(
        reactants={},
        products={"D_V": 1},
        propensity=lambda Dloc, Cloc, r, h: r["mu"] * h,
        state_change={"D_V": +1},
        label="R5",
        description="0 -> D^V",
    )

    engine = SRCMEngine(
        reactions=reactions,
        pde_reaction_terms=si_pde_terms,
        diffusion_rates={"U": D, "V": D},
        domain=domain,
        conversion=conversion,
        reaction_rates={"alpha": alpha, "beta": beta, "mu": mu},
    )

    # Metadata you want saved WITH the results
    meta = {
        # run controls
        "total_time": float(total_time),
        "dt": float(dt),
        "repeats": None,  # fill at runtime
        "seed": None,     # fill at runtime

        # conversion
        "threshold_particles": float(threshold_particles),
        "conversion_rate": float(conversion_rate),

        # nondimensional knobs
        "omega": float(omega),
        "bar_alpha": float(bar_alpha),
        "bar_beta": float(bar_beta),
        "bar_mu": float(bar_mu),
        "bar_D": float(bar_D),

        # actual rates used
        "alpha": float(alpha),
        "beta": float(beta),
        "mu": float(mu),
        "D": float(D),

        # domain
        "L": float(L),
        "K": int(K),
        "pde_multiple": int(pde_multiple),
        "boundary": str(boundary),

        # bookkeeping
        "model": "SI_hybrid_SRCMEngine",
        "species": ["U", "V"],
    }

    return engine, meta


def initial_conditions(domain: Domain):
    initial_particle_mass = 50
    init_ssa = np.zeros((2, domain.K), dtype=int)
    init_ssa[0, : domain.K // 4] = initial_particle_mass
    init_ssa[1, 3 * domain.K // 4 :] = initial_particle_mass
    init_pde = np.zeros((2, domain.n_pde), dtype=float)
    return init_ssa, init_pde


def main():
    engine, meta = build_engine()
    init_ssa, init_pde = initial_conditions(engine.domain)

    repeats = 5
    seed = 1
    meta["repeats"] = int(repeats)
    meta["seed"] = int(seed)

    print("Running...")
    res_mean = engine.run_repeats(
        initial_ssa=init_ssa,
        initial_pde=init_pde,
        time=float(meta["total_time"]),
        dt=float(meta["dt"]),
        repeats=repeats,
        seed=seed,
    )

    out_npz = "data/si_hybrid_mean.npz"

    # ✅ New API: meta dict is stored as meta_json in the npz
    save_npz(res_mean, out_npz, meta=meta)
    print(f"\nSaved results + metadata to: {out_npz}")

    loaded_res, loaded_meta = load_npz(out_npz)
    print("\nLoaded OK.")
    print("Metadata check:")
    print("  threshold_particles:", loaded_meta.get("threshold_particles"))
    print("  conversion_rate:", loaded_meta.get("conversion_rate"))
    print("  beta:", loaded_meta.get("beta"))
    print("  K:", loaded_meta.get("K"))
    print("  dt:", loaded_meta.get("dt"))

    # Optional: animate using loaded threshold
    thr = loaded_meta.get("threshold_particles", None)
    cfg = AnimationConfig(
        stride=20,
        interval_ms=20,
        threshold_particles=thr,
        title="SRCM – SI hybrid (loaded from npz)",
        mass_plot_mode="per_species",
    )
    animate_results(loaded_res, cfg=cfg)

    # Optional: static mass plot
    plot_mass_time_series(loaded_res, plot_mode="per_species")


if __name__ == "__main__":
    main()