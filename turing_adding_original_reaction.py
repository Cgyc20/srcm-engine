"""
End-to-end Turing-style hybrid model using the SRCMEngine package,
with cosine-perturbed steady-state initial conditions (steady-state + bump).

Run:
  python check_turing_engine_bump.py
"""

import numpy as np

from srcm_engine import (
    Domain,
    ConversionParams,
    HybridReactionSystem,
    SRCMEngine,
)
from srcm_engine.results.io import save_npz, load_npz

# Optional plotting
from srcm_engine.animation_util import AnimationConfig, animate_results, plot_mass_time_series


# ============================================================================
# PDE reaction terms (macroscopic)
# ============================================================================
def turing_pde_terms(C: np.ndarray, rates: dict) -> np.ndarray:
    """
    Reaction-only PDE terms.

    C: (n_species, n_pde_points)
    Interpreting C as integrated mass per PDE cell (not concentration).
    """
    U, V = C[0], C[1]
    r_11 = float(rates["r_11"])
    r_12 = float(rates["r_12"])
    r_2 = float(rates["r_2"])
    r_3 = float(rates["r_3"])

    dU = r_11 * U**2 - r_2 * U * V
    dV = r_12 * U**2 - r_3 * V
    return np.array([dU, dV])


def build_engine():
    # ----------------------------
    # Controls
    # ----------------------------
    N_COMPARTMENTS = 8
    OMEGA = 500
    PERT_A = 0.07

    total_time = 100.0
    dt = 0.001

    # ----------------------------
    # Rates ("bar" parameters)
    # ----------------------------
    bar_r_11 = 1.0
    bar_r_12 = 1.0
    bar_r_2 = 2.0
    bar_r_3 = 0.6

    # Rescale like SSA code
    r_11 = bar_r_11 / OMEGA
    r_12 = bar_r_12 / OMEGA
    r_2 = bar_r_2 / OMEGA
    r_3 = bar_r_3

    # ----------------------------
    # Domain / diffusion
    # ----------------------------
    L = 1.0
    K = N_COMPARTMENTS
    pde_multiple = 4
    boundary = "zero-flux"

    Du = 2.782559e-03
    Dv = Du * 24.7

    diffusion_rates = {"U": float(Du), "V": float(Dv)}
    reaction_rates = {"r_11": float(r_11), "r_12": float(r_12), "r_2": float(r_2), "r_3": float(r_3)}

    domain = Domain(length=L, n_ssa=K, pde_multiple=pde_multiple, boundary=boundary)

    # Conversion
    threshold_particles = 80
    conversion_rate = 1.0
    conversion = ConversionParams(threshold=threshold_particles, rate=conversion_rate)

    # =========================================================================
    # Hybrid reactions: AUTO-DECOMPOSED (no manual add_hybrid_reaction calls)
    # =========================================================================
    reactions = HybridReactionSystem(species=["U", "V"])

    # Optional: store macroscopic reactions for documentation/reference only
    reactions.add_reaction({"U": 2}, {"U": 3}, rate=bar_r_11)           # 2U -> 3U
    reactions.add_reaction({"U": 2}, {"U": 2, "V": 1}, rate=bar_r_12)   # 2U -> 2U + V
    reactions.add_reaction({"U": 1, "V": 1}, {"V": 1}, rate=bar_r_2)    # U + V -> V
    reactions.add_reaction({"V": 1}, {}, rate=bar_r_3)                  # V -> 0

    # Actual hybrid set used by the engine (rate_name matches reaction_rates keys)
    reactions.add_reaction_original({"U": 2}, {"U": 3}, rate=r_11, rate_name="r_11")
    reactions.add_reaction_original({"U": 2}, {"U": 2, "V": 1}, rate=r_12, rate_name="r_12")
    reactions.add_reaction_original({"U": 1, "V": 1}, {"V": 1}, rate=r_2, rate_name="r_2")
    reactions.add_reaction_original({"V": 1}, {}, rate=r_3, rate_name="r_3")

    reactions.describe()

    engine = SRCMEngine(
        reactions=reactions,
        pde_reaction_terms=turing_pde_terms,
        diffusion_rates=diffusion_rates,
        domain=domain,
        conversion=conversion,
        reaction_rates=reaction_rates,
    )

    meta = {
        # run controls
        "total_time": float(total_time),
        "dt": float(dt),
        "repeats": None,
        "seed": None,

        # matching params
        "omega": float(OMEGA),
        "n_compartments": int(N_COMPARTMENTS),
        "perturbation_amplitude": float(PERT_A),

        # rates
        "bar_r_11": float(bar_r_11),
        "bar_r_12": float(bar_r_12),
        "bar_r_2": float(bar_r_2),
        "bar_r_3": float(bar_r_3),
        "reaction_rates": dict(reaction_rates),

        # diffusion
        "Du": float(Du),
        "Dv": float(Dv),

        # conversion
        "threshold_particles": float(threshold_particles),
        "conversion_rate": float(conversion_rate),

        # domain
        "L": float(L),
        "K": int(K),
        "pde_multiple": int(pde_multiple),
        "boundary": str(boundary),

        # bookkeeping
        "model": "Turing_hybrid_SRCMEngine_bump",
        "species": ["U", "V"],
        "hybrid_labels": [hr.label for hr in reactions.hybrid_reactions],
    }

    return engine, meta


def initial_conditions(domain: Domain, meta: dict):
    """
    Cosine-perturbed steady state (SSA only), matching your non-hybrid SSA script.

    Returns:
      init_ssa: (2, K) ints
      init_pde: (2, n_pde) floats (all zeros)
    """
    K = int(meta["n_compartments"])
    omega = float(meta["omega"])
    A = float(meta["perturbation_amplitude"])
    L = float(meta["L"])

    bar_r_11 = float(meta["bar_r_11"])
    bar_r_12 = float(meta["bar_r_12"])
    bar_r_2 = float(meta["bar_r_2"])
    bar_r_3 = float(meta["bar_r_3"])

    h = L / K

    # Dimensionless steady states
    dim_U_ss = bar_r_11 * bar_r_3 / (bar_r_12 * bar_r_2)
    dim_V_ss = (bar_r_11**2) * bar_r_3 / (bar_r_12 * (bar_r_2**2))

    # Convert to compartment mass (like your SSA code)
    U_ss_mass = dim_U_ss * omega * h
    V_ss_mass = dim_V_ss * omega * h

    # Compartment centres x in [0,1]
    x = (np.arange(K) + 0.5) / K
    bump = 1.0 + A * np.cos(2.0 * np.pi * (x - 0.5))

    U0 = np.round(U_ss_mass * bump).astype(int)
    V0 = np.round(V_ss_mass * bump).astype(int)

    U0 = np.clip(U0, 0, None)
    V0 = np.clip(V0, 0, None)

    init_ssa = np.zeros((2, domain.K), dtype=int)
    init_ssa[0, :] = U0
    init_ssa[1, :] = V0

    # No PDE initial mass
    init_pde = np.zeros((2, domain.n_pde), dtype=float)

    return init_ssa, init_pde


def main():
    engine, meta = build_engine()

    repeats = 50
    seed = 1
    meta["repeats"] = int(repeats)
    meta["seed"] = int(seed)

    init_ssa, init_pde = initial_conditions(engine.domain, meta)

    print("Running...")
    res_mean = engine.run_repeats(
        init_ssa,
        init_pde,
        time=float(meta["total_time"]),
        dt=float(meta["dt"]),
        repeats=repeats,
        seed=seed,
        parallel=True,
        n_jobs=-1,
        progress=True,
    )

    out_npz = "data/turing_hybrid_mean_bump.npz"
    save_npz(res_mean, out_npz, meta=meta)
    print(f"\nSaved results + metadata to: {out_npz}")

    loaded_res, loaded_meta = load_npz(out_npz)
    print("\nLoaded OK.")
    print("Metadata check:")
    print("  omega:", loaded_meta.get("omega"))
    print("  n_compartments:", loaded_meta.get("n_compartments"))
    print("  perturbation_amplitude:", loaded_meta.get("perturbation_amplitude"))
    print("  reaction_rates:", loaded_meta.get("reaction_rates"))
    print("  dt:", loaded_meta.get("dt"))

    thr = loaded_meta.get("threshold_particles", None)
    cfg = AnimationConfig(
        stride=20,
        interval_ms=20,
        threshold_particles=thr,
        title="SRCM – Turing hybrid (steady state + cosine bump)",
        mass_plot_mode="per_species",
    )
    animate_results(loaded_res, cfg=cfg)
    plot_mass_time_series(loaded_res, plot_mode="per_species")


if __name__ == "__main__":
    main()
