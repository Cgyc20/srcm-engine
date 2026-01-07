import numpy as np
from srcm_engine.core import HybridModel
from srcm_engine.results.io import save_npz, load_npz
from srcm_engine.animation_util import AnimationConfig, animate_results, plot_mass_time_series

# -----------------------------
# Build model
# -----------------------------
m = HybridModel(species=["A", "B"])

m.domain(L=10.0, K=40, pde_multiple=8, boundary="zero-flux")
m.diffusion(A=0.1, B=0.1)
m.conversion(threshold=4, rate=1.0)

m.reaction_terms(lambda A, B, r: (
    r["beta"] * B - r["alpha"] * A,
    r["alpha"] * A - r["beta"] * B,
))

m.add_reaction({"A": 1}, {"B": 1}, rate_name="alpha")
m.add_reaction({"B": 1}, {"A": 1}, rate_name="beta")

m.build(rates={"alpha": 0.01, "beta": 0.01})
m.describe_reactions()

# -----------------------------
# Initial conditions
# -----------------------------
init_ssa = np.zeros((2, 40), dtype=int)
init_pde = np.zeros((2, 40 * 8), dtype=float)

# Example IC: A in left quarter, B in right quarter (optional)
init_ssa[0, : 40 // 4] = 10
init_ssa[1, 3 * 40 // 4 :] = 10

# -----------------------------
# Run simulation
# -----------------------------
total_time = 30.0
dt = 0.006
repeats = 100
seed = 1

res = m.run_repeats(
    init_ssa,
    init_pde,
    time=total_time,
    dt=dt,
    repeats=repeats,
    seed=seed,
    parallel=True,
    n_jobs=-1,
    progress=True,
)

# -----------------------------
# Save results + metadata
# -----------------------------
meta = m.metadata()
meta.update({
    "model": "A<->B",
    "total_time": float(total_time),
    "dt": float(dt),
    "repeats": int(repeats),
    "seed": int(seed),
})

out_npz = "ab_switch_mean.npz"
save_npz(res, out_npz, meta=meta)
print("Saved:", out_npz)

# -----------------------------
# (Optional) Load back to verify
# -----------------------------
loaded_res, loaded_meta = load_npz(out_npz)
print("Loaded meta keys:", sorted(list(loaded_meta.keys()))[:10], "...")

# -----------------------------
# Animate + plot
# -----------------------------
cfg = AnimationConfig(
    stride=20,
    interval_ms=25,
    threshold_particles=meta["threshold_particles"],
    title="Hybrid Simulation: A ⇌ B",
    mass_plot_mode="per_species",
)

animate_results(loaded_res, cfg=cfg)
plot_mass_time_series(loaded_res, plot_mode="per_species")
