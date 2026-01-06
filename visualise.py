"""
Visualise a previously-saved SRCMEngine run.

This DOES NOT run a simulation.
It only loads saved results (created by save_results) and animates/plots them.

Usage:
  python check_visualise_saved.py

Or with a custom prefix:
  python check_visualise_saved.py data/si_hybrid_mean
"""

from __future__ import annotations

import sys

from srcm_engine import load_results
from srcm_engine.animation_util import animate_results, AnimationConfig, plot_mass_time_series


def main(prefix: str):
    # Load saved simulation results
    res, meta = load_results(prefix)

    
    print("Loaded results:")
    print("  prefix:", prefix)
    print("  species:", res.species)
    print("  time steps:", len(res.time))
    print("  ssa shape:", res.ssa.shape)
    print("  pde shape:", res.pde.shape)
    print("  domain:", f"L={res.domain.length}, K={res.domain.K}, pde_multiple={res.domain.pde_multiple}")
    print("Threshold" f"{meta.get("threshold_particles")}")
    # --- Animation config ---
    cfg = AnimationConfig(
        stride=20,                 # show every 20th frame
        interval_ms=20,
        threshold_particles=10,   # set to res.domain? no; set explicitly if you want
        title="SRCM – Loaded Results",
        mass_plot_mode="per_species",  # "single" | "per_species" | "none"
        blit=False,
    )

    # If you saved threshold in conversion params somewhere else, set it here manually:
    # cfg.threshold_particles = 20

    # Animate
    animate_results(res, cfg=cfg)

    # Optional: static mass plots (pick one)
    plot_mass_time_series(res, plot_mode="per_species")
    # plot_mass_time_series(res, plot_mode="single")
    # plot_mass_time_series(res, plot_mode="comparison")


if __name__ == "__main__":
    # default prefix (matches your earlier script)
    prefix = "data/si_hybrid_mean"

    # allow overriding from CLI
    if len(sys.argv) >= 2:
        prefix = sys.argv[1]

    main(prefix)
