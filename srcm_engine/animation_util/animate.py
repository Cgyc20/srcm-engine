from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec

from ..results import SimulationResults



# ------------------------------------------------------------
# Styling helpers
# ------------------------------------------------------------
def setup_cinematic_style():
    plt.style.use("dark_background")

    colors = {
        "species_a": "#4DC3FF",
        "species_b": "#FF4D6D",
        "species_a_light": "#7BD5FF",
        "species_b_light": "#FF7B94",
        "threshold": "#C77DFF",
        "background": "#0A0A12",
        "grid": "#1E1E2E",
        "text": "#E8E8F0",
        "ssa_total": "#00FF88",
        "pde_total": "#FFAA00",
    }

    mpl.rcParams["figure.figsize"] = [16, 9]
    mpl.rcParams["figure.dpi"] = 100
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["axes.titlepad"] = 20

    return colors


def _validate_two_species(species: Sequence[str]) -> None:
    if len(species) != 2:
        raise ValueError("This animation helper currently supports exactly 2 species.")


# ------------------------------------------------------------
# Core computations
# ------------------------------------------------------------
def ssa_to_concentration_on_pde_grid(
    ssa: np.ndarray,
    domain_h: float,
    pde_multiple: int,
) -> np.ndarray:
    """
    Convert SSA counts per compartment into a concentration field on the PDE grid
    by distributing uniformly over the pde_multiple fine cells in each compartment.

    Parameters
    ----------
    ssa : (n_species, K, T)
        SSA particle counts per compartment.
    domain_h : float
        SSA compartment size.
    pde_multiple : int
        PDE cells per SSA compartment.

    Returns
    -------
    ssa_conc_on_pde : (n_species, Npde, T)
    """
    n_species, K, T = ssa.shape
    Npde = K * pde_multiple

    out = np.zeros((n_species, Npde, T), dtype=float)
    # Each SSA particle corresponds to concentration 1/h over the compartment,
    # and we spread uniformly across the pde_multiple cells.
    for i in range(K):
        s = i * pde_multiple
        e = s + pde_multiple
        out[:, s:e, :] = (ssa[:, i:i+1, :] / domain_h)  # broadcast across pde_multiple
    return out


def total_mass_time_series(
    ssa: np.ndarray,
    pde: np.ndarray,
    h: float,
    dx: float,
) -> dict:
    """
    Mass (particles) per time:
      SSA mass is sum of counts (already particles)
      PDE mass is integral (sum * dx) per species
      Combined mass uses (PDE + SSA/h on fine grid) integrated
    """
    # ssa: (n_species, K, T)
    # pde: (n_species, Npde, T)
    n_species, _, T = ssa.shape

    ssa_mass = np.sum(ssa, axis=1)  # (n_species, T)
    pde_mass = np.sum(pde, axis=1) * dx  # (n_species, T)

    # Combined: compute fine-grid SSA concentration + PDE concentration, integrate
    # (this matches your old combined_grid logic)
    # We avoid building full combined_grid unless needed:
    # combined_mass = ∫(PDE) + ∑ SSA particles  (should equal pde_mass + ssa_mass if consistent)
    combined_mass = pde_mass + ssa_mass  # in particle units

    return {
        "ssa_mass": ssa_mass,
        "pde_mass": pde_mass,
        "combined_mass": combined_mass,
        "total_ssa": np.sum(ssa_mass, axis=0),
        "total_pde": np.sum(pde_mass, axis=0),
        "total_combined": np.sum(combined_mass, axis=0),
    }


# ------------------------------------------------------------
# Animation
# ------------------------------------------------------------
@dataclass
class AnimationConfig:
    stride: int = 10                 # show every Nth frame
    interval_ms: int = 30
    show_threshold: bool = True
    threshold_particles: Optional[float] = None  # threshold in particles per compartment
    title: str = "SRCM Hybrid Simulation"
    blit: bool = False


def animate_results(res: SimulationResults, cfg: Optional[AnimationConfig] = None):
    """
    Create an interactive matplotlib animation from SimulationResults.

    Notes
    -----
    - SSA shown as bars (concentration = count/h)
    - PDE shown as solid lines
    - "combined" shown as dashed lines:
        combined = PDE + SSA contribution mapped to PDE grid

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
    """
    if cfg is None:
        cfg = AnimationConfig()

    _validate_two_species(res.species)

    colors = setup_cinematic_style()

    time = res.time
    ssa = res.ssa
    pde = res.pde

    # Shapes
    n_species, K, T = ssa.shape
    _, Npde, _ = pde.shape

    h = float(res.domain.h)
    dx = float(res.domain.dx)
    pde_multiple = int(res.domain.pde_multiple)
    L = float(res.domain.length)

    # Coordinates (scaled 0..1 like your old plots)
    ssa_x = (np.arange(K) * h) / L
    pde_x = (np.linspace(0.0, L, Npde)) / L
    bar_w = h / L

    # SSA concentrations
    ssa_conc = ssa.astype(float) / h  # (n_species, K, T)

    # SSA mapped to PDE grid (for combined curve)
    ssa_conc_on_pde = ssa_to_concentration_on_pde_grid(ssa.astype(float), h, pde_multiple)  # (n_species, Npde, T)
    combined_conc = pde + ssa_conc_on_pde

    # Mass info
    masses = total_mass_time_series(ssa.astype(float), pde.astype(float), h=h, dx=dx)
    max_mass = float(np.max([masses["total_ssa"].max(), masses["total_pde"].max(), masses["total_combined"].max()]))

    # Plot scaling
    max_conc = float(np.max(combined_conc))
    if max_conc <= 0:
        max_conc = 1.0

    # threshold (concentration units)
    if cfg.threshold_particles is not None:
        conc_threshold = float(cfg.threshold_particles) / h
    else:
        conc_threshold = None

    # ---- Figure layout (main + mass panel) ----
    fig = plt.figure(figsize=(16, 9), facecolor=colors["background"])
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25)

    ax_main = plt.subplot(gs[0])
    ax_mass = plt.subplot(gs[1])

    for ax in (ax_main, ax_mass):
        ax.set_facecolor(colors["background"])
        ax.grid(True, alpha=0.2, color=colors["grid"], linestyle="-", linewidth=0.5)
        ax.tick_params(colors=colors["text"])

    # --- SSA bars ---
    bar_A = ax_main.bar(
        ssa_x, ssa_conc[0, :, 0],
        width=bar_w, align="edge",
        color=colors["species_a_light"], alpha=0.8,
        edgecolor=colors["species_a"], linewidth=0.5, label=f"SSA {res.species[0]}"
    )
    bar_B = ax_main.bar(
        ssa_x, ssa_conc[1, :, 0],
        width=bar_w, align="edge",
        color=colors["species_b_light"], alpha=0.8,
        edgecolor=colors["species_b"], linewidth=0.5, label=f"SSA {res.species[1]}"
    )

    # --- PDE lines ---
    line_pde_A, = ax_main.plot(pde_x, pde[0, :, 0], color=colors["species_a"], linewidth=3.0, label=f"PDE {res.species[0]}")
    line_pde_B, = ax_main.plot(pde_x, pde[1, :, 0], color=colors["species_b"], linewidth=3.0, label=f"PDE {res.species[1]}")

    # --- Combined lines ---
    line_comb_A, = ax_main.plot(pde_x, combined_conc[0, :, 0], color=colors["species_a"], linestyle="--", linewidth=3.0, alpha=0.85, label=f"Combined {res.species[0]}")
    line_comb_B, = ax_main.plot(pde_x, combined_conc[1, :, 0], color=colors["species_b"], linestyle="--", linewidth=3.0, alpha=0.85, label=f"Combined {res.species[1]}")

    # --- Threshold ---
    threshold_line = None
    if cfg.show_threshold and conc_threshold is not None:
        threshold_line = ax_main.axhline(conc_threshold, color=colors["threshold"], linestyle=":", linewidth=2.5, alpha=0.9, label="Threshold")

    # Axis styling
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, max_conc * 1.15)
    ax_main.set_xlabel("Scaled Domain [0,1]", fontsize=13, color=colors["text"])
    ax_main.set_ylabel("Concentration", fontsize=13, color=colors["text"])
    ax_main.set_title(cfg.title, fontsize=18, color=colors["text"], fontweight="bold")

    leg = ax_main.legend(loc="upper right", framealpha=0.9, facecolor=colors["background"], edgecolor=colors["grid"], fontsize=10)
    for t in leg.get_texts():
        t.set_color(colors["text"])

    time_text = ax_main.text(
        0.02, 0.95, "", transform=ax_main.transAxes,
        fontsize=12, color=colors["text"], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor=colors["background"], edgecolor=colors["grid"], alpha=0.9)
    )

    # --- Mass plot ---
    mass_A_line, = ax_mass.plot([], [], color=colors["species_a"], linewidth=2, alpha=0.8, label=f"Mass {res.species[0]}")
    mass_B_line, = ax_mass.plot([], [], color=colors["species_b"], linewidth=2, alpha=0.8, label=f"Mass {res.species[1]}")

    total_comb_line, = ax_mass.plot([], [], color="white", linewidth=3.0, alpha=0.9, label="Total Combined")
    total_ssa_line, = ax_mass.plot([], [], color=colors["ssa_total"], linestyle="--", linewidth=2.2, alpha=0.9, label="Total SSA")
    total_pde_line, = ax_mass.plot([], [], color=colors["pde_total"], linestyle="-.", linewidth=2.2, alpha=0.9, label="Total PDE")

    ax_mass.set_xlim(float(time[0]), float(time[-1]))
    ax_mass.set_ylim(0, max_mass * 1.15)
    ax_mass.set_xlabel("Time", fontsize=13, color=colors["text"])
    ax_mass.set_ylabel("Total Mass (particles)", fontsize=13, color=colors["text"])

    leg2 = ax_mass.legend(loc="upper right", framealpha=0.9, facecolor=colors["background"], edgecolor=colors["grid"], fontsize=9, ncol=2)
    for t in leg2.get_texts():
        t.set_color(colors["text"])

    mass_text = ax_mass.text(
        0.02, 0.90, "", transform=ax_mass.transAxes,
        fontsize=10, color=colors["text"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor=colors["background"], edgecolor=colors["grid"], alpha=0.85)
    )

    frames = list(range(0, T, max(1, int(cfg.stride))))

    def update(frame_idx: int):
        # SSA bars
        for b, hgt in zip(bar_A, ssa_conc[0, :, frame_idx]):
            b.set_height(float(hgt))
        for b, hgt in zip(bar_B, ssa_conc[1, :, frame_idx]):
            b.set_height(float(hgt))

        # PDE / combined
        line_pde_A.set_ydata(pde[0, :, frame_idx])
        line_pde_B.set_ydata(pde[1, :, frame_idx])
        line_comb_A.set_ydata(combined_conc[0, :, frame_idx])
        line_comb_B.set_ydata(combined_conc[1, :, frame_idx])

        # Time text
        time_text.set_text(f"t = {time[frame_idx]:.4g}\nframe = {frame_idx}/{T-1}")

        # Mass traces
        sl = slice(0, frame_idx + 1)
        mass_A_line.set_data(time[sl], masses["combined_mass"][0, sl])
        mass_B_line.set_data(time[sl], masses["combined_mass"][1, sl])
        total_comb_line.set_data(time[sl], masses["total_combined"][sl])
        total_ssa_line.set_data(time[sl], masses["total_ssa"][sl])
        total_pde_line.set_data(time[sl], masses["total_pde"][sl])

        # Mass conservation %
        m0 = float(masses["total_combined"][0])
        mt = float(masses["total_combined"][frame_idx])
        if m0 != 0:
            pct = (mt - m0) / m0 * 100.0
        else:
            pct = 0.0
        mass_text.set_text(f"Mass conservation: {pct:+.4f}%")

        artists = (
            list(bar_A) + list(bar_B)
            + [line_pde_A, line_pde_B, line_comb_A, line_comb_B, time_text]
            + [mass_A_line, mass_B_line, total_comb_line, total_ssa_line, total_pde_line, mass_text]
        )
        if threshold_line is not None:
            artists.append(threshold_line)
        return artists

    ani = FuncAnimation(
        fig, update, frames=frames,
        interval=int(cfg.interval_ms),
        blit=bool(cfg.blit), repeat=True
    )

    plt.tight_layout()
    plt.show()
    return ani
