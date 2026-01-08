from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Literal

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
        "species_c": "#4DFF9E",
        "species_d": "#FFD700",
        "species_a_light": "#7BD5FF",
        "species_b_light": "#FF7B94",
        "species_c_light": "#7BFFB8",
        "species_d_light": "#FFE44D",
        "threshold": "#C77DFF",
        "background": "#0A0A12",
        "grid": "#1E1E2E",
        "text": "#E8E8F0",
        "ssa_total": "#00FF88",
        "pde_total": "#FFAA00",
        "combined_total": "#FFFFFF",
        "mass_ssa_line": "#00FF88",
        "mass_pde_line": "#FFAA00",
        "mass_combined_line": "#4DC3FF",
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


def _species_color(colors: dict, i: int) -> str:
    # supports up to 4 with your palette
    key = f"species_{chr(97 + i)}"
    return colors.get(key, "white")


def _species_light_color(colors: dict, i: int) -> str:
    key = f"species_{chr(97 + i)}_light"
    return colors.get(key, "white")


# ------------------------------------------------------------
# Core computations
# ------------------------------------------------------------
def ssa_to_concentration_on_pde_grid(
    ssa: np.ndarray,
    domain_h: float,
    pde_multiple: int,
) -> np.ndarray:
    """
    ssa shape: (n_species, K, T)
    return:    (n_species, Npde, T)
    """
    n_species, K, T = ssa.shape
    Npde = K * pde_multiple

    out = np.zeros((n_species, Npde, T), dtype=float)
    for i in range(K):
        s = i * pde_multiple
        e = s + pde_multiple
        out[:, s:e, :] = (ssa[:, i : i + 1, :] / domain_h)  # broadcast across fine cells
    return out


def total_mass_time_series(
    ssa: np.ndarray,
    pde: np.ndarray,
    *,
    dx: float,
) -> dict:
    """
    Mass in particle units:
      SSA mass = sum of counts over compartments
      PDE mass = integral sum(C) * dx
      Combined = SSA + PDE
    """
    ssa_mass = np.sum(ssa, axis=1)  # (n_species, T)
    pde_mass = np.sum(pde, axis=1) * dx  # (n_species, T)
    combined_mass = ssa_mass + pde_mass

    return {
        "ssa_mass": ssa_mass,
        "pde_mass": pde_mass,
        "combined_mass": combined_mass,
        "total_ssa": np.sum(ssa_mass, axis=0),
        "total_pde": np.sum(pde_mass, axis=0),
        "total_combined": np.sum(combined_mass, axis=0),
    }


# ------------------------------------------------------------
# Animation Configuration
# ------------------------------------------------------------
@dataclass
class AnimationConfig:
    stride: int = 10
    interval_ms: int = 30
    show_threshold: bool = True
    threshold_particles: Optional[float] = None  # threshold in particles per compartment
    title: str = "SRCM Hybrid Simulation"
    blit: bool = False
    mass_plot_mode: Literal["single", "per_species", "none"] = "single"


def animate_results(res: SimulationResults, cfg: Optional[AnimationConfig] = None):
    if cfg is None:
        cfg = AnimationConfig()

    _validate_two_species(res.species)
    colors = setup_cinematic_style()

    time = res.time
    ssa = res.ssa
    pde = res.pde

    n_species, K, T = ssa.shape
    _, Npde, _ = pde.shape

    h = float(res.domain.h)
    dx = float(res.domain.dx)
    pde_multiple = int(res.domain.pde_multiple)
    L = float(res.domain.length)

    # Coordinates scaled to [0,1]
    ssa_x = (np.arange(K) * h) / L
    pde_x = (np.linspace(0.0, L, Npde)) / L
    bar_w = h / L

    # Concentrations
    ssa_conc = ssa.astype(float) / h  # (n_species, K, T)
    ssa_conc_on_pde = ssa_to_concentration_on_pde_grid(ssa.astype(float), h, pde_multiple)
    combined_conc = pde + ssa_conc_on_pde

    # Mass time-series
    masses = total_mass_time_series(ssa.astype(float), pde.astype(float), dx=dx)

    # scaling
    max_conc = float(np.max(combined_conc))
    if not np.isfinite(max_conc) or max_conc <= 0:
        max_conc = 1.0

    max_mass_per_species = [
        float(np.max([masses["ssa_mass"][i].max(), masses["pde_mass"][i].max(), masses["combined_mass"][i].max()]))
        for i in range(n_species)
    ]
    max_total_mass = float(np.max([masses["total_ssa"].max(), masses["total_pde"].max(), masses["total_combined"].max()]))
    if not np.isfinite(max_total_mass) or max_total_mass <= 0:
        max_total_mass = 1.0

    # threshold in concentration units
    conc_threshold = None
    if cfg.threshold_particles is not None:
        conc_threshold = float(cfg.threshold_particles) / h

    # ------------------------------------------------------------
    # Figure + axes (FIXED: figure created FIRST)
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(16, 9), facecolor=colors["background"])

    ax_main = None
    ax_mass_list: list[plt.Axes] = []

    if cfg.mass_plot_mode == "per_species":
        n_rows = 1 + n_species
        height_ratios = [3] + [1] * n_species
        gs = gridspec.GridSpec(n_rows, 1, height_ratios=height_ratios, hspace=0.30, figure=fig)

        ax_main = fig.add_subplot(gs[0])
        ax_mass_list = [fig.add_subplot(gs[i + 1]) for i in range(n_species)]

    elif cfg.mass_plot_mode == "single":
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25, figure=fig)
        ax_main = fig.add_subplot(gs[0])
        ax_mass_list = [fig.add_subplot(gs[1])]

    else:  # none
        ax_main = fig.add_subplot(111)
        ax_mass_list = []

    # style axes
    for ax in [ax_main] + ax_mass_list:
        ax.set_facecolor(colors["background"])
        ax.grid(True, alpha=0.2, color=colors["grid"], linestyle="-", linewidth=0.5)
        ax.tick_params(colors=colors["text"])

    # ------------------------------------------------------------
    # Main plot artists
    # ------------------------------------------------------------
    bar_A = ax_main.bar(
        ssa_x, ssa_conc[0, :, 0],
        width=bar_w, align="edge",
        color=_species_light_color(colors, 0), alpha=0.8,
        edgecolor=_species_color(colors, 0), linewidth=0.5,
        label=f"SSA {res.species[0]}",
    )
    bar_B = ax_main.bar(
        ssa_x, ssa_conc[1, :, 0],
        width=bar_w, align="edge",
        color=_species_light_color(colors, 1), alpha=0.8,
        edgecolor=_species_color(colors, 1), linewidth=0.5,
        label=f"SSA {res.species[1]}",
    )

    line_pde_A, = ax_main.plot(pde_x, pde[0, :, 0], color=_species_color(colors, 0), linewidth=3.0, label=f"PDE {res.species[0]}")
    line_pde_B, = ax_main.plot(pde_x, pde[1, :, 0], color=_species_color(colors, 1), linewidth=3.0, label=f"PDE {res.species[1]}")

    line_comb_A, = ax_main.plot(pde_x, combined_conc[0, :, 0], color=_species_color(colors, 0), linestyle="--", linewidth=3.0, alpha=0.85, label=f"Combined {res.species[0]}")
    line_comb_B, = ax_main.plot(pde_x, combined_conc[1, :, 0], color=_species_color(colors, 1), linestyle="--", linewidth=3.0, alpha=0.85, label=f"Combined {res.species[1]}")

    threshold_line = None
    if cfg.show_threshold and conc_threshold is not None:
        threshold_line = ax_main.axhline(conc_threshold, color=colors["threshold"], linestyle=":", linewidth=2.5, alpha=0.9, label="Threshold")

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

    # ------------------------------------------------------------
    # Mass plot artists (FIXED: consistent variables)
    # ------------------------------------------------------------
    per_species_mass_lines: list[tuple] = []     # list[(ssa_line, pde_line, comb_line)]
    per_species_mass_texts: list = []

    single_species_comb_lines: list = []         # list[Line2D] for each species combined
    single_total_lines: tuple | None = None      # (total_ssa, total_pde, total_comb)
    single_mass_text = None

    if cfg.mass_plot_mode == "per_species":
        for i, sp in enumerate(res.species):
            axm = ax_mass_list[i]

            ssa_line, = axm.plot([], [], color=colors["mass_ssa_line"], linestyle="--", linewidth=2.0, alpha=0.9, label="SSA Mass")
            pde_line, = axm.plot([], [], color=colors["mass_pde_line"], linestyle="-.", linewidth=2.0, alpha=0.9, label="PDE Mass")
            comb_line, = axm.plot([], [], color=_species_color(colors, i), linewidth=2.5, alpha=0.9, label="Combined Mass")

            per_species_mass_lines.append((ssa_line, pde_line, comb_line))

            axm.set_xlim(float(time[0]), float(time[-1]))
            axm.set_ylim(0, max_mass_per_species[i] * 1.15)
            axm.set_ylabel(f"{sp}\nMass", fontsize=11, color=colors["text"])

            if i == n_species - 1:
                axm.set_xlabel("Time", fontsize=13, color=colors["text"])
            else:
                axm.tick_params(labelbottom=False)

            lg = axm.legend(loc="upper right", framealpha=0.9, facecolor=colors["background"], edgecolor=colors["grid"], fontsize=8)
            for t in lg.get_texts():
                t.set_color(colors["text"])

            txt = axm.text(
                0.02, 0.85, "", transform=axm.transAxes,
                fontsize=9, color=colors["text"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors["background"], edgecolor=colors["grid"], alpha=0.85)
            )
            per_species_mass_texts.append(txt)

    elif cfg.mass_plot_mode == "single":
        axm = ax_mass_list[0]

        for i, sp in enumerate(res.species):
            line, = axm.plot([], [], color=_species_color(colors, i), linewidth=2.5, alpha=0.9, label=f"{sp} Combined")
            single_species_comb_lines.append(line)

        total_ssa_line, = axm.plot([], [], color=colors["mass_ssa_line"], linestyle="--", linewidth=2.2, alpha=0.9, label="Total SSA")
        total_pde_line, = axm.plot([], [], color=colors["mass_pde_line"], linestyle="-.", linewidth=2.2, alpha=0.9, label="Total PDE")
        total_comb_line, = axm.plot([], [], color=colors["combined_total"], linewidth=3.0, alpha=0.9, label="Total Combined")
        single_total_lines = (total_ssa_line, total_pde_line, total_comb_line)

        axm.set_xlim(float(time[0]), float(time[-1]))
        axm.set_ylim(0, max_total_mass * 1.15)
        axm.set_xlabel("Time", fontsize=13, color=colors["text"])
        axm.set_ylabel("Total Mass (particles)", fontsize=13, color=colors["text"])

        lg = axm.legend(loc="upper right", framealpha=0.9, facecolor=colors["background"], edgecolor=colors["grid"], fontsize=9, ncol=2)
        for t in lg.get_texts():
            t.set_color(colors["text"])

        single_mass_text = axm.text(
            0.02, 0.90, "", transform=axm.transAxes,
            fontsize=10, color=colors["text"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor=colors["background"], edgecolor=colors["grid"], alpha=0.85)
        )

    # ------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------
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

        sl = slice(0, frame_idx + 1)

        # per-species mass plots
        if cfg.mass_plot_mode == "per_species":
            for i in range(n_species):
                ssa_line, pde_line, comb_line = per_species_mass_lines[i]
                ssa_line.set_data(time[sl], masses["ssa_mass"][i, sl])
                pde_line.set_data(time[sl], masses["pde_mass"][i, sl])
                comb_line.set_data(time[sl], masses["combined_mass"][i, sl])

                m0 = float(masses["combined_mass"][i, 0])
                mt = float(masses["combined_mass"][i, frame_idx])
                pct = ((mt - m0) / m0 * 100.0) if m0 != 0 else 0.0
                per_species_mass_texts[i].set_text(f"{res.species[i]}: {pct:+.3f}%")

        # single mass plot
        elif cfg.mass_plot_mode == "single":
            for i, line in enumerate(single_species_comb_lines):
                line.set_data(time[sl], masses["combined_mass"][i, sl])

            if single_total_lines is not None:
                total_ssa_line, total_pde_line, total_comb_line = single_total_lines
                total_ssa_line.set_data(time[sl], masses["total_ssa"][sl])
                total_pde_line.set_data(time[sl], masses["total_pde"][sl])
                total_comb_line.set_data(time[sl], masses["total_combined"][sl])

            m0 = float(masses["total_combined"][0])
            mt = float(masses["total_combined"][frame_idx])
            pct = ((mt - m0) / m0 * 100.0) if m0 != 0 else 0.0
            if single_mass_text is not None:
                single_mass_text.set_text(f"Mass conservation: {pct:+.4f}%")

        # collect artists
        artists = (
            list(bar_A) + list(bar_B)
            + [line_pde_A, line_pde_B, line_comb_A, line_comb_B, time_text]
        )
        if threshold_line is not None:
            artists.append(threshold_line)

        if cfg.mass_plot_mode == "per_species":
            for trio in per_species_mass_lines:
                artists.extend(trio)
            artists.extend(per_species_mass_texts)

        elif cfg.mass_plot_mode == "single":
            artists.extend(single_species_comb_lines)
            if single_total_lines is not None:
                artists.extend(list(single_total_lines))
            if single_mass_text is not None:
                artists.append(single_mass_text)

        return artists

    ani = FuncAnimation(
        fig, update, frames=frames,
        interval=int(cfg.interval_ms),
        blit=bool(cfg.blit),
        repeat=True,
    )

    plt.tight_layout()
    plt.show()
    return ani


# ------------------------------------------------------------
# Separate mass plot function
# ------------------------------------------------------------
def plot_mass_time_series(
    res: SimulationResults,
    plot_mode: Literal["single", "per_species", "comparison"] = "single",
    save_path: Optional[str] = None,
):
    colors = setup_cinematic_style()

    time = res.time
    ssa = res.ssa.astype(float)
    pde = res.pde.astype(float)

    dx = float(res.domain.dx)

    masses = total_mass_time_series(ssa, pde, dx=dx)
    n_species = len(res.species)

    def style_ax(ax):
        ax.set_facecolor(colors["background"])
        ax.grid(True, alpha=0.2, color=colors["grid"])
        ax.tick_params(colors=colors["text"])

    fig = None

    if plot_mode == "single":
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=colors["background"])
        style_ax(ax)

        for i, sp in enumerate(res.species):
            ax.plot(time, masses["combined_mass"][i], color=_species_color(colors, i), linewidth=2.5, label=f"{sp} Combined")

        ax.plot(time, masses["total_ssa"], color=colors["mass_ssa_line"], linestyle="--", linewidth=2, alpha=0.8, label="Total SSA")
        ax.plot(time, masses["total_pde"], color=colors["mass_pde_line"], linestyle="-.", linewidth=2, alpha=0.8, label="Total PDE")
        ax.plot(time, masses["total_combined"], color=colors["combined_total"], linewidth=3, alpha=0.9, label="Total Combined")

        ax.set_xlabel("Time", fontsize=13, color=colors["text"])
        ax.set_ylabel("Total Mass (particles)", fontsize=13, color=colors["text"])
        ax.set_title("Total Mass Evolution", fontsize=16, color=colors["text"], fontweight="bold")
        ax.legend(loc="best", framealpha=0.9, facecolor=colors["background"], edgecolor=colors["grid"], fontsize=10)

        m0 = float(masses["total_combined"][0])
        m1 = float(masses["total_combined"][-1])
        pct = ((m1 - m0) / m0 * 100.0) if m0 != 0 else 0.0
        ax.text(
            0.02, 0.95, f"Mass conservation: {pct:+.4f}%",
            transform=ax.transAxes, fontsize=11, color=colors["text"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors["background"], edgecolor=colors["grid"], alpha=0.85),
        )

    elif plot_mode == "per_species":
        fig, axes = plt.subplots(n_species, 1, figsize=(12, 4 * n_species), sharex=True, facecolor=colors["background"])
        if n_species == 1:
            axes = [axes]

        for i, (ax, sp) in enumerate(zip(axes, res.species)):
            style_ax(ax)
            ax.plot(time, masses["ssa_mass"][i], color=colors["mass_ssa_line"], linestyle="--", linewidth=2, alpha=0.9, label="SSA Mass")
            ax.plot(time, masses["pde_mass"][i], color=colors["mass_pde_line"], linestyle="-.", linewidth=2, alpha=0.9, label="PDE Mass")
            ax.plot(time, masses["combined_mass"][i], color=_species_color(colors, i), linewidth=2.5, alpha=0.9, label="Combined Mass")

            ax.set_ylabel(f"{sp}\nMass (particles)", fontsize=12, color=colors["text"])
            ax.legend(loc="best", framealpha=0.9, facecolor=colors["background"], edgecolor=colors["grid"], fontsize=9)

            m0 = float(masses["combined_mass"][i, 0])
            m1 = float(masses["combined_mass"][i, -1])
            pct = ((m1 - m0) / m0 * 100.0) if m0 != 0 else 0.0
            ax.text(
                0.02, 0.90, f"Mass conservation: {pct:+.3f}%",
                transform=ax.transAxes, fontsize=10, color=colors["text"],
                bbox=dict(boxstyle="round,pad=0.25", facecolor=colors["background"], edgecolor=colors["grid"], alpha=0.85),
            )

        axes[-1].set_xlabel("Time", fontsize=13, color=colors["text"])
        fig.suptitle("Mass Evolution per Species", fontsize=16, color=colors["text"], fontweight="bold")

    elif plot_mode == "comparison":
        fig, axes = plt.subplots(1, n_species, figsize=(6 * n_species, 5), facecolor=colors["background"])
        if n_species == 1:
            axes = [axes]

        for i, (ax, sp) in enumerate(zip(axes, res.species)):
            style_ax(ax)
            ax.plot(time, masses["ssa_mass"][i], color=colors["mass_ssa_line"], linestyle="--", linewidth=2, alpha=0.9, label="SSA Mass")
            ax.plot(time, masses["pde_mass"][i], color=colors["mass_pde_line"], linestyle="-.", linewidth=2, alpha=0.9, label="PDE Mass")
            ax.plot(time, masses["combined_mass"][i], color=_species_color(colors, i), linewidth=2.5, alpha=0.9, label="Combined Mass")

            ax.set_xlabel("Time", fontsize=12, color=colors["text"])
            ax.set_ylabel("Mass (particles)", fontsize=12, color=colors["text"])
            ax.set_title(f"{sp} Mass Components", fontsize=14, color=colors["text"])
            ax.legend(loc="best", framealpha=0.9, facecolor=colors["background"], edgecolor=colors["grid"], fontsize=9)

    else:
        raise ValueError("plot_mode must be one of: 'single', 'per_species', 'comparison'")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, facecolor=colors["background"], edgecolor="none", bbox_inches="tight")
        print(f"Mass plot saved to: {save_path}")

    plt.show()
    return fig
