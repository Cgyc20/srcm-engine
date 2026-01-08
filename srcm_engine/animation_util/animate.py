from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Literal, List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec

from ..results import SimulationResults


# ============================================================
# Utilities
# ============================================================
def _map_frames_by_time(t_ref: np.ndarray, t_other: np.ndarray) -> np.ndarray:
    """
    For each time in t_ref, pick nearest index in t_other.
    Assumes both are 1D increasing.
    """
    if len(t_other) == 0:
        raise ValueError("t_other is empty")
    if len(t_ref) == 0:
        return np.array([], dtype=int)

    idx = np.searchsorted(t_other, t_ref)
    idx = np.clip(idx, 1, len(t_other) - 1)
    left = idx - 1
    right = idx
    choose_right = (np.abs(t_other[right] - t_ref) < np.abs(t_other[left] - t_ref))
    return np.where(choose_right, right, left)


def _validate_one_or_two_species(species: Sequence[str]) -> None:
    if len(species) not in (1, 2):
        raise ValueError("This animation helper currently supports 1 or 2 species.")


def setup_cinematic_style() -> Dict[str, str]:
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


def _species_color(colors: Dict[str, str], i: int) -> str:
    key = f"species_{chr(97 + i)}"  # a,b,c,d
    return colors.get(key, "white")


def _species_light_color(colors: Dict[str, str], i: int) -> str:
    key = f"species_{chr(97 + i)}_light"
    return colors.get(key, "white")


def sync_particles_axis(ax_left: plt.Axes, ax_right: plt.Axes, *, h: float):
    """Keep right axis (particles per SSA box) consistent with left (concentration)."""
    y0, y1 = ax_left.get_ylim()
    ax_right.set_ylim(y0 * h, y1 * h)


def ssa_to_concentration_on_pde_grid(
    ssa: np.ndarray,
    domain_h: float,
    pde_multiple: int,
) -> np.ndarray:
    """
    ssa shape: (n_species, K, T)
    return:    (n_species, Npde, T) where Npde = K * pde_multiple
    """
    n_species, K, T = ssa.shape
    Npde = K * pde_multiple

    out = np.zeros((n_species, Npde, T), dtype=float)
    for i in range(K):
        s = i * pde_multiple
        e = s + pde_multiple
        # broadcast across fine cells; convert count to concentration (count / h)
        out[:, s:e, :] = (ssa[:, i : i + 1, :] / domain_h)
    return out


def total_mass_time_series(
    ssa: np.ndarray,
    pde: np.ndarray,
    *,
    dx: float,
) -> Dict[str, np.ndarray]:
    """
    Mass in particle units:
      SSA mass = sum of counts over SSA compartments
      PDE mass = integral sum(C) * dx  (C is conc in particles/length)
      Combined = SSA + PDE
    """
    # ssa: (n_species, K, T) counts
    # pde: (n_species, Npde, T) concentration
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


# ============================================================
# Animation Configuration
# ============================================================
@dataclass
class AnimationConfig:
    stride: int = 10
    interval_ms: int = 30
    show_threshold: bool = True
    threshold_particles: Optional[float] = None  # particles per SSA compartment
    title: str = "SRCM Hybrid Simulation"
    blit: bool = False
    mass_plot_mode: Literal["single", "per_species", "none"] = "single"


# ============================================================
# Animation: Results (single SimulationResults)
# ============================================================
def animate_results(res: SimulationResults, cfg: Optional[AnimationConfig] = None):
    """
    Animate SSA bars + PDE lines for 1 or 2 species.
    """
    if cfg is None:
        cfg = AnimationConfig()

    _validate_one_or_two_species(res.species)
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
    ssa_conc = ssa.astype(float) / h  # particles/length
    ssa_conc_on_pde = ssa_to_concentration_on_pde_grid(ssa.astype(float), h, pde_multiple)
    combined_conc = pde + ssa_conc_on_pde

    masses = total_mass_time_series(ssa.astype(float), pde.astype(float), dx=dx)

    max_conc = float(np.max(combined_conc))
    if not np.isfinite(max_conc) or max_conc <= 0:
        max_conc = 1.0

    # ------------------------------------------------------------
    # Figure + Axes Setup
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(16, 9), facecolor=colors["background"])

    if cfg.mass_plot_mode == "per_species":
        gs = gridspec.GridSpec(1 + n_species, 1, height_ratios=[3] + [1] * n_species, hspace=0.3, figure=fig)
    elif cfg.mass_plot_mode == "single":
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25, figure=fig)
    else:
        gs = gridspec.GridSpec(1, 1, figure=fig)

    ax_main = fig.add_subplot(gs[0])
    ax_particle = ax_main.twinx()

    # Style
    ax_main.set_facecolor(colors["background"])
    ax_main.grid(True, alpha=0.2, color=colors["grid"])
    for ax in [ax_main, ax_particle]:
        ax.tick_params(colors=colors["text"])

    # ------------------------------------------------------------
    # Artists
    # ------------------------------------------------------------
    bars: List[mpl.container.BarContainer] = []
    lines_pde: List[mpl.lines.Line2D] = []

    for i, sp in enumerate(res.species):
        bar = ax_main.bar(
            ssa_x,
            ssa_conc[i, :, 0],
            width=bar_w,
            align="edge",
            color=_species_light_color(colors, i),
            alpha=0.7,
            edgecolor=_species_color(colors, i),
            linewidth=0.5,
            label=f"SSA {sp}",
        )
        bars.append(bar)

        line, = ax_main.plot(
            pde_x,
            pde[i, :, 0],
            color=_species_color(colors, i),
            linewidth=2.5,
            label=f"PDE {sp}",
        )
        lines_pde.append(line)

    threshold_line = None
    if cfg.show_threshold and (cfg.threshold_particles is not None):
        conc_threshold = float(cfg.threshold_particles) / h
        threshold_line = ax_main.axhline(
            conc_threshold,
            color=colors["threshold"],
            linestyle=":",
            linewidth=2.5,
            alpha=0.9,
            label=f"Threshold ({cfg.threshold_particles} particles)",
        )

    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, max_conc * 1.15)
    ax_main.set_ylabel("Concentration (particles / length)", color=colors["text"], fontsize=12)
    ax_particle.set_ylabel("Particles per SSA compartment", color=colors["text"], fontsize=12)
    ax_main.set_title(cfg.title, color=colors["text"], fontweight="bold")

    sync_particles_axis(ax_main, ax_particle, h=h)

    # Legend + time label
    leg = ax_main.legend(loc="upper right", framealpha=0.9, facecolor=colors["background"], edgecolor=colors["grid"])
    for t in leg.get_texts():
        t.set_color(colors["text"])

    time_text = ax_main.text(
        0.02, 0.95, "",
        transform=ax_main.transAxes,
        color=colors["text"],
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor=colors["background"], edgecolor=colors["grid"]),
    )

    # ------------------------------------------------------------
    # Mass plot (optional)
    # ------------------------------------------------------------
    mass_axes: List[plt.Axes] = []
    mass_lines: List[Tuple[mpl.lines.Line2D, mpl.lines.Line2D, mpl.lines.Line2D]] = []  # (ssa, pde, combined)

    def _style_ax(ax: plt.Axes):
        ax.set_facecolor(colors["background"])
        ax.grid(True, alpha=0.2, color=colors["grid"])
        ax.tick_params(colors=colors["text"])

    if cfg.mass_plot_mode == "single":
        ax_mass = fig.add_subplot(gs[1], sharex=None)
        _style_ax(ax_mass)
        ax_mass.set_xlabel("Time", color=colors["text"])
        ax_mass.set_ylabel("Mass (particles)", color=colors["text"])
        ax_mass.set_title("Mass evolution", color=colors["text"], fontweight="bold")

        l_ssa, = ax_mass.plot(time, masses["total_ssa"], color=colors["mass_ssa_line"], linestyle="--", linewidth=2, alpha=0.9, label="Total SSA")
        l_pde, = ax_mass.plot(time, masses["total_pde"], color=colors["mass_pde_line"], linestyle="-.", linewidth=2, alpha=0.9, label="Total PDE")
        l_comb, = ax_mass.plot(time, masses["total_combined"], color=colors["combined_total"], linewidth=3, alpha=0.9, label="Total Combined")
        ax_mass.legend(loc="best", framealpha=0.9, facecolor=colors["background"], edgecolor=colors["grid"])

        mass_axes = [ax_mass]
        mass_lines = [(l_ssa, l_pde, l_comb)]

    elif cfg.mass_plot_mode == "per_species":
        # rows 1..n_species
        for i, sp in enumerate(res.species):
            ax_m = fig.add_subplot(gs[1 + i])
            _style_ax(ax_m)
            ax_m.set_ylabel(f"{sp}\nMass", color=colors["text"])
            if i == n_species - 1:
                ax_m.set_xlabel("Time", color=colors["text"])

            l_ssa, = ax_m.plot(time, masses["ssa_mass"][i], color=colors["mass_ssa_line"], linestyle="--", linewidth=2, alpha=0.9, label="SSA")
            l_pde, = ax_m.plot(time, masses["pde_mass"][i], color=colors["mass_pde_line"], linestyle="-.", linewidth=2, alpha=0.9, label="PDE")
            l_comb, = ax_m.plot(time, masses["combined_mass"][i], color=_species_color(colors, i), linewidth=2.5, alpha=0.9, label="Combined")
            ax_m.legend(loc="best", framealpha=0.9, facecolor=colors["background"], edgecolor=colors["grid"])

            mass_axes.append(ax_m)
            mass_lines.append((l_ssa, l_pde, l_comb))

    # ------------------------------------------------------------
    # Update function
    # ------------------------------------------------------------
    frames = list(range(0, T, max(1, int(cfg.stride))))

    def update(frame_idx: int):
        # bars
        for i, bar in enumerate(bars):
            for b, hgt in zip(bar, ssa_conc[i, :, frame_idx]):
                b.set_height(float(hgt))

        # PDE lines
        for i, line in enumerate(lines_pde):
            line.set_ydata(pde[i, :, frame_idx])

        sync_particles_axis(ax_main, ax_particle, h=h)

        time_text.set_text(f"t = {time[frame_idx]:.4g}   frame={frame_idx}")

        # mass lines (optional) - no per-frame slicing needed if already full time-series,
        # but we can add a vertical indicator if desired later.
        # Keep as-is.

        artists: List[object] = []
        for bar in bars:
            artists += list(bar)
        artists += lines_pde
        artists.append(time_text)
        if threshold_line is not None:
            artists.append(threshold_line)
        # blit doesn't play perfectly with shared axes sometimes; cfg.blit controls it anyway.
        return artists

    ani = FuncAnimation(fig, update, frames=frames, interval=int(cfg.interval_ms), blit=bool(cfg.blit))
    plt.tight_layout()
    plt.show()
    return ani


# ============================================================
# Animation: Overlay (two SimulationResults)
# ============================================================
def animate_overlay(
    res_main: SimulationResults,
    res_overlay: SimulationResults,
    *,
    cfg: Optional[AnimationConfig] = None,
    label_main: str = "Hybrid",
    label_overlay: str = "SSA",
):
    """
    Overlay animation:
      - main SSA bars (filled)
      - overlay SSA bars (outlined dashed)
      - main combined line = PDE + main SSA (mapped onto PDE grid)
    Supports 1 or 2 species.
    """
    if cfg is None:
        cfg = AnimationConfig()

    _validate_one_or_two_species(res_main.species)
    _validate_one_or_two_species(res_overlay.species)

    if list(res_main.species) != list(res_overlay.species):
        raise ValueError(f"Species mismatch: {res_main.species} vs {res_overlay.species}")

    if (res_main.domain.length != res_overlay.domain.length) or (res_main.domain.n_ssa != res_overlay.domain.n_ssa):
        raise ValueError("Domain mismatch: need same length and same n_ssa (K) for overlay.")

    colors = setup_cinematic_style()

    # Data
    time = res_main.time
    ssa = res_main.ssa
    pde = res_main.pde

    time2 = res_overlay.time
    ssa2 = res_overlay.ssa

    n_species, K, T = ssa.shape
    h = float(res_main.domain.h)
    pde_multiple = int(res_main.domain.pde_multiple)
    L = float(res_main.domain.length)
    Npde = K * pde_multiple

    tmap = _map_frames_by_time(time, time2)

    # Coordinates
    ssa_x = (np.arange(K) * h) / L
    pde_x = (np.linspace(0.0, L, Npde)) / L
    bar_w = h / L

    # Concentrations
    ssa_conc = ssa.astype(float) / h
    ssa2_conc = ssa2.astype(float) / h

    ssa_on_pde = ssa_to_concentration_on_pde_grid(ssa.astype(float), h, pde_multiple)
    combined = pde + ssa_on_pde

    max_conc = float(np.max([combined.max(), ssa2_conc.max()]))
    if not np.isfinite(max_conc) or max_conc <= 0:
        max_conc = 1.0

    # Figure
    fig = plt.figure(figsize=(16, 9), facecolor=colors["background"])
    ax = fig.add_subplot(111)
    ax_right = ax.twinx()

    ax.set_facecolor(colors["background"])
    ax.grid(True, alpha=0.2, color=colors["grid"], linestyle="-", linewidth=0.5)
    for a in [ax, ax_right]:
        a.tick_params(colors=colors["text"])

    # Artists per species
    bar_main: List[mpl.container.BarContainer] = []
    bar_ov: List[mpl.container.BarContainer] = []
    line_main: List[mpl.lines.Line2D] = []

    for i, sp in enumerate(res_main.species):
        bm = ax.bar(
            ssa_x, ssa_conc[i, :, 0],
            width=bar_w, align="edge",
            color=_species_light_color(colors, i), alpha=0.4,
            edgecolor=_species_color(colors, i), linewidth=0.5,
            label=f"{label_main} SSA {sp}",
        )
        bar_main.append(bm)

        bo = ax.bar(
            ssa_x, ssa2_conc[i, :, int(tmap[0])],
            width=bar_w, align="edge",
            color="none",
            edgecolor=_species_color(colors, i),
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
            label=f"{label_overlay} SSA {sp}",
        )
        bar_ov.append(bo)

        ln, = ax.plot(
            pde_x, combined[i, :, 0],
            color=_species_color(colors, i), linewidth=3.0,
            label=f"{label_main} combined {sp}",
        )
        line_main.append(ln)

    threshold_line = None
    if cfg.show_threshold and cfg.threshold_particles is not None:
        conc_threshold = float(cfg.threshold_particles) / h
        threshold_line = ax.axhline(
            conc_threshold,
            color=colors["threshold"],
            linestyle=":",
            linewidth=2.5,
            alpha=0.9,
            label=f"Threshold ({cfg.threshold_particles} ptcl)",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, max_conc * 1.15)
    ax.set_xlabel("Scaled Domain [0,1]", color=colors["text"])
    ax.set_ylabel("Concentration", color=colors["text"])
    ax_right.set_ylabel("Particles per SSA box", color=colors["text"])
    ax.set_title(cfg.title, color=colors["text"], fontweight="bold")

    sync_particles_axis(ax, ax_right, h=h)

    leg = ax.legend(loc="upper right", framealpha=0.9, facecolor=colors["background"], edgecolor=colors["grid"])
    for t in leg.get_texts():
        t.set_color(colors["text"])

    time_text = ax.text(
        0.02, 0.95, "",
        transform=ax.transAxes,
        color=colors["text"],
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor=colors["background"], edgecolor=colors["grid"]),
    )

    frames = list(range(0, T, max(1, int(cfg.stride))))

    def update(frame_idx: int):
        # Main bars
        for i, bm in enumerate(bar_main):
            for b, hgt in zip(bm, ssa_conc[i, :, frame_idx]):
                b.set_height(float(hgt))

        # Overlay bars (mapped)
        j = int(tmap[frame_idx])
        for i, bo in enumerate(bar_ov):
            for b, hgt in zip(bo, ssa2_conc[i, :, j]):
                b.set_height(float(hgt))

        # Main combined lines
        for i, ln in enumerate(line_main):
            ln.set_ydata(combined[i, :, frame_idx])

        sync_particles_axis(ax, ax_right, h=h)
        time_text.set_text(f"t = {time[frame_idx]:.4g}\nmain={frame_idx}  overlay={j}")

        artists: List[object] = []
        for bm in bar_main:
            artists += list(bm)
        for bo in bar_ov:
            artists += list(bo)
        artists += line_main
        artists.append(time_text)
        if threshold_line is not None:
            artists.append(threshold_line)
        return artists

    ani = FuncAnimation(fig, update, frames=frames, interval=int(cfg.interval_ms), blit=bool(cfg.blit))
    plt.tight_layout()
    plt.show()
    return ani


# ============================================================
# Mass plot helper
# ============================================================
def plot_mass_time_series(
    res: SimulationResults,
    plot_mode: Literal["single", "per_species", "comparison"] = "single",
    save_path: Optional[str] = None,
):
    """
    Plot mass time-series for 1 or 2 species.
    """
    _validate_one_or_two_species(res.species)
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

        # per-species combined (if 1 species, just one line)
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
