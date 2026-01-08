from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec

from ..results import SimulationResults

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def _map_frames_by_time(t_ref: np.ndarray, t_other: np.ndarray) -> np.ndarray:
    """
    For each time in t_ref, pick nearest index in t_other.
    Assumes both are 1D increasing.
    """
    idx = np.searchsorted(t_other, t_ref)
    idx = np.clip(idx, 1, len(t_other) - 1)
    left = idx - 1
    right = idx
    choose_right = (np.abs(t_other[right] - t_ref) < np.abs(t_other[left] - t_ref))
    return np.where(choose_right, right, left)

def animate_overlay(
    res_main: SimulationResults,
    res_overlay: SimulationResults,
    *,
    cfg: Optional[AnimationConfig] = None,
    label_main: str = "Hybrid",
    label_overlay: str = "SSA",
):
    if cfg is None:
        cfg = AnimationConfig()

    _validate_two_species(res_main.species)
    _validate_two_species(res_overlay.species)

    if list(res_main.species) != list(res_overlay.species):
        raise ValueError(f"Species mismatch: {res_main.species} vs {res_overlay.species}")

    if (res_main.domain.length != res_overlay.domain.length) or (res_main.domain.n_ssa != res_overlay.domain.n_ssa):
        raise ValueError("Domain mismatch: need same length and same n_ssa (K) for overlay.")

    colors = setup_cinematic_style()

    # Data extraction
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

    fig = plt.figure(figsize=(16, 9), facecolor=colors["background"])
    ax = fig.add_subplot(111)
    
    # NEW: Particle Count Axis (Right)
    ax_right = ax.twinx()
    
    ax.set_facecolor(colors["background"])
    ax.grid(True, alpha=0.2, color=colors["grid"], linestyle="-", linewidth=0.5)
    
    # Style axes
    for a in [ax, ax_right]:
        a.tick_params(colors=colors["text"])

    # 1. MAIN SSA BARS
    bar_main_A = ax.bar(
        ssa_x, ssa_conc[0, :, 0],
        width=bar_w, align="edge",
        color=_species_light_color(colors, 0), alpha=0.4,
        edgecolor=_species_color(colors, 0), linewidth=0.5,
        label=f"{label_main} SSA {res_main.species[0]}",
    )
    bar_main_B = ax.bar(
        ssa_x, ssa_conc[1, :, 0],
        width=bar_w, align="edge",
        color=_species_light_color(colors, 1), alpha=0.4,
        edgecolor=_species_color(colors, 1), linewidth=0.5,
        label=f"{label_main} SSA {res_main.species[1]}",
    )

    # 2. OVERLAY SSA BARS
    bar_ov_A = ax.bar(
        ssa_x, ssa2_conc[0, :, tmap[0]],
        width=bar_w, align="edge",
        color="none", edgecolor=_species_color(colors, 0), 
        linewidth=1.5, linestyle="--", alpha=0.8,
        label=f"{label_overlay} SSA {res_main.species[0]}",
    )
    bar_ov_B = ax.bar(
        ssa_x, ssa2_conc[1, :, tmap[0]],
        width=bar_w, align="edge",
        color="none", edgecolor=_species_color(colors, 1), 
        linewidth=1.5, linestyle="--", alpha=0.8,
        label=f"{label_overlay} SSA {res_main.species[1]}",
    )

    # 3. MAIN COMBINED LINES
    line_main_A, = ax.plot(
        pde_x, combined[0, :, 0],
        color=_species_color(colors, 0), linewidth=3.0,
        label=f"{label_main} combined {res_main.species[0]}",
    )
    line_main_B, = ax.plot(
        pde_x, combined[1, :, 0],
        color=_species_color(colors, 1), linewidth=3.0,
        label=f"{label_main} combined {res_main.species[1]}",
    )

    # NEW: Threshold Line
    threshold_line = None
    if cfg.show_threshold and cfg.threshold_particles is not None:
        conc_threshold = float(cfg.threshold_particles) / h
        threshold_line = ax.axhline(
            conc_threshold, color=colors["threshold"], 
            linestyle=":", linewidth=2.5, alpha=0.9, 
            label=f"Threshold ({cfg.threshold_particles} ptcl)"
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, max_conc * 1.15)
    ax.set_xlabel("Scaled Domain [0,1]", color=colors["text"])
    ax.set_ylabel("Concentration", color=colors["text"])
    ax_right.set_ylabel("Particles per SSA box", color=colors["text"])
    ax.set_title(cfg.title, color=colors["text"], fontweight="bold")

    # Initial sync of the right axis
    sync_particles_axis(ax, ax_right, h=h)

    leg = ax.legend(loc="upper right", framealpha=0.9, facecolor=colors["background"], edgecolor=colors["grid"])
    for t in leg.get_texts(): t.set_color(colors["text"])

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color=colors["text"], fontweight="bold",
                        bbox=dict(boxstyle="round", facecolor=colors["background"], edgecolor=colors["grid"]))

    frames = list(range(0, T, max(1, int(cfg.stride))))

    def update(frame_idx: int):
        # Update Main Bars
        for b, hgt in zip(bar_main_A, ssa_conc[0, :, frame_idx]):
            b.set_height(float(hgt))
        for b, hgt in zip(bar_main_B, ssa_conc[1, :, frame_idx]):
            b.set_height(float(hgt))

        # Update Overlay Bars (Mapped index)
        j = int(tmap[frame_idx])
        for b, hgt in zip(bar_ov_A, ssa2_conc[0, :, j]):
            b.set_height(float(hgt))
        for b, hgt in zip(bar_ov_B, ssa2_conc[1, :, j]):
            b.set_height(float(hgt))

        # Update Main Lines
        line_main_A.set_ydata(combined[0, :, frame_idx])
        line_main_B.set_ydata(combined[1, :, frame_idx])

        # Keep axes synced
        sync_particles_axis(ax, ax_right, h=h)

        time_text.set_text(f"t = {time[frame_idx]:.4g}\nmain={frame_idx}  overlay={j}")

        artists = (list(bar_main_A) + list(bar_main_B) + 
                   list(bar_ov_A) + list(bar_ov_B) + 
                   [line_main_A, line_main_B, time_text])
        if threshold_line:
            artists.append(threshold_line)
        return artists

    ani = FuncAnimation(fig, update, frames=frames, interval=int(cfg.interval_ms), blit=bool(cfg.blit))
    plt.tight_layout()
    plt.show()
    return ani


def add_particles_per_box_axis(ax_left: plt.Axes, *, h: float, colors: dict) -> plt.Axes:
    """
    Left axis is concentration (particles / length).
    Right axis will display equivalent particles-per-SSA-box.
    Mapping: particles_per_box = conc * h
    """
    ax_right = ax_left.twinx()

    # Keep the two axes perfectly aligned
    y0, y1 = ax_left.get_ylim()
    ax_right.set_ylim(y0 * h, y1 * h)

    ax_right.set_ylabel("Particles per SSA box", fontsize=13, color=colors["text"])
    ax_right.tick_params(colors=colors["text"])

    # If left ylim changes later, call sync function again (see below).
    return ax_right


def sync_particles_axis(ax_left: plt.Axes, ax_right: plt.Axes, *, h: float):
    """Call after any ax_left.set_ylim(...) to keep right axis consistent."""
    y0, y1 = ax_left.get_ylim()
    ax_right.set_ylim(y0 * h, y1 * h)



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
    ssa_conc = ssa.astype(float) / h  
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
    
    # NEW: Create the Particle Count axis (Right Y-axis)
    ax_particle = ax_main.twinx()
    
    # Style the axes
    for ax in [ax_main, ax_particle]:
        ax.set_facecolor("none") # Transparency for overlay
        ax.tick_params(colors=colors["text"])
    
    ax_main.set_facecolor(colors["background"])
    ax_main.grid(True, alpha=0.2, color=colors["grid"])

    # ------------------------------------------------------------
    # Main Plot Artists
    # ------------------------------------------------------------
    bar_A = ax_main.bar(ssa_x, ssa_conc[0, :, 0], width=bar_w, align="edge",
                        color=_species_light_color(colors, 0), alpha=0.7,
                        edgecolor=_species_color(colors, 0), linewidth=0.5, label=f"SSA {res.species[0]}")
    bar_B = ax_main.bar(ssa_x, ssa_conc[1, :, 0], width=bar_w, align="edge",
                        color=_species_light_color(colors, 1), alpha=0.7,
                        edgecolor=_species_color(colors, 1), linewidth=0.5, label=f"SSA {res.species[1]}")

    line_pde_A, = ax_main.plot(pde_x, pde[0, :, 0], color=_species_color(colors, 0), linewidth=2.5, label=f"PDE {res.species[0]}")
    line_pde_B, = ax_main.plot(pde_x, pde[1, :, 0], color=_species_color(colors, 1), linewidth=2.5, label=f"PDE {res.species[1]}")

    # NEW: Plot the Conversion Threshold Line
    threshold_line = None
    if cfg.threshold_particles is not None:
        conc_threshold = float(cfg.threshold_particles) / h
        threshold_line = ax_main.axhline(conc_threshold, color=colors["threshold"], 
                                         linestyle=":", linewidth=2.5, alpha=0.9, 
                                         label=f"Threshold ({cfg.threshold_particles} particles)")

    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, max_conc * 1.15)
    ax_main.set_ylabel("Concentration (particles/L)", color=colors["text"], fontsize=12)
    ax_particle.set_ylabel("Particles per SSA Compartment", color=colors["text"], fontsize=12)
    
    # Sync the right axis limits initially
    sync_particles_axis(ax_main, ax_particle, h=h)

    # ... [Rest of mass axis setup remains similar to your original code] ...

    # ------------------------------------------------------------
    # Animation Update Function
    # ------------------------------------------------------------
    def update(frame_idx: int):
        # Update Bars and Lines
        for b, hgt in zip(bar_A, ssa_conc[0, :, frame_idx]):
            b.set_height(float(hgt))
        for b, hgt in zip(bar_B, ssa_conc[1, :, frame_idx]):
            b.set_height(float(hgt))

        line_pde_A.set_ydata(pde[0, :, frame_idx])
        line_pde_B.set_ydata(pde[1, :, frame_idx])
        
        # Ensure right axis stays synced if left axis autoscale triggers
        sync_particles_axis(ax_main, ax_particle, h=h)

        # ... [Update mass plots logic same as before] ...

        artists = list(bar_A) + list(bar_B) + [line_pde_A, line_pde_B]
        if threshold_line: artists.append(threshold_line)
        # Add other artists (text, mass lines) as per original...
        return artists

    ani = FuncAnimation(fig, update, frames=range(0, T, cfg.stride), interval=cfg.interval_ms, blit=False)
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
