from __future__ import annotations
import numpy as np


def pde_mass_per_compartment(pde_conc: np.ndarray, pde_multiple: int, dx: float) -> np.ndarray:
    """
    Project fine-grid PDE concentrations onto SSA compartments using a left-hand rule sum.

    Conventions
    ----------
    pde_conc shape: (n_species, Npde)
    where Npde = K * pde_multiple

    Returns
    -------
    pde_mass : np.ndarray
        Shape (n_species, K), where each entry is ∫_compartment C(x) dx (approx).
    """
    if pde_conc.ndim != 2:
        raise ValueError("pde_conc must be 2D with shape (n_species, Npde)")
    if pde_multiple <= 0:
        raise ValueError("pde_multiple must be > 0")
    if dx <= 0:
        raise ValueError("dx must be > 0")

    n_species, Npde = pde_conc.shape
    if Npde % pde_multiple != 0:
        raise ValueError("Npde must be divisible by pde_multiple")

    K = Npde // pde_multiple

    # Reshape so each SSA compartment is a block of pde_multiple PDE cells
    # shape becomes (n_species, K, pde_multiple)
    blocks = pde_conc.astype(float, copy=False).reshape(n_species, K, pde_multiple)

    # Left-hand rule mass approximation per compartment
    return blocks.sum(axis=2) * dx


def combined_mass(ssa_counts: np.ndarray, pde_conc: np.ndarray, pde_multiple: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute combined mass = SSA counts + projected PDE mass (per compartment).

    Parameters
    ----------
    ssa_counts : np.ndarray
        Shape (n_species, K) integer discrete counts.
    pde_conc : np.ndarray
        Shape (n_species, Npde) continuous concentrations.
    pde_multiple : int
        PDE cells per SSA compartment.
    dx : float
        PDE grid spacing.

    Returns
    -------
    combined : np.ndarray
        Shape (n_species, K) = ssa_counts + pde_mass
    pde_mass : np.ndarray
        Shape (n_species, K)
    """
    if ssa_counts.ndim != 2:
        raise ValueError("ssa_counts must be 2D with shape (n_species, K)")
    if pde_conc.ndim != 2:
        raise ValueError("pde_conc must be 2D with shape (n_species, Npde)")

    pde_mass = pde_mass_per_compartment(pde_conc, pde_multiple, dx)

    # Basic shape consistency check
    n_species, K = ssa_counts.shape
    if pde_mass.shape != (n_species, K):
        raise ValueError(
            f"Shape mismatch: ssa_counts {(n_species, K)} vs pde_mass {pde_mass.shape}. "
            "Check that Npde == K * pde_multiple."
        )

    combined = ssa_counts.astype(int, copy=False) + pde_mass
    return combined, pde_mass
