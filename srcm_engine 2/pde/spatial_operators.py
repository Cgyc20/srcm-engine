from __future__ import annotations
import numpy as np
from srcm_engine.domain import Domain


def laplacian_1d(domain: Domain) -> np.ndarray:
    """
    Construct the 1D Laplacian matrix on the PDE grid.

    Boundary conditions:
      - zero-flux (Neumann)
      - periodic

    Returns
    -------
    L : np.ndarray
        Shape (Npde, Npde)
        Discrete Laplacian WITHOUT diffusion coefficient or dx^2 scaling.
    """
    N = domain.n_pde
    L = np.zeros((N, N), dtype=float)

    if domain.boundary == "zero-flux":
        # Left boundary
        L[0, 0] = -1.0
        L[0, 1] = 1.0

        # Interior points
        for i in range(1, N - 1):
            L[i, i - 1] = 1.0
            L[i, i] = -2.0
            L[i, i + 1] = 1.0

        # Right boundary
        L[N - 1, N - 2] = 1.0
        L[N - 1, N - 1] = -1.0

    elif domain.boundary == "periodic":
        for i in range(N):
            L[i, i] = -2.0
            L[i, (i - 1) % N] = 1.0
            L[i, (i + 1) % N] = 1.0

    else:
        raise ValueError(f"Unknown boundary condition '{domain.boundary}'")

    return L
