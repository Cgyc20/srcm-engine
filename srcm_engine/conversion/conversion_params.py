from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConversionParams:
    """
    SRCM conversion parameters.

    Conventions
    ----------
    K : number of SSA compartments
    n_species : number of species (often 'M' in notes)

    Arrays use:
      SSA counts: (n_species, K)
      PDE conc:   (n_species, Npde) where Npde = K * pde_multiple
    """
    threshold: float   # mass threshold for regime decision
    rate: float        # gamma conversion rate

    def __post_init__(self):
        if self.threshold < 0:
            raise ValueError("ConversionParams.threshold must be >= 0")
        if self.rate < 0:
            raise ValueError("ConversionParams.rate must be >= 0")

    def exceeds_threshold_mask(self, combined_mass: np.ndarray) -> np.ndarray:
        """
        combined_mass: (n_species, K)
        Returns: (n_species, K) int8 mask where 1 means > threshold else 0
        """
        if combined_mass.ndim != 2:
            raise ValueError("combined_mass must be 2D (n_species, K)")
        return (combined_mass > self.threshold).astype(np.int8)

    def sufficient_pde_mass_mask(self, pde_conc: np.ndarray, pde_multiple: int, h: float) -> np.ndarray:
        """
        pde_conc: (n_species, Npde), Npde must be divisible by pde_multiple
        Returns: (n_species, K) int8 mask.

        Rule:
          sufficient = all fine PDE cells in the compartment satisfy C >= 1/h
        """
        if pde_conc.ndim != 2:
            raise ValueError("pde_conc must be 2D (n_species, Npde)")
        if pde_multiple <= 0:
            raise ValueError("pde_multiple must be > 0")
        if h <= 0:
            raise ValueError("h must be > 0")

        n_species, Npde = pde_conc.shape
        if Npde % pde_multiple != 0:
            raise ValueError("Npde must be divisible by pde_multiple")

        K = Npde // pde_multiple
        conc_threshold = 1.0 / h

        reshaped = pde_conc.reshape(n_species, K, pde_multiple)
        return np.all(reshaped >= conc_threshold, axis=2).astype(np.int8)
