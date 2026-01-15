from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union
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
    # NOTE:
    #   For backward-compatibility, `threshold` and `rate` may be provided either
    #   as scalars (single global values) or as per-species 1D arrays of length
    #   n_species. All internal computations broadcast accordingly.
    threshold: Union[float, np.ndarray]   # mass threshold for regime decision
    rate: Union[float, np.ndarray]        # gamma conversion rate

    def __post_init__(self):
        thr = self.threshold
        rt = self.rate

        # validate threshold
        if np.isscalar(thr):
            if float(thr) < 0:
                raise ValueError("ConversionParams.threshold must be >= 0")
        else:
            arr = np.asarray(thr, dtype=float)
            if arr.ndim != 1:
                raise ValueError("ConversionParams.threshold must be a scalar or 1D array")
            if np.any(arr < 0):
                raise ValueError("ConversionParams.threshold values must be >= 0")
            object.__setattr__(self, "threshold", arr)

        # validate rate
        if np.isscalar(rt):
            if float(rt) < 0:
                raise ValueError("ConversionParams.rate must be >= 0")
        else:
            arr = np.asarray(rt, dtype=float)
            if arr.ndim != 1:
                raise ValueError("ConversionParams.rate must be a scalar or 1D array")
            if np.any(arr < 0):
                raise ValueError("ConversionParams.rate values must be >= 0")
            object.__setattr__(self, "rate", arr)

    def rate_for(self, s_idx: int) -> float:
        """Return conversion rate for a given species index."""
        if np.isscalar(self.rate):
            return float(self.rate)
        rate_arr = np.asarray(self.rate, dtype=float)
        return float(rate_arr[s_idx])

    def threshold_for(self, s_idx: int) -> float:
        """Return conversion threshold for a given species index."""
        if np.isscalar(self.threshold):
            return float(self.threshold)
        thr_arr = np.asarray(self.threshold, dtype=float)
        return float(thr_arr[s_idx])

    def exceeds_threshold_mask(self, combined_mass: np.ndarray) -> np.ndarray:
        """
        combined_mass: (n_species, K)
        Returns: (n_species, K) int8 mask where 1 means > threshold else 0
        """
        if combined_mass.ndim != 2:
            raise ValueError("combined_mass must be 2D (n_species, K)")
        thr = self.threshold
        if np.isscalar(thr):
            return (combined_mass > float(thr)).astype(np.int8)

        thr_arr = np.asarray(thr, dtype=float)
        if thr_arr.shape != (combined_mass.shape[0],):
            raise ValueError(
                "Per-species threshold must have shape (n_species,) matching combined_mass"
            )
        return (combined_mass > thr_arr[:, None]).astype(np.int8)

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
