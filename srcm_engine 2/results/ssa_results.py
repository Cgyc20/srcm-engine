from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
import numpy as np

from srcm_engine.domain import Domain


@dataclass(frozen=True)
class SSAResults:
    """
    Results container for *pure SSA* runs.

    This mirrors the attributes expected by srcm_engine.results.io.save_results()
    so SSA-only simulations can be saved with the exact same file layout
    (<prefix>.npz + <prefix>.json) as hybrid runs.

    Shapes
    ------
    ssa : (n_species, K, n_steps)
    time : (n_steps,)

    Notes
    -----
    - For compatibility with plotting/animation code that expects a PDE array,
      the `pde` field may be None. The IO layer will write a zero PDE array
      of shape (n_species, domain.n_pde, n_steps) when needed.
    """
    time: np.ndarray
    ssa: np.ndarray
    domain: Domain
    species: Sequence[str]
    pde: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None
