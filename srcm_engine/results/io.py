from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np

from srcm_engine.results.simulation_results import SimulationResults
from srcm_engine.domain import Domain


def save_results(results: SimulationResults, path_prefix: str | Path) -> None:
    """
    Save SimulationResults to:
      - <path_prefix>.npz  (arrays)
      - <path_prefix>.json (metadata)

    Parameters
    ----------
    results : SimulationResults
    path_prefix : str | Path
        Example: "data/turing_run1" -> saves "turing_run1.npz" and "turing_run1.json"
    """
    path_prefix = Path(path_prefix)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)

    npz_path = path_prefix.with_suffix(".npz")
    json_path = path_prefix.with_suffix(".json")

    # Save arrays
    np.savez_compressed(
        npz_path,
        time=results.time,
        ssa=results.ssa,
        pde=results.pde,
    )

    # Save metadata (keep it stable + readable)
    meta = {
        "species": results.species,
        "domain": {
            "length": results.domain.length,
            "n_ssa": results.domain.n_ssa,
            "pde_multiple": results.domain.pde_multiple,
            "boundary": results.domain.boundary,
        },
        "shapes": {
            "time": list(results.time.shape),
            "ssa": list(results.ssa.shape),
            "pde": list(results.pde.shape),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Optional: print is annoying in libraries, so we don't.


def load_results(path_prefix: str | Path) -> SimulationResults:
    """
    Load SimulationResults saved by save_results().

    Parameters
    ----------
    path_prefix : str | Path
        Example: "data/turing_run1" -> loads "turing_run1.npz" and "turing_run1.json"

    Returns
    -------
    SimulationResults
    """
    path_prefix = Path(path_prefix)
    npz_path = path_prefix.with_suffix(".npz")
    json_path = path_prefix.with_suffix(".json")

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing npz file: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing json file: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    dom_meta = meta["domain"]
    domain = Domain(
        length=float(dom_meta["length"]),
        n_ssa=int(dom_meta["n_ssa"]),
        pde_multiple=int(dom_meta["pde_multiple"]),
        boundary=str(dom_meta["boundary"]),
    )
    species = list(meta["species"])

    data = np.load(npz_path)
    time = data["time"]
    ssa = data["ssa"]
    pde = data["pde"]

    # Light validation
    if ssa.shape[1] != domain.K:
        raise ValueError("Loaded SSA array does not match domain.K")
    if pde.shape[1] != domain.n_pde:
        raise ValueError("Loaded PDE array does not match domain.n_pde")

    return SimulationResults(
        time=time,
        ssa=ssa,
        pde=pde,
        domain=domain,
        species=species,
    )
