from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import json
import numpy as np

from srcm_engine.domain import Domain
from srcm_engine.results.simulation_results import SimulationResults
from srcm_engine.results.ssa_results import SSAResults

PathLike = Union[str, Path]
ResultsLike = Union[SimulationResults, SSAResults]

def _ensure_pde_array(results: ResultsLike) -> np.ndarray:
    """Return a PDE array for saving, materializing zeros if none exists."""
    pde = getattr(results, "pde", None)
    if pde is not None:
        return pde
    
    time = results.time
    ssa = results.ssa
    n_species = int(ssa.shape[0])
    n_steps = int(time.shape[0])
    
    # Use Domain properties to determine PDE grid size
    K = int(getattr(results.domain, "K", ssa.shape[1]))
    pde_mult = int(getattr(results.domain, "pde_multiple", 1))
    n_pde = K * pde_mult
    
    return np.zeros((n_species, n_pde, n_steps), dtype=float)

# ============================================================
# Pair format: <prefix>.npz + <prefix>.json
# ============================================================
def save_results(
    results: ResultsLike,
    path_prefix: PathLike,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save to separate .npz (arrays) and .json (metadata) files."""
    path_prefix = Path(path_prefix)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)

    npz_path = path_prefix.with_suffix(".npz")
    json_path = path_prefix.with_suffix(".json")

    np.savez_compressed(
        npz_path,
        time=results.time,
        ssa=results.ssa,
        pde=_ensure_pde_array(results),
    )

    # Build comprehensive metadata dictionary
    meta_out: Dict[str, Any] = dict(meta or {})
    meta_out.setdefault("species", list(results.species))
    meta_out.setdefault("run_type", (meta or {}).get("run_type", "hybrid"))
    
    # Domain metadata
    meta_out.setdefault("domain", {
        "length": float(results.domain.length),
        "n_ssa": int(results.domain.K),
        "pde_multiple": int(results.domain.pde_multiple),
        "boundary": str(results.domain.boundary),
    })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Results saved to:\n  {npz_path}\n  {json_path}")


def load_results(path_prefix: PathLike) -> Tuple[SimulationResults, Dict[str, Any]]:
    """Load results from separate .npz and .json files."""
    path_prefix = Path(path_prefix)
    npz_path = path_prefix.with_suffix(".npz")
    json_path = path_prefix.with_suffix(".json")

    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"Missing files for prefix: {path_prefix}")

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    dom_meta = meta.get("domain", {})
    domain = Domain(
        length=float(dom_meta["length"]),
        n_ssa=int(dom_meta["n_ssa"]),
        pde_multiple=int(dom_meta["pde_multiple"]),
        boundary=str(dom_meta["boundary"]),
    )

    data = np.load(npz_path, allow_pickle=True)
    res = SimulationResults(
        time=data["time"],
        ssa=data["ssa"],
        pde=data["pde"] if "pde" in data.files else None,
        domain=domain,
        species=list(meta.get("species", [])),
    )
    return res, meta


# ============================================================
# Single-file format: <file>.npz with meta_json
# ============================================================
def save_npz(
    res: SimulationResults,
    path: PathLike,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save to a single self-contained .npz file including reaction metadata."""
    path = str(path)

    # Core data payload
    payload: Dict[str, Any] = {
        "time": res.time,
        "ssa": res.ssa,
        "pde": res.pde,
        "species": np.array(list(res.species), dtype=object),
        "domain_length": float(res.domain.length),
        "n_ssa": int(res.domain.K),
        "pde_multiple": int(res.domain.pde_multiple),
        "boundary": str(res.domain.boundary),
    }

    # Prepare metadata (includes threshold, reactions, etc.)
    meta_out: Dict[str, Any] = dict(meta or {})
    # Ensure species is always in metadata for easy inspection
    if "species" not in meta_out:
        meta_out["species"] = list(res.species)

    payload["meta_json"] = json.dumps(meta_out)

    np.savez_compressed(path, **payload)
    print(f"Results saved to single file: {path}")


def load_npz(path: PathLike) -> Tuple[SimulationResults, Dict[str, Any]]:
    """Load from a self-contained .npz file."""
    data = np.load(str(path), allow_pickle=True)

    species = [str(s) for s in data["species"].tolist()]

    domain = Domain(
        length=float(data["domain_length"]),
        n_ssa=int(data["n_ssa"]),
        pde_multiple=int(data["pde_multiple"]),
        boundary=str(data["boundary"]),
    )

    # Load arrays, ensuring PDE is present
    time = data["time"]
    ssa = data["ssa"]
    pde = data["pde"] if "pde" in data.files else np.zeros((ssa.shape[0], int(domain.n_pde), time.shape[0]))

    # Extract metadata
    meta: Dict[str, Any] = {}
    if "meta_json" in data.files:
        try:
            meta = json.loads(str(data["meta_json"]))
        except (json.JSONDecodeError, TypeError):
            meta = {}

    res = SimulationResults(
        time=time,
        ssa=ssa,
        pde=pde,
        domain=domain,
        species=species,
    )
    return res, meta


# ============================================================
# Convenience wrapper
# ============================================================
def save_all(
    results: SimulationResults,
    path_prefix: PathLike,
    *,
    meta: Optional[Dict[str, Any]] = None,
    also_write_single_npz: bool = True,
) -> None:
    """Save both formats simultaneously."""
    path_prefix = Path(path_prefix)
    save_results(results, path_prefix, meta=meta)

    if also_write_single_npz:
        save_npz(results, path_prefix.with_suffix(".full.npz"), meta=meta)