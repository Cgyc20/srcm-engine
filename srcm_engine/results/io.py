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
    """Return a PDE array for saving.

    Hybrid runs carry a real PDE array. Pure SSA runs may have `pde=None`.
    For compatibility, we materialize a zero PDE array with the right shape.
    """
    pde = getattr(results, "pde", None)
    if pde is not None:
        return pde
    # Infer shape from SSA + domain
    time = results.time
    ssa = results.ssa
    n_species = int(ssa.shape[0])
    n_steps = int(time.shape[0])
    n_pde = int(getattr(results.domain, "n_pde", 0))
    if n_pde <= 0:
        # Fallback: if domain lacks n_pde, infer from K * pde_multiple when present
        K = int(getattr(results.domain, "K", getattr(results.domain, "n_ssa", ssa.shape[1])))
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
    """
    Save SimulationResults to:
      - <path_prefix>.npz  (arrays)
      - <path_prefix>.json (metadata)

    Notes
    -----
    - Metadata is written ONLY to the json, not injected into SimulationResults.
    """
    path_prefix = Path(path_prefix)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)

    npz_path = path_prefix.with_suffix(".npz")
    json_path = path_prefix.with_suffix(".json")

    # arrays
    np.savez_compressed(
        npz_path,
        time=results.time,
        ssa=results.ssa,
        pde=_ensure_pde_array(results),
    )

    # metadata
    meta_out: Dict[str, Any] = dict(meta or {})
    meta_out.setdefault("species", list(results.species))
    meta_out.setdefault("run_type", (meta or {}).get("run_type", "hybrid"))
    meta_out.setdefault(
        "domain",
        {
            "length": float(results.domain.length),
            "n_ssa": int(results.domain.K),
            "pde_multiple": int(results.domain.pde_multiple),
            "boundary": str(results.domain.boundary),
        },
    )
    meta_out.setdefault(
        "shapes",
        {
            "time": list(results.time.shape),
            "ssa": list(results.ssa.shape),
            "pde": list(_ensure_pde_array(results).shape),
        },
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Results saved to:\n  {npz_path}\n  {json_path}")


def load_results(path_prefix: PathLike) -> Tuple[SimulationResults, Dict[str, Any]]:
    """
    Load SimulationResults saved by save_results().

    Returns
    -------
    (SimulationResults, meta_dict)
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

    dom_meta = meta.get("domain", {})
    domain = Domain(
        length=float(dom_meta["length"]),
        n_ssa=int(dom_meta["n_ssa"]),
        pde_multiple=int(dom_meta["pde_multiple"]),
        boundary=str(dom_meta["boundary"]),
    )

    species = list(meta.get("species", []))

    data = np.load(npz_path, allow_pickle=True)
    time = data["time"]
    ssa = data["ssa"]
    pde = data["pde"] if "pde" in data.files else None

    if pde is None:
        # Pure SSA files may omit PDE output; create zeros for compatibility
        n_species = int(ssa.shape[0])
        n_steps = int(time.shape[0])
        n_pde = int(getattr(domain, "n_pde", 0))
        if n_pde <= 0:
            n_pde = int(domain.K) * int(domain.pde_multiple)
        pde = np.zeros((n_species, n_pde, n_steps), dtype=float)

    # Light validation
    if ssa.shape[1] != domain.K:
        raise ValueError("Loaded SSA array does not match domain.K")
    if pde.shape[1] != domain.n_pde:
        raise ValueError("Loaded PDE array does not match domain.n_pde")

    res = SimulationResults(
        time=time,
        ssa=ssa,
        pde=pde,
        domain=domain,
        species=species,
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
    """
    Save SimulationResults to a single self-contained .npz file.

    Stores:
      - time, ssa, pde arrays
      - domain + species
      - meta_json: JSON string containing *all* experiment metadata
    """
    path = str(path)

    payload: Dict[str, Any] = {
        "time": res.time,
        "ssa": res.ssa,
        "pde": res.pde,
        "species": np.array(list(res.species), dtype=object),

        # Domain metadata (scalars)
        "domain_length": float(res.domain.length),
        "n_ssa": int(res.domain.K),
        "pde_multiple": int(res.domain.pde_multiple),
        "boundary": str(res.domain.boundary),
    }

    meta_out: Dict[str, Any] = dict(meta or {})
    payload["meta_json"] = json.dumps(meta_out)

    np.savez_compressed(path, **payload)
    print(f"Results saved to single file: {path}")


def load_npz(path: PathLike) -> Tuple[SimulationResults, Dict[str, Any]]:
    """
    Load a self-contained .npz file produced by save_npz().

    Returns
    -------
    (SimulationResults, meta_dict)
    """
    data = np.load(str(path), allow_pickle=True)

    species = [str(s) for s in data["species"].tolist()]

    domain = Domain(
        length=float(data["domain_length"]),
        n_ssa=int(data["n_ssa"]),
        pde_multiple=int(data["pde_multiple"]),
        boundary=str(data["boundary"]),
    )

    res = SimulationResults(
        time=data["time"],
        ssa=data["ssa"],
        pde=(data["pde"] if "pde" in data.files else np.zeros((data["ssa"].shape[0], int(domain.n_pde), data["time"].shape[0]), dtype=float)),
        domain=domain,
        species=species,
    )

    meta: Dict[str, Any] = {}
    if "meta_json" in data.files:
        try:
            meta = json.loads(str(data["meta_json"]))
            if not isinstance(meta, dict):
                meta = {}
        except Exception:
            meta = {}

    return res, meta


# ============================================================
# Convenience wrapper: save BOTH formats with one call
# ============================================================
def save_all(
    results: SimulationResults,
    path_prefix: PathLike,
    *,
    meta: Optional[Dict[str, Any]] = None,
    also_write_single_npz: bool = True,
) -> None:
    """
    Save:
      - <prefix>.npz + <prefix>.json
      - optionally also <prefix>.full.npz (single file)
    """
    path_prefix = Path(path_prefix)
    save_results(results, path_prefix, meta=meta)

    if also_write_single_npz:
        save_npz(results, path_prefix.with_suffix(".full.npz"), meta=meta)