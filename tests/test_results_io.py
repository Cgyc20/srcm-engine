import numpy as np
from pathlib import Path

from srcm_engine.domain import Domain
from srcm_engine.results import SimulationResults, save_results, load_results


def test_save_and_load_results_roundtrip(tmp_path: Path):
    domain = Domain(length=1.0, n_ssa=3, pde_multiple=2, boundary="periodic")
    species = ["U", "V"]

    time = np.array([0.0, 0.5, 1.0])
    ssa = np.random.default_rng(0).integers(0, 5, size=(2, domain.K, time.size))
    pde = np.random.default_rng(1).random(size=(2, domain.n_pde, time.size))

    res = SimulationResults(time=time, ssa=ssa, pde=pde, domain=domain, species=species)

    prefix = tmp_path / "run1"
    save_results(res, prefix)

    loaded = load_results(prefix)

    assert loaded.species == species
    assert loaded.domain == domain

    assert np.allclose(loaded.time, time)
    assert np.allclose(loaded.ssa, ssa)
    assert np.allclose(loaded.pde, pde)
