import numpy as np
from srcm_engine.conversion import combined_mass, pde_mass_per_compartment


def test_pde_mass_projection():
    # 1 species, K=2, pde_multiple=3
    pde = np.array([[1.0, 1.0, 1.0,
                     2.0, 2.0, 2.0]])
    dx = 0.1
    pde_multiple = 3

    mass = pde_mass_per_compartment(pde, pde_multiple, dx)

    expected = np.array([[0.3, 0.6]])
    assert np.allclose(mass, expected)


def test_combined_mass():
    ssa = np.array([[1, 2]])
    pde = np.array([[1.0, 1.0, 1.0,
                     2.0, 2.0, 2.0]])
    dx = 0.1
    pde_multiple = 3

    combined, pde_mass = combined_mass(ssa, pde, pde_multiple, dx)

    assert np.allclose(pde_mass, [[0.3, 0.6]])
    assert np.allclose(combined, [[1.3, 2.6]])
