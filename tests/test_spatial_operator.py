import numpy as np
from srcm_engine.domain import Domain
from srcm_engine.pde import laplacian_1d


def test_laplacian_zero_flux_small():
    dom = Domain(length=1.0, n_ssa=1, pde_multiple=5, boundary="zero-flux")
    L = laplacian_1d(dom)

    expected = np.array([
        [-1,  1,  0,  0,  0],
        [ 1, -2,  1,  0,  0],
        [ 0,  1, -2,  1,  0],
        [ 0,  0,  1, -2,  1],
        [ 0,  0,  0,  1, -1],
    ], dtype=float)

    assert np.array_equal(L, expected)


def test_laplacian_periodic_small():
    dom = Domain(length=1.0, n_ssa=1, pde_multiple=5, boundary="periodic")
    L = laplacian_1d(dom)

    expected = np.array([
        [-2,  1,  0,  0,  1],
        [ 1, -2,  1,  0,  0],
        [ 0,  1, -2,  1,  0],
        [ 0,  0,  1, -2,  1],
        [ 1,  0,  0,  1, -2],
    ], dtype=float)

    assert np.array_equal(L, expected)
