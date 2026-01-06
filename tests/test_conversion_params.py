import numpy as np
from srcm_engine.conversion import ConversionParams


def test_exceeds_threshold_mask():
    conv = ConversionParams(threshold=5, rate=1.0)

    combined = np.array([[0, 5, 6],
                          [10, 2, 5]])

    mask = conv.exceeds_threshold_mask(combined)

    expected = np.array([[0, 0, 1],
                          [1, 0, 0]])

    assert np.array_equal(mask, expected)


def test_sufficient_pde_mass_mask():
    conv = ConversionParams(threshold=10, rate=1.0)

    h = 0.5
    pde_multiple = 2

    # shape (n_species=1, Npde=4)
    pde = np.array([[2.0, 2.0, 1.9, 2.0]])  # threshold = 1/h = 2.0

    mask = conv.sufficient_pde_mass_mask(pde, pde_multiple, h)

    expected = np.array([[1, 0]])
    assert np.array_equal(mask, expected)
