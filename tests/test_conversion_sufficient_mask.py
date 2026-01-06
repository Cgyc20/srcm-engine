# tests/test_sufficient_pde_concentration_mask.py

import numpy as np
import pytest

from srcm_engine.domain import Domain
from srcm_engine.conversion.regime_utils import sufficient_pde_concentration_mask


def test_sufficient_pde_concentration_mask_shape_and_values():
    """
    For each compartment i, mask=1 iff ALL PDE cells in that compartment
    have concentration >= 1/h. Otherwise mask=0.
    """
    dom = Domain(length=2.0, n_ssa=2, pde_multiple=4, boundary="periodic")
    # h = length/n_ssa = 1.0 => threshold = 1/h = 1.0
    threshold = 1.0 / dom.h
    assert threshold == pytest.approx(1.0)

    # 1 species, 8 PDE cells (K=2, pde_multiple=4)
    pde = np.zeros((1, dom.n_pde), dtype=float)

    # compartment 0: all cells exactly at threshold -> sufficient
    pde[0, 0:4] = threshold

    # compartment 1: one cell below threshold -> not sufficient
    pde[0, 4:8] = threshold
    pde[0, 6] = threshold - 1e-3

    mask = sufficient_pde_concentration_mask(pde, dom.pde_multiple, dom.h)

    assert mask.shape == (1, dom.K)
    assert mask.dtype == np.dtype(int)

    assert mask[0, 0] == 1
    assert mask[0, 1] == 0


def test_sufficient_pde_concentration_mask_two_species_independent():
    """
    Species are checked independently per compartment.
    """
    dom = Domain(length=3.0, n_ssa=3, pde_multiple=2, boundary="periodic")
    # h = 1.0 => threshold = 1.0
    threshold = 1.0 / dom.h

    pde = np.zeros((2, dom.n_pde), dtype=float)

    # species 0: all cells sufficient everywhere
    pde[0, :] = threshold

    # species 1: break compartment 2 (last)
    pde[1, :] = threshold
    # compartment 2 covers cells [4:6]
    pde[1, 5] = 0.0

    mask = sufficient_pde_concentration_mask(pde, dom.pde_multiple, dom.h)

    assert mask.shape == (2, dom.K)

    # species 0 all good
    assert np.array_equal(mask[0, :], np.ones(dom.K, dtype=int))

    # species 1 last compartment fails
    assert np.array_equal(mask[1, :], np.array([1, 1, 0], dtype=int))


def test_sufficient_pde_concentration_mask_raises_if_bad_shape():
    """
    Npde must be divisible by pde_multiple.
    """
    pde = np.zeros((1, 7), dtype=float)  # not divisible by pde_multiple=4
    with pytest.raises(ValueError):
        sufficient_pde_concentration_mask(pde, pde_multiple=4, h=1.0)
