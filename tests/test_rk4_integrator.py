import numpy as np
from srcm_engine.pde import rk4_step


def test_rk4_exponential_decay():
    def rhs(y, t):
        return -y

    y0 = np.array([1.0])
    t0 = 0.0
    dt = 0.1

    y1 = rk4_step(y0, t0, dt, rhs)

    expected = np.exp(-dt)
    assert np.allclose(y1[0], expected, atol=1e-7)


def test_rk4_shape_preserved():
    def rhs(y, t):
        return np.ones_like(y)

    y0 = np.zeros((3, 10))
    y1 = rk4_step(y0, 0.0, 0.5, rhs)

    assert y1.shape == (3, 10)
    assert np.allclose(y1, 0.5)  # since y' = 1
