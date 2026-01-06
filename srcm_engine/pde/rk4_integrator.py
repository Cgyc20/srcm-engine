from __future__ import annotations
from typing import Callable
import numpy as np

RHSFn = Callable[[np.ndarray, float], np.ndarray]
# Signature: rhs(state, t) -> dstate/dt, same shape as state


def rk4_step(y: np.ndarray, t: float, dt: float, rhs: RHSFn) -> np.ndarray:
    """
    Perform one RK4 step for an ODE system y' = rhs(y, t).

    Parameters
    ----------
    y : np.ndarray
        Current state, shape arbitrary (for us: (n_species, Npde)).
    t : float
        Current time.
    dt : float
        Time step.
    rhs : callable
        Function rhs(y, t) returning dy/dt with same shape as y.

    Returns
    -------
    y_next : np.ndarray
        State at time t + dt.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")

    k1 = rhs(y, t)
    k2 = rhs(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(y + dt * k3, t + dt)

    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
