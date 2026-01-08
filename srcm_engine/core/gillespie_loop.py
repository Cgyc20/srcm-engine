from __future__ import annotations
import numpy as np

def gillespie_draw(
    propensities: np.ndarray,
    rng: np.random.Generator,
    cumulative: np.ndarray | None = None,
    *,
    check_negative: bool = False,
):
    if propensities.ndim != 1:
        raise ValueError("propensities must be a 1D array")

    # Debug-only: avoid scanning every event in production
    if check_negative:
        # min() is one pass; any(<0) is also one pass, but min is slightly simpler
        mn = float(propensities.min(initial=0.0))
        if mn < 0.0:
            j = int(np.argmin(propensities))
            raise ValueError(f"propensities must be nonnegative: idx={j}, val={propensities[j]}")

    # Preallocate cum array if not provided (still allocates, but you can pass one from caller)
    if cumulative is None:
        cumulative = np.cumsum(propensities)
    else:
        if cumulative.shape != propensities.shape:
            raise ValueError("cumulative must have same shape as propensities")
        np.cumsum(propensities, out=cumulative)

    a0 = float(cumulative[-1])
    if a0 <= 0.0:
        return np.inf, -1

    u1, u2 = rng.random(2)
    tau = np.log(1.0 / u1) / a0
    idx = int(np.searchsorted(cumulative, u2 * a0, side="left"))
    return tau, idx
