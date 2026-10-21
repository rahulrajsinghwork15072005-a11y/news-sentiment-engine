"""Return calculations."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def simple_returns(prices: ArrayLike) -> NDArray[np.float64]:
    """Simple (arithmetic) returns. ``r[i] = P[i+1]/P[i] - 1``."""
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("prices must be a 1-D array with at least two points")
    if np.any(p <= 0):
        raise ValueError("prices must be strictly positive")
    return p[1:] / p[:-1] - 1.0


def log_returns(prices: ArrayLike) -> NDArray[np.float64]:
    """Continuously compounded (log) returns."""
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("prices must be a 1-D array with at least two points")
    if np.any(p <= 0):
        raise ValueError("prices must be strictly positive")
    return np.log(p[1:] / p[:-1])
