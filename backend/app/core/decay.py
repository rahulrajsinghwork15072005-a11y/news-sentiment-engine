"""Sentiment-reaction decay fitting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit


def _exp_decay(t: np.ndarray, s0: float, lam: float) -> np.ndarray:
    return s0 * np.exp(-lam * t)


@dataclass(frozen=True)
class DecayFit:
    s0: float
    decay_rate: float
    half_life: float


def fit_decay(times: ArrayLike, magnitudes: ArrayLike) -> DecayFit:
    """Fit ``effect(t) = s0 * exp(-lambda * t)`` and report the half-life.

    ``times`` are event-relative days (>= 0) and ``magnitudes`` are the
    corresponding |average abnormal return| values.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(magnitudes, dtype=float)
    if t.shape != y.shape or t.size < 2:
        raise ValueError("times and magnitudes must be equal-length arrays of length >= 2")

    s0_guess = float(max(y[0], 1e-6))
    try:
        popt, _ = curve_fit(
            _exp_decay,
            t,
            y,
            p0=[s0_guess, 0.1],
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            maxfev=10000,
        )
        s0, lam = float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        s0, lam = s0_guess, 0.0

    half_life = float(np.log(2.0) / lam) if lam > 0 else float("inf")
    return DecayFit(s0=s0, decay_rate=lam, half_life=half_life)
