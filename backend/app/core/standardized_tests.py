"""Standardized event-study significance tests (Patell, BMP).

The gold-standard parametric tests for abnormal-return significance. Each event's
cumulative abnormal return (CAR) is standardized by its own forecast standard
error, then aggregated across events:

* **Patell (1976)** sums the standardized CARs and divides by the aggregate
  variance, treating each standardized CAR as approximately ``N(0, 1)``.
* **BMP / Boehmer-Musumeci-Poulsen (1991)** divides the mean standardized CAR by
  its cross-sectional standard deviation, making the test robust to
  event-induced volatility (variance inflation around the event).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class StandardizedTestResult:
    z: float
    p_value: float
    n_events: int
    mean_scar: float


def standardized_car(car: float, resid_std: float, window_len: int) -> float:
    """Standardized cumulative abnormal return: ``CAR / (s * sqrt(L))``."""
    if window_len <= 0:
        raise ValueError("window_len must be positive")
    se = resid_std * np.sqrt(window_len)
    return float(car / se) if se > 0 else 0.0


def patell_test(
    cars: Sequence[float],
    resid_stds: Sequence[float],
    window_lens: Sequence[int],
    est_lens: Sequence[int],
) -> StandardizedTestResult:
    """Patell (1976) standardized-residual test (``Z ~ N(0, 1)`` under the null).

    Each standardized CAR is a t-variate with ``M - 2`` degrees of freedom and so
    has variance ``(M - 2) / (M - 4)``; the Patell statistic divides the sum of
    standardized CARs by the square root of the summed variances.
    """
    n = len(cars)
    if not n == len(resid_stds) == len(window_lens) == len(est_lens):
        raise ValueError("all inputs must have the same length")
    if n == 0:
        return StandardizedTestResult(0.0, 1.0, 0, 0.0)
    scars = [standardized_car(c, s, length) for c, s, length in zip(cars, resid_stds, window_lens)]
    variance = sum((m - 2) / (m - 4) if m > 4 else 1.0 for m in est_lens)
    z = float(np.sum(scars) / np.sqrt(variance)) if variance > 0 else 0.0
    p = float(2.0 * stats.norm.sf(abs(z)))
    return StandardizedTestResult(z, p, n, float(np.mean(scars)))


def bmp_test(
    cars: Sequence[float],
    resid_stds: Sequence[float],
    window_lens: Sequence[int],
) -> StandardizedTestResult:
    """BMP (1991) standardized cross-sectional test (robust to event-induced var).

    Standardizes each CAR, then forms ``sqrt(N) * mean(SCAR) / sd(SCAR)``, which is
    Student-t with ``N - 1`` dof. Unlike Patell, it does not assume the
    cross-sectional variance equals the time-series forecast variance.
    """
    n = len(cars)
    if not n == len(resid_stds) == len(window_lens):
        raise ValueError("all inputs must have the same length")
    if n < 2:
        return StandardizedTestResult(0.0, 1.0, n, 0.0)
    scars = np.array(
        [standardized_car(c, s, length) for c, s, length in zip(cars, resid_stds, window_lens)]
    )
    sd = float(scars.std(ddof=1))
    mean = float(scars.mean())
    if sd <= 0:
        return StandardizedTestResult(0.0, 1.0, n, mean)
    z = float(np.sqrt(n) * mean / sd)
    p = float(2.0 * stats.t.sf(abs(z), n - 1))
    return StandardizedTestResult(z, p, n, mean)
