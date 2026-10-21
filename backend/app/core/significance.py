"""Statistical significance tests for abnormal returns."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def car_t_stat(car: float, resid_std: float, window_len: int) -> float:
    """t-statistic for a single-event CAR.

    The standard error of a ``window_len``-day CAR is ``resid_std * sqrt(L)``
    under the assumption of i.i.d. abnormal returns.
    """
    if window_len <= 0:
        raise ValueError("window_len must be positive")
    se = resid_std * np.sqrt(window_len)
    if se <= 0:
        return 0.0
    return float(car / se)


def car_p_value(t_stat: float, dof: int) -> float:
    """Two-sided p-value from a Student-t distribution."""
    dof = max(int(dof), 1)
    return float(2.0 * stats.t.sf(abs(t_stat), dof))


def caar_t_stat(car_values: ArrayLike) -> tuple[float, float, float, int]:
    """Cross-sectional test of the average CAR across events.

    Returns ``(mean_car, t_stat, p_value, n_events)``.
    """
    values = np.asarray(car_values, dtype=float)
    n = int(values.size)
    if n == 0:
        return 0.0, 0.0, 1.0, 0
    mean = float(values.mean())
    if n == 1:
        return mean, 0.0, 1.0, 1
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(n)
    if se <= 0:
        return mean, 0.0, 1.0, n
    t = mean / se
    p = float(2.0 * stats.t.sf(abs(t), n - 1))
    return mean, float(t), p, n


def wilcoxon_signed_rank(car_values: ArrayLike) -> tuple[float, float]:
    """Non-parametric Wilcoxon signed-rank test on event CARs (stat, p_value)."""
    values = np.asarray(car_values, dtype=float)
    if values.size < 1 or np.allclose(values, 0.0):
        return 0.0, 1.0
    try:
        result = stats.wilcoxon(values)
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return 0.0, 1.0


def sign_test(car_values: ArrayLike) -> dict[str, float]:
    """Binomial sign test: are positive CARs more frequent than chance?"""
    values = np.asarray(car_values, dtype=float)
    n = int(values.size)
    if n == 0:
        return {"positive": 0, "n": 0, "p_value": 1.0}
    positive = int((values > 0).sum())
    p_value = float(stats.binomtest(positive, n, 0.5).pvalue)
    return {"positive": positive, "n": n, "p_value": p_value}

