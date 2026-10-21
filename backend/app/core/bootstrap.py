"""Bootstrap confidence intervals for the cross-sectional mean CAR (CAAR)."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["bootstrap_caar_ci"]


def bootstrap_caar_ci(
    car_values: ArrayLike, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Percentile bootstrap CI for the mean CAR across events.

    Resamples the per-event CARs with replacement ``n_boot`` times and returns the
    observed mean plus the ``(alpha/2, 1 - alpha/2)`` percentile interval and the
    bootstrap standard error.
    """
    cars = np.asarray(car_values, dtype=float).ravel()
    if cars.size < 2:
        raise ValueError("need at least 2 events to bootstrap")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, cars.size, size=(n_boot, cars.size))
    boot_means = cars[idx].mean(axis=1)
    return {
        "mean_car": float(cars.mean()),
        "ci_low": float(np.quantile(boot_means, alpha / 2.0)),
        "ci_high": float(np.quantile(boot_means, 1.0 - alpha / 2.0)),
        "boot_std_error": float(boot_means.std(ddof=1)),
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "n_events": int(cars.size),
    }
