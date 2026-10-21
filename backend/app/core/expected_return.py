"""Expected (normal) return models used as the event-study baseline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class MarketModel:
    """OLS market model ``R_i = alpha + beta * R_m + eps``."""

    alpha: float
    beta: float
    resid_std: float
    n_obs: int


def fit_market_model(asset_ret: ArrayLike, market_ret: ArrayLike) -> MarketModel:
    """Fit ``alpha``/``beta`` by ordinary least squares over an estimation window."""
    y = np.asarray(asset_ret, dtype=float)
    x = np.asarray(market_ret, dtype=float)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("asset_ret and market_ret must be 1-D arrays of equal length")
    if y.size < 3:
        raise ValueError("need at least 3 observations to fit the market model")

    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - design @ coef
    dof = max(y.size - 2, 1)
    resid_std = float(np.sqrt(np.sum(resid**2) / dof))
    return MarketModel(alpha=alpha, beta=beta, resid_std=resid_std, n_obs=int(y.size))


def market_model_expected(model: MarketModel, market_ret: ArrayLike) -> NDArray[np.float64]:
    """Expected returns implied by a fitted market model."""
    x = np.asarray(market_ret, dtype=float)
    return model.alpha + model.beta * x


def mean_adjusted_expected(estimation_ret: ArrayLike, n: int) -> NDArray[np.float64]:
    """Constant expected return equal to the estimation-window mean."""
    mean = float(np.mean(np.asarray(estimation_ret, dtype=float)))
    return np.full(n, mean, dtype=float)


def market_adjusted_expected(market_ret: ArrayLike) -> NDArray[np.float64]:
    """Expected return equals the benchmark return (beta = 1, alpha = 0)."""
    return np.asarray(market_ret, dtype=float)
