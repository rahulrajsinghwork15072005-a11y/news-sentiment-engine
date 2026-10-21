"""Event-study computations: abnormal returns, CAR, AAR, CAAR."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .expected_return import (
    MarketModel,
    fit_market_model,
    market_model_expected,
)
from .significance import car_p_value, car_t_stat, caar_t_stat


def abnormal_returns(asset_ret: ArrayLike, expected_ret: ArrayLike) -> NDArray[np.float64]:
    """``AR = R_actual - E[R]``."""
    a = np.asarray(asset_ret, dtype=float)
    e = np.asarray(expected_ret, dtype=float)
    if a.shape != e.shape:
        raise ValueError("asset_ret and expected_ret must have the same shape")
    return a - e


def car_path(ar: ArrayLike) -> NDArray[np.float64]:
    """Cumulative abnormal return path."""
    return np.cumsum(np.asarray(ar, dtype=float))


def average_abnormal_returns(ar_matrix: ArrayLike) -> NDArray[np.float64]:
    """Average abnormal return at each relative day across events."""
    m = np.atleast_2d(np.asarray(ar_matrix, dtype=float))
    return m.mean(axis=0)


@dataclass(frozen=True)
class EventStudyResult:
    relative_days: list[int]
    ar: list[float]
    car_path: list[float]
    car: float
    t_stat: float
    p_value: float
    alpha: float
    beta: float
    resid_std: float


def run_event_study(
    asset_ret: ArrayLike,
    market_ret: ArrayLike,
    event_idx: int,
    est_len: int = 120,
    est_gap: int = 10,
    pre: int = 1,
    post: int = 5,
) -> EventStudyResult:
    """Run a single-event market-model study.

    All indices are in *returns* space. The estimation window is
    ``[event_idx - est_gap - est_len, event_idx - est_gap)`` and the event
    window is ``[event_idx - pre, event_idx + post]`` (inclusive).
    """
    asset = np.asarray(asset_ret, dtype=float)
    market = np.asarray(market_ret, dtype=float)
    if asset.shape != market.shape:
        raise ValueError("asset_ret and market_ret must have equal length")

    est_end = event_idx - est_gap
    est_start = est_end - est_len
    if est_start < 0:
        raise ValueError("not enough history for the estimation window")
    w_start = event_idx - pre
    w_end = event_idx + post
    if w_start < 0 or w_end >= asset.size:
        raise ValueError("event window falls outside the available data")
    if w_start < est_end:
        raise ValueError("event window overlaps the estimation window (look-ahead leakage)")

    model: MarketModel = fit_market_model(asset[est_start:est_end], market[est_start:est_end])
    window = np.arange(w_start, w_end + 1)
    expected = market_model_expected(model, market[window])
    ar = asset[window] - expected
    path = car_path(ar)
    car = float(path[-1])
    window_len = int(window.size)
    t = car_t_stat(car, model.resid_std, window_len)
    p = car_p_value(t, dof=max(est_len - 2, 1))

    return EventStudyResult(
        relative_days=list(range(-pre, post + 1)),
        ar=ar.tolist(),
        car_path=path.tolist(),
        car=car,
        t_stat=t,
        p_value=p,
        alpha=model.alpha,
        beta=model.beta,
        resid_std=model.resid_std,
    )


@dataclass(frozen=True)
class CaarResult:
    relative_days: list[int]
    aar: list[float]
    caar_path: list[float]
    mean_car: float
    t_stat: float
    p_value: float
    n_events: int


def aggregate_caar(ar_windows: Sequence[ArrayLike], pre: int, post: int) -> CaarResult:
    """Aggregate per-event AR windows into AAR / CAAR with a cross-sectional test."""
    matrix = np.vstack([np.asarray(w, dtype=float) for w in ar_windows])
    if matrix.shape[1] != (pre + post + 1):
        raise ValueError("AR window width does not match pre + post + 1")
    aar = matrix.mean(axis=0)
    caar = np.cumsum(aar)
    car_values = matrix.sum(axis=1)
    mean_car, t, p, n = caar_t_stat(car_values)
    return CaarResult(
        relative_days=list(range(-pre, post + 1)),
        aar=aar.tolist(),
        caar_path=caar.tolist(),
        mean_car=mean_car,
        t_stat=t,
        p_value=p,
        n_events=n,
    )
