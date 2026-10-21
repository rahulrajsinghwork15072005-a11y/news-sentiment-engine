"""Synthetic market + news generator.

Events inject decaying abnormal returns proportional to the *scored* sentiment of
their headline, so the downstream event study, correlation, and decay analyses
all operate on an internally consistent, reproducible dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from ..nlp.sentiment import score_text

_POSITIVE_TEMPLATES = [
    "{co} surges after earnings beat and strong guidance",
    "{co} jumps as profit growth exceeds expectations",
    "Analysts upgrade {co}; shares rally to a record high",
    "{co} soars on robust demand and a raised outlook",
]
_NEGATIVE_TEMPLATES = [
    "{co} plunges after an earnings miss and weak guidance",
    "{co} tumbles as losses widen and the outlook is cut",
    "Analysts downgrade {co}; shares slump on a demand warning",
    "{co} drops on a fraud investigation and lawsuit fears",
]
_COMPANIES = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Nvidia": "NVDA",
    "Amazon": "AMZN",
    "Microsoft": "MSFT",
}


@dataclass(frozen=True)
class NewsEvent:
    ret_index: int   # index into the returns arrays (the day the move is realized)
    ticker: str
    headline: str
    sentiment: float


@dataclass(frozen=True)
class DemoDataset:
    dates: list[str]
    asset_prices: list[float]
    market_prices: list[float]
    asset_ret: np.ndarray
    market_ret: np.ndarray
    events: list[NewsEvent]


def build_demo_dataset(
    n_returns: int = 320,
    n_events: int = 16,
    seed: int = 7,
    base_effect: float = 0.045,
    decay_gamma: float = 0.5,
) -> DemoDataset:
    """Construct a reproducible synthetic dataset with sentiment-driven events."""
    rng = np.random.default_rng(seed)

    market_ret = rng.normal(0.0003, 0.010, n_returns)
    alpha, beta = 0.0001, 1.1
    idio = rng.normal(0.0, 0.008, n_returns)
    asset_ret = alpha + beta * market_ret + idio

    earliest, latest = 140, n_returns - 12
    event_days = sorted(
        int(d) for d in rng.choice(range(earliest, latest), size=n_events, replace=False)
    )

    companies = list(_COMPANIES.keys())
    events: list[NewsEvent] = []
    for d in event_days:
        positive = bool(rng.random() < 0.5)
        company = companies[int(rng.integers(len(companies)))]
        templates = _POSITIVE_TEMPLATES if positive else _NEGATIVE_TEMPLATES
        headline = templates[int(rng.integers(len(templates)))].format(co=company)
        sentiment = score_text(headline).score

        # inject a decaying abnormal return starting on the event day
        for offset in range(0, 6):
            idx = d + offset
            if idx < n_returns:
                asset_ret[idx] += base_effect * sentiment * np.exp(-decay_gamma * offset)

        events.append(
            NewsEvent(
                ret_index=d,
                ticker=_COMPANIES[company],
                headline=headline,
                sentiment=sentiment,
            )
        )

    asset_prices = 100.0 * np.cumprod(1.0 + asset_ret)
    market_prices = 100.0 * np.cumprod(1.0 + market_ret)
    asset_prices = np.insert(asset_prices, 0, 100.0)
    market_prices = np.insert(market_prices, 0, 100.0)

    start = date(2024, 1, 1)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(n_returns + 1)]

    return DemoDataset(
        dates=dates,
        asset_prices=asset_prices.tolist(),
        market_prices=market_prices.tolist(),
        asset_ret=asset_ret,
        market_ret=market_ret,
        events=events,
    )
