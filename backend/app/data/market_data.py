"""Real market-data adapter (yfinance) with a deterministic synthetic fallback.

``allow_network=False`` (or a missing ``yfinance`` / failed request) always falls
back to a reproducible synthetic series, so the rest of the system — and the test
suite — runs identically offline.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass

import numpy as np

try:  # optional dependency
    import yfinance as _yf

    _HAS_YF = True
except Exception:  # pragma: no cover - import guard
    _HAS_YF = False


@dataclass(frozen=True)
class PriceSeries:
    ticker: str
    dates: list[str]
    prices: list[float]
    source: str  # "yfinance" | "synthetic"


def _stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "little")


def synthetic_prices(ticker: str, n: int = 252, mu: float = 0.0004, sigma: float = 0.012) -> PriceSeries:
    """Reproducible geometric-Brownian price path keyed on the ticker symbol."""
    rng = np.random.default_rng(_stable_seed(ticker))
    returns = rng.normal(mu, sigma, n)
    prices = 100.0 * np.cumprod(1.0 + returns)
    start = _dt.date(2024, 1, 1)
    dates = [(start + _dt.timedelta(days=i)).isoformat() for i in range(n)]
    return PriceSeries(ticker, dates, prices.tolist(), "synthetic")


def fetch_prices(
    ticker: str, period: str = "1y", interval: str = "1d", allow_network: bool = True
) -> PriceSeries:
    """Fetch real daily closes via yfinance, or fall back to a synthetic series."""
    if allow_network and _HAS_YF:
        try:
            frame = _yf.Ticker(ticker).history(period=period, interval=interval)
            closes = frame["Close"].dropna()
            if len(closes) > 30:
                dates = [idx.date().isoformat() for idx in closes.index]
                return PriceSeries(ticker, dates, [float(v) for v in closes.values], "yfinance")
        except Exception:  # pragma: no cover - network/parse failures fall through
            pass
    return synthetic_prices(ticker)


def aligned_closes(a: PriceSeries, b: PriceSeries) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Inner-join two series on common dates; returns (dates, a_prices, b_prices)."""
    b_map = dict(zip(b.dates, b.prices))
    dates, av, bv = [], [], []
    for d, p in zip(a.dates, a.prices):
        if d in b_map:
            dates.append(d)
            av.append(p)
            bv.append(b_map[d])
    return dates, np.asarray(av, dtype=float), np.asarray(bv, dtype=float)
