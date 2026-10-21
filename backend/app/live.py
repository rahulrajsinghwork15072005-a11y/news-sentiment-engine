"""Live-data orchestration: real prices + news wired into the existing core."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .core.event_study import run_event_study
from .core.returns import simple_returns
from .data.market_data import aligned_closes, fetch_prices
from .data.news_feed import fetch_headlines
from .nlp.sentiment import score_text
from .nlp.tickers import extract_tickers


def live_prices(ticker: str, allow_network: bool = True) -> dict:
    series = fetch_prices(ticker, allow_network=allow_network)
    return {
        "ticker": series.ticker,
        "source": series.source,
        "dates": series.dates,
        "prices": series.prices,
    }


def live_news(ticker: str, allow_network: bool = True) -> dict:
    headlines = fetch_headlines(ticker, allow_network=allow_network)
    items = []
    for h in headlines:
        scored = score_text(h.title)
        items.append(
            {
                "title": h.title,
                "published": h.published,
                "source": h.source,
                "sentiment": scored.score,
                "label": scored.label,
                "tickers": extract_tickers(h.title),
            }
        )
    avg = float(np.mean([i["sentiment"] for i in items])) if items else 0.0
    return {"ticker": ticker, "count": len(items), "avg_sentiment": avg, "headlines": items}


def live_study(
    ticker: str,
    benchmark: str = "SPY",
    est_len: int = 120,
    est_gap: int = 10,
    pre: int = 1,
    post: int = 5,
    allow_network: bool = True,
) -> dict:
    asset = fetch_prices(ticker, allow_network=allow_network)
    bench = fetch_prices(benchmark, allow_network=allow_network)
    dates, asset_p, bench_p = aligned_closes(asset, bench)
    if asset_p.size < est_len + est_gap + pre + post + 5:
        raise ValueError("not enough overlapping history for an event study")

    asset_ret = simple_returns(asset_p)
    market_ret = simple_returns(bench_p)
    n = asset_ret.size
    lo, hi = est_len + est_gap, n - post
    if hi <= lo:
        raise ValueError("insufficient window for an event study")

    # auto-select the most extreme abnormal day as the event
    event_idx = int(lo + np.argmax(np.abs(asset_ret[lo:hi])))
    result = run_event_study(
        asset_ret, market_ret, event_idx, est_len=est_len, est_gap=est_gap, pre=pre, post=post
    )
    event_date = dates[event_idx + 1] if event_idx + 1 < len(dates) else dates[event_idx]
    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "source": asset.source,
        "event_date": event_date,
        "event_return": float(asset_ret[event_idx]),
        **asdict(result),
    }
