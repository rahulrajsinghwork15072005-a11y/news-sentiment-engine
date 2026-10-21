"""Demo orchestration: composes the core engine over the synthetic dataset.

Each function returns plain JSON-serializable dicts ready for the API/dashboard.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

import numpy as np

from .core.bootstrap import bootstrap_caar_ci
from .core.correlation import sentiment_return_correlation
from .core.decay import fit_decay
from .core.event_study import aggregate_caar, run_event_study
from .core.report import event_study_html
from .core.standardized_tests import bmp_test, patell_test
from .data.synthetic import DemoDataset, build_demo_dataset


@lru_cache(maxsize=1)
def dataset() -> DemoDataset:
    return build_demo_dataset()


def demo_overview() -> dict:
    ds = dataset()
    return {
        "n_returns": int(ds.asset_ret.size),
        "n_events": len(ds.events),
        "dates": ds.dates,
        "asset_prices": ds.asset_prices,
        "market_prices": ds.market_prices,
        "events": [
            {
                "ret_index": e.ret_index,
                "date": ds.dates[e.ret_index + 1],
                "ticker": e.ticker,
                "headline": e.headline,
                "sentiment": e.sentiment,
            }
            for e in ds.events
        ],
    }


def demo_event_study(event_number: int = 0, pre: int = 1, post: int = 5) -> dict:
    ds = dataset()
    if not 0 <= event_number < len(ds.events):
        raise IndexError("event_number out of range")
    event = ds.events[event_number]
    result = run_event_study(ds.asset_ret, ds.market_ret, event.ret_index, pre=pre, post=post)
    return {
        "event": {
            "date": ds.dates[event.ret_index + 1],
            "ticker": event.ticker,
            "headline": event.headline,
            "sentiment": event.sentiment,
        },
        **asdict(result),
    }


def demo_caar(pre: int = 1, post: int = 10) -> dict:
    ds = dataset()
    positive: list[list[float]] = []
    negative: list[list[float]] = []
    for event in ds.events:
        result = run_event_study(ds.asset_ret, ds.market_ret, event.ret_index, pre=pre, post=post)
        (positive if event.sentiment >= 0 else negative).append(result.ar)

    out: dict = {"relative_days": list(range(-pre, post + 1))}
    if positive:
        out["positive"] = asdict(aggregate_caar(positive, pre, post))
    if negative:
        out["negative"] = asdict(aggregate_caar(negative, pre, post))
    return out


def demo_correlation(post: int = 5) -> dict:
    ds = dataset()
    sentiments: list[float] = []
    cars: list[float] = []
    for event in ds.events:
        result = run_event_study(ds.asset_ret, ds.market_ret, event.ret_index, pre=0, post=post)
        sentiments.append(event.sentiment)
        cars.append(result.car)
    stats = sentiment_return_correlation(sentiments, cars)
    return {"horizon": post, "sentiment": sentiments, "car": cars, **stats}


def demo_standardized_tests(pre: int = 1, post: int = 5) -> dict:
    """Patell (1976) and BMP (1991) standardized tests over all demo events."""
    ds = dataset()
    cars: list[float] = []
    resid_stds: list[float] = []
    est_len = 120  # run_event_study default estimation window
    for event in ds.events:
        result = run_event_study(
            ds.asset_ret, ds.market_ret, event.ret_index, est_len=est_len, pre=pre, post=post
        )
        cars.append(result.car)
        resid_stds.append(result.resid_std)
    window_len = pre + post + 1
    n = len(cars)
    window_lens = [window_len] * n
    est_lens = [est_len] * n
    return {
        "n_events": n,
        "window_len": window_len,
        "estimation_len": est_len,
        "cars": cars,
        "patell": asdict(patell_test(cars, resid_stds, window_lens, est_lens)),
        "bmp": asdict(bmp_test(cars, resid_stds, window_lens)),
    }


def demo_decay(post: int = 8) -> dict:
    ds = dataset()
    aligned: list[np.ndarray] = []
    for event in ds.events:
        result = run_event_study(ds.asset_ret, ds.market_ret, event.ret_index, pre=0, post=post)
        sign = 1.0 if event.sentiment >= 0 else -1.0
        aligned.append(np.asarray(result.ar, dtype=float) * sign)

    aar = np.vstack(aligned).mean(axis=0)
    times = list(range(0, post + 1))
    fit = fit_decay(times, np.abs(aar))
    return {"relative_days": times, "aar_aligned": aar.tolist(), "fit": asdict(fit)}


def demo_caar_ci(post: int = 5, n_boot: int = 2000) -> dict:
    """Per-event CARs + a percentile bootstrap CI for the mean CAR (CAAR)."""
    ds = dataset()
    cars = [
        run_event_study(ds.asset_ret, ds.market_ret, e.ret_index, pre=0, post=post).car
        for e in ds.events
    ]
    ci = bootstrap_caar_ci(cars, n_boot=n_boot)
    return {"horizon": post, "cars": cars, **ci}


def demo_report(event_number: int = 0, pre: int = 1, post: int = 5) -> str:
    """Render a single-event study as a standalone HTML report."""
    return event_study_html(demo_event_study(event_number, pre=pre, post=post))
