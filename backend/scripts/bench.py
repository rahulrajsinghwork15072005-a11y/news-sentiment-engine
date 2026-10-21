"""Micro-benchmarks for the news event-study engine.

Times a single market-model event study and the full Patell/BMP standardized
test aggregated across the demo events. Runs entirely offline.

    python scripts/bench.py            # from backend/

Numbers are wall-clock and machine-dependent; treat them as relative signals.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.event_study import run_event_study  # noqa: E402
from app.core.standardized_tests import bmp_test, patell_test  # noqa: E402
from app.data.synthetic import build_demo_dataset  # noqa: E402


def _best(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the event-study engine.")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args(argv)

    ds = build_demo_dataset()
    n_events = len(ds.events)

    print(f"\nNews event-study benchmarks  (events={n_events})")
    print("-" * 56)

    one = _best(
        lambda: run_event_study(ds.asset_ret, ds.market_ret, ds.events[0].ret_index),
        args.repeats,
    )
    print(f"  single event study     {one * 1e3:8.3f} ms   {1.0 / one:>10,.0f} studies/s")

    def _full() -> tuple:
        cars, stds = [], []
        for e in ds.events:
            r = run_event_study(ds.asset_ret, ds.market_ret, e.ret_index, pre=1, post=5)
            cars.append(r.car)
            stds.append(r.resid_std)
        wl = [7] * len(cars)
        el = [120] * len(cars)
        return patell_test(cars, stds, wl, el), bmp_test(cars, stds, wl)

    full = _best(_full, args.repeats)
    patell, bmp = _full()
    print(f"  Patell + BMP ({n_events} ev)   {full * 1e3:8.3f} ms")
    print(f"    Patell Z = {patell.z:+.3f} (p={patell.p_value:.3f})")
    print(f"    BMP    Z = {bmp.z:+.3f} (p={bmp.p_value:.3f})")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
