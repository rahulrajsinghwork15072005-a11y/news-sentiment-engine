import numpy as np
import pytest

from app.core.event_study import (
    abnormal_returns,
    aggregate_caar,
    average_abnormal_returns,
    car_path,
    run_event_study,
)


def test_abnormal_and_car_path():
    ar = abnormal_returns([0.02, 0.01, -0.01], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(ar, [0.02, 0.01, -0.01])
    np.testing.assert_allclose(car_path(ar), [0.02, 0.03, 0.02])


def test_run_event_study_recovers_injected_car():
    rng = np.random.default_rng(0)
    n = 200
    market = rng.normal(0.0, 0.01, n)
    asset = market + rng.normal(0.0, 1e-4, n)  # beta ~ 1, alpha ~ 0
    event_idx, bump = 150, 0.02
    asset[event_idx] += bump

    res = run_event_study(asset, market, event_idx, est_len=120, est_gap=10, pre=1, post=5)

    assert res.beta == pytest.approx(1.0, abs=0.05)
    assert res.car == pytest.approx(bump, abs=5e-3)
    assert len(res.ar) == 7  # days -1..+5


def test_event_window_leakage_guard():
    market = np.zeros(200)
    asset = np.zeros(200)
    # est_gap < pre -> event window overlaps estimation window -> must raise
    with pytest.raises(ValueError):
        run_event_study(asset, market, 150, est_len=120, est_gap=2, pre=5, post=5)


def test_aggregate_caar():
    windows = [np.array([0.01, 0.02, 0.0]), np.array([0.03, 0.0, 0.0])]
    res = aggregate_caar(windows, pre=1, post=1)
    np.testing.assert_allclose(res.aar, [0.02, 0.01, 0.0])
    np.testing.assert_allclose(res.caar_path, [0.02, 0.03, 0.03])
    assert res.n_events == 2
    assert res.mean_car == pytest.approx(0.03)


def test_average_abnormal_returns():
    m = [[0.01, 0.03], [0.03, 0.01]]
    np.testing.assert_allclose(average_abnormal_returns(m), [0.02, 0.02])
