import numpy as np
import pytest

from app.core.decay import fit_decay


def test_fit_recovers_known_decay():
    t = np.arange(0, 10)
    y = 5.0 * np.exp(-0.3 * t)
    fit = fit_decay(t, y)
    assert fit.decay_rate == pytest.approx(0.3, abs=1e-3)
    assert fit.s0 == pytest.approx(5.0, abs=1e-2)
    assert fit.half_life == pytest.approx(np.log(2) / 0.3, rel=1e-2)


def test_constant_series_has_infinite_half_life():
    t = np.arange(0, 6)
    y = np.full_like(t, 0.02, dtype=float)
    fit = fit_decay(t, y)
    # a flat series has a negligible decay rate and an effectively unbounded half-life
    assert fit.decay_rate == pytest.approx(0.0, abs=1e-3)
    assert fit.half_life > 100


def test_requires_two_points():
    with pytest.raises(ValueError):
        fit_decay([0], [1.0])
