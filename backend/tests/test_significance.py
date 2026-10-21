import numpy as np
import pytest

from app.core.significance import caar_t_stat, car_p_value, car_t_stat


def test_car_t_stat_exact():
    # se = resid_std * sqrt(L) = 0.01 * 2 = 0.02 -> t = 0.02 / 0.02 = 1.0
    assert car_t_stat(0.02, resid_std=0.01, window_len=4) == pytest.approx(1.0)


def test_car_t_stat_zero_se():
    assert car_t_stat(0.02, resid_std=0.0, window_len=4) == 0.0


def test_caar_t_stat_known_values():
    mean, t, p, n = caar_t_stat([0.01, 0.02, 0.03])
    assert n == 3
    assert mean == pytest.approx(0.02)
    assert t == pytest.approx(0.02 / (0.01 / np.sqrt(3)))  # ~3.464
    assert 0.0 < p < 1.0


def test_caar_t_stat_constant_is_insignificant():
    mean, t, p, n = caar_t_stat([0.02, 0.02, 0.02])
    assert t == 0.0
    assert p == 1.0


def test_car_p_value_symmetry():
    assert car_p_value(2.0, dof=100) == pytest.approx(car_p_value(-2.0, dof=100))
