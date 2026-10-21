"""Tests for the Patell and BMP standardized event-study tests."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.standardized_tests import bmp_test, patell_test, standardized_car
from app.main import app

client = TestClient(app)


def test_standardized_car_basic():
    # CAR=0.02, resid_std=0.01, L=4 -> SE = 0.01 * 2 = 0.02 -> SCAR = 1.0
    assert standardized_car(0.02, 0.01, 4) == pytest.approx(1.0)


def test_patell_significant_for_consistent_positive_cars():
    n = 20
    res = patell_test([0.03] * n, [0.01] * n, [5] * n, [120] * n)
    assert res.n_events == n
    assert res.z > 2.0
    assert res.p_value < 0.05


def test_patell_insignificant_for_zero_cars():
    n = 15
    res = patell_test([0.0] * n, [0.01] * n, [5] * n, [120] * n)
    assert abs(res.z) < 1e-9
    assert res.p_value == pytest.approx(1.0, abs=1e-6)


def test_bmp_significant_for_positive_mean_with_variation():
    rng = np.random.default_rng(0)
    cars = (0.02 + 0.003 * rng.standard_normal(25)).tolist()  # positive, varying
    res = bmp_test(cars, [0.01] * 25, [5] * 25)
    assert res.n_events == 25
    assert res.z > 2.0
    assert res.p_value < 0.05


def test_bmp_zero_variance_returns_insignificant():
    res = bmp_test([0.02, 0.02, 0.02], [0.01, 0.01, 0.01], [5, 5, 5])
    assert res.z == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        patell_test([0.01, 0.02], [0.01], [5, 5], [120, 120])


def test_api_demo_standardized_tests():
    resp = client.get("/api/demo/standardized-tests", params={"post": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert "patell" in body and "bmp" in body
    assert body["patell"]["n_events"] == body["n_events"]


def test_api_post_standardized():
    body = {
        "cars": [0.03, 0.025, 0.04, 0.02, 0.035],
        "resid_stds": [0.01, 0.01, 0.012, 0.009, 0.011],
        "window_lens": [5, 5, 5, 5, 5],
        "est_lens": [120, 120, 120, 120, 120],
    }
    resp = client.post("/api/significance/standardized", json=body)
    assert resp.status_code == 200
    out = resp.json()
    assert out["patell"]["z"] > 0
    assert "bmp" in out
