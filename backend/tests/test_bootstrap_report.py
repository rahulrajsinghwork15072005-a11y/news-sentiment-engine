"""Tests for bootstrap CAAR confidence intervals and the HTML event-study report."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.bootstrap import bootstrap_caar_ci
from app.core.report import event_study_html
from app.main import app

client = TestClient(app)


def test_bootstrap_reproducible_and_brackets_mean():
    cars = [0.01, -0.02, 0.03, 0.04, -0.01, 0.02, 0.05, -0.03]
    a = bootstrap_caar_ci(cars, n_boot=1000, seed=1)
    b = bootstrap_caar_ci(cars, n_boot=1000, seed=1)
    assert a == b
    assert a["ci_low"] <= a["mean_car"] <= a["ci_high"]
    assert a["n_events"] == len(cars)


def test_bootstrap_lower_alpha_is_wider():
    cars = list(np.linspace(-0.05, 0.05, 20))
    wide = bootstrap_caar_ci(cars, n_boot=3000, alpha=0.01, seed=2)
    narrow = bootstrap_caar_ci(cars, n_boot=3000, alpha=0.10, seed=2)
    assert (wide["ci_high"] - wide["ci_low"]) >= (narrow["ci_high"] - narrow["ci_low"])


def test_bootstrap_validation():
    with pytest.raises(ValueError):
        bootstrap_caar_ci([0.01])  # need >= 2 events
    with pytest.raises(ValueError):
        bootstrap_caar_ci([0.01, 0.02], alpha=1.5)


def test_report_html_contains_chart_and_stats():
    result = {
        "event": {"ticker": "AAPL", "headline": "Beats earnings", "sentiment": 0.4},
        "car": 0.0123,
        "t_stat": 2.1,
        "p_value": 0.03,
        "alpha": 0.0001,
        "beta": 1.1,
        "car_path": [0.001, 0.004, 0.009, 0.0123],
    }
    html = event_study_html(result, title="AAPL Study")
    assert "<svg" in html and "<table" in html
    assert "AAPL Study" in html
    assert "+1.23%" in html


def test_caar_ci_endpoint():
    body = client.get("/api/demo/caar-ci?post=5&n_boot=500").json()
    assert {"mean_car", "ci_low", "ci_high", "n_events"} <= set(body)
    assert body["n_events"] >= 2
    assert len(body["cars"]) == body["n_events"]


def test_report_endpoint_returns_html():
    resp = client.get("/api/demo/report?event_number=0")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<svg" in resp.text


def test_report_endpoint_bad_event_returns_404():
    assert client.get("/api/demo/report?event_number=9999").status_code == 404
