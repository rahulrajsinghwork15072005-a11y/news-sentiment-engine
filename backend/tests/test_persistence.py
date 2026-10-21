import numpy as np
from fastapi.testclient import TestClient

from app.core.significance import sign_test, wilcoxon_signed_rank
from app.main import app
from app.persistence.store import StudyStore

client = TestClient(app)


def test_store_save_get_list_count():
    store = StudyStore(":memory:")
    sid = store.save("AAPL", {"car": 0.05, "t_stat": 2.1, "p_value": 0.03})
    assert store.count() == 1
    assert store.get(sid)["car"] == 0.05
    rows = store.list("AAPL")
    assert rows[0]["id"] == sid
    assert store.get("missing") is None


def test_wilcoxon_and_sign_test():
    values = [0.02, 0.03, 0.01, 0.04, -0.005, 0.02]
    _, w_p = wilcoxon_signed_rank(values)
    assert 0.0 <= w_p <= 1.0
    result = sign_test(values)
    assert result["positive"] == 5
    assert result["n"] == 6


def _prices(seed: int, n: int = 60) -> list[float]:
    rng = np.random.default_rng(seed)
    return (100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))).tolist()


def test_studies_endpoints_roundtrip():
    payload = {
        "asset_prices": _prices(1),
        "market_prices": _prices(2),
        "event_index": 40,
        "est_len": 20,
        "est_gap": 5,
        "pre": 1,
        "post": 5,
        "ticker": "AAPL",
    }
    created = client.post("/api/studies", json=payload)
    assert created.status_code == 200
    sid = created.json()["id"]

    listed = client.get("/api/studies", params={"ticker": "AAPL"}).json()["studies"]
    assert any(s["id"] == sid for s in listed)

    fetched = client.get(f"/api/studies/{sid}").json()
    assert "car" in fetched
    assert client.get("/api/studies/does-not-exist").status_code == 404


def test_significance_endpoint():
    r = client.post("/api/significance", json={"car_values": [0.02, 0.03, -0.01, 0.04, 0.01]})
    assert r.status_code == 200
    body = r.json()
    assert "wilcoxon" in body
    assert "sign_test" in body
