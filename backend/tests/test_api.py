from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_dashboard_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "News Sentiment" in r.text


def test_sentiment_endpoint():
    r = client.post("/api/sentiment/score", json={"text": "profits surge on strong demand"})
    assert r.status_code == 200
    assert r.json()["label"] == "positive"


def test_tickers_endpoint():
    r = client.post("/api/tickers/extract", json={"text": "Apple beat estimates"})
    assert r.status_code == 200
    assert "AAPL" in r.json()["tickers"]


def test_demo_overview():
    r = client.get("/api/demo/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["n_events"] > 0
    assert len(body["events"]) == body["n_events"]


def test_demo_caar_significant():
    r = client.get("/api/demo/caar")
    assert r.status_code == 200
    body = r.json()
    # positive-news events should show a positive average CAR
    assert body["positive"]["mean_car"] > 0


def test_demo_correlation_positive():
    r = client.get("/api/demo/correlation")
    assert r.status_code == 200
    # sentiment drives the injected reaction, so correlation must be positive
    assert r.json()["pearson_r"] > 0.3


def test_demo_decay_present():
    r = client.get("/api/demo/decay")
    assert r.status_code == 200
    assert "fit" in r.json()
