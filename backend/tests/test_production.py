from fastapi.testclient import TestClient

from app.main import app
from app.observability import RateLimiter

client = TestClient(app)


def test_version_endpoint():
    body = client.get("/api/version").json()
    assert body["name"] == "news-sentiment-engine"
    assert "version" in body


def test_ready_endpoint():
    body = client.get("/api/ready").json()
    assert body["ready"] is True
    assert body["uptime_seconds"] >= 0


def test_metrics_endpoint():
    # generate at least one request first
    client.get("/api/health")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text


def test_security_headers_and_request_id():
    r = client.get("/api/health")
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"
    assert "x-request-id" in r.headers
    assert "x-response-time-ms" in r.headers


def test_rate_limiter_allows_then_blocks():
    limiter = RateLimiter(limit=2, window=60.0)
    assert limiter.allow("client") is True
    assert limiter.allow("client") is True
    assert limiter.allow("client") is False  # third within window blocked
    assert limiter.allow("other") is True     # different key independent


def test_rate_limiter_disabled_when_zero():
    limiter = RateLimiter(limit=0)
    assert all(limiter.allow("x") for _ in range(100))
