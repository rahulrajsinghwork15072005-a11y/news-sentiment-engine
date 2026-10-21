"""Live-data adapters are tested via their deterministic offline fallback so the
suite never depends on network access."""
from app.data.market_data import fetch_prices, synthetic_prices
from app.data.news_feed import fetch_headlines
from app.live import live_news, live_prices, live_study


def test_synthetic_prices_deterministic():
    a = synthetic_prices("AAPL")
    b = synthetic_prices("AAPL")
    assert a.prices == b.prices
    assert a.source == "synthetic"
    assert len(a.prices) > 30


def test_fetch_prices_offline_falls_back():
    ps = fetch_prices("AAPL", allow_network=False)
    assert ps.source == "synthetic"
    assert all(p > 0 for p in ps.prices)


def test_headlines_offline_fallback():
    heads = fetch_headlines("AAPL", allow_network=False)
    assert len(heads) > 0
    assert all(h.source == "synthetic" for h in heads)


def test_live_news_scores_headlines():
    out = live_news("AAPL", allow_network=False)
    assert out["count"] > 0
    assert all("sentiment" in h for h in out["headlines"])


def test_live_prices_offline():
    out = live_prices("MSFT", allow_network=False)
    assert out["source"] == "synthetic"


def test_live_study_offline_runs():
    out = live_study("AAPL", "SPY", allow_network=False)
    assert "car" in out
    assert "t_stat" in out
    assert out["source"] == "synthetic"
