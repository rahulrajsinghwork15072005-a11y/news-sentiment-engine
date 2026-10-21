from app.nlp.tickers import extract_tickers


def test_cashtag():
    assert "AAPL" in extract_tickers("$AAPL is up today")


def test_company_names():
    out = extract_tickers("Apple and Tesla both rallied")
    assert "AAPL" in out
    assert "TSLA" in out


def test_unknown_cashtag_ignored():
    assert extract_tickers("$ZZZZ to the moon") == []


def test_no_tickers():
    assert extract_tickers("the market was quiet today") == []


def test_sorted_unique():
    out = extract_tickers("Tesla Tesla $TSLA Apple")
    assert out == ["AAPL", "TSLA"]
