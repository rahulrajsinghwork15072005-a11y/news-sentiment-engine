
from app.nlp.sentiment import score_text


def test_positive_headline():
    r = score_text("Nvidia surges after a strong earnings beat and raised guidance")
    assert r.label == "positive"
    assert r.score > 0
    assert r.hits >= 2


def test_negative_headline():
    r = score_text("Shares plunge after an earnings miss and a fraud investigation")
    assert r.label == "negative"
    assert r.score < 0


def test_negation_flips_polarity():
    plain = score_text("the outlook is strong")
    negated = score_text("the outlook is not strong")
    assert plain.score > 0
    assert negated.score < plain.score


def test_score_bounded():
    r = score_text("beat surge gain profit growth rally soar boost record strong " * 5)
    assert -1.0 <= r.score <= 1.0


def test_neutral_text():
    r = score_text("the company scheduled a meeting on tuesday")
    assert r.label == "neutral"
    assert r.hits == 0
