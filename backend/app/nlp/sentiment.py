"""Lexicon-based financial sentiment scorer (VADER-style normalization).

This is a transparent, deterministic baseline that needs no model download.
A FinBERT backend can be slotted in behind the same ``score_text`` signature.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Finance-flavoured polarity lexicon (Loughran-McDonald inspired, compact).
POSITIVE: dict[str, float] = {
    "beat": 2.0, "beats": 2.0, "surge": 2.5, "surged": 2.5, "surges": 2.5,
    "gain": 1.5, "gains": 1.5, "gained": 1.5, "profit": 1.8, "profits": 1.8,
    "growth": 1.6, "grow": 1.4, "upgrade": 2.2, "upgraded": 2.2, "bullish": 2.4,
    "record": 1.8, "strong": 1.6, "stronger": 1.8, "rally": 2.2, "rallied": 2.2,
    "outperform": 2.3, "outperformed": 2.3, "jump": 2.0, "jumped": 2.0,
    "soar": 2.6, "soared": 2.6, "rise": 1.3, "rose": 1.3, "boost": 1.7,
    "boosted": 1.7, "exceed": 2.0, "exceeded": 2.0, "optimistic": 1.9,
    "positive": 1.4, "win": 1.6, "wins": 1.6, "approval": 1.5, "approved": 1.5,
    "expansion": 1.4, "robust": 1.7, "rebound": 1.8, "rebounded": 1.8,
}
NEGATIVE: dict[str, float] = {
    "miss": 2.0, "misses": 2.0, "missed": 2.0, "loss": 1.9, "losses": 1.9,
    "decline": 1.7, "declined": 1.7, "drop": 1.8, "dropped": 1.8, "drops": 1.8,
    "downgrade": 2.2, "downgraded": 2.2, "bearish": 2.4, "weak": 1.6,
    "weaker": 1.8, "plunge": 2.6, "plunged": 2.6, "fall": 1.4, "fell": 1.4,
    "cut": 1.5, "cuts": 1.5, "lawsuit": 2.0, "bankruptcy": 2.8, "fraud": 2.7,
    "slump": 2.1, "slumped": 2.1, "underperform": 2.3, "warn": 1.9,
    "warned": 1.9, "warning": 1.9, "recession": 2.2, "negative": 1.4,
    "investigation": 1.8, "probe": 1.7, "halt": 1.6, "halted": 1.6,
    "default": 2.3, "selloff": 2.2, "tumble": 2.3, "tumbled": 2.3,
}
NEGATIONS = {"not", "no", "never", "without", "hardly", "barely", "n't", "cannot"}
INTENSIFIERS = {
    "very": 1.4, "highly": 1.4, "sharply": 1.6, "significantly": 1.5,
    "substantially": 1.5, "slightly": 0.6, "modestly": 0.7, "marginally": 0.6,
}

_NEGATION_SCALE = -0.74
_TOKEN_RE = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class SentimentResult:
    score: float       # normalized polarity in [-1, 1]
    label: str         # positive | neutral | negative
    hits: int          # number of polarity-bearing tokens


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def score_text(text: str, neutral_band: float = 0.05) -> SentimentResult:
    """Score free text into a bounded sentiment in ``[-1, 1]``."""
    tokens = _tokenize(text)
    total = 0.0
    hits = 0
    for i, tok in enumerate(tokens):
        base = POSITIVE.get(tok, 0.0) or -NEGATIVE.get(tok, 0.0)
        if base == 0.0:
            continue
        mult = 1.0
        # look back up to two tokens for negation / intensifier context
        for j in (i - 1, i - 2):
            if j < 0:
                continue
            prev = tokens[j]
            if prev in NEGATIONS:
                mult *= _NEGATION_SCALE
            elif prev in INTENSIFIERS:
                mult *= INTENSIFIERS[prev]
        total += base * mult
        hits += 1

    # VADER-style normalization keeps the score bounded in (-1, 1).
    score = total / math.sqrt(total * total + 15.0) if hits else 0.0
    if score > neutral_band:
        label = "positive"
    elif score < -neutral_band:
        label = "negative"
    else:
        label = "neutral"
    return SentimentResult(score=round(float(score), 4), label=label, hits=hits)
