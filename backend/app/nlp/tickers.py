"""Ticker extraction via cashtags, company-name map, and known-symbol matching."""
from __future__ import annotations

import re

# Compact company -> ticker resolution map (extend as needed).
COMPANY_TO_TICKER: dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "intel": "INTC",
    "advanced micro devices": "AMD",
    "amd": "AMD",
    "jpmorgan": "JPM",
    "goldman sachs": "GS",
    "exxon": "XOM",
    "boeing": "BA",
    "ford": "F",
    "general motors": "GM",
    "walmart": "WMT",
    "coca-cola": "KO",
}

KNOWN_TICKERS: set[str] = set(COMPANY_TO_TICKER.values()) | {
    "AAPL", "MSFT", "AMZN", "TSLA", "NVDA", "GOOGL", "META", "NFLX",
    "INTC", "AMD", "JPM", "GS", "XOM", "BA", "F", "GM", "WMT", "KO",
}

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})\b")
_UPPER_RE = re.compile(r"\b([A-Z]{2,5})\b")


def extract_tickers(text: str) -> list[str]:
    """Return a sorted list of unique tickers mentioned in ``text``."""
    found: set[str] = set()

    # 1) explicit cashtags like $AAPL
    for match in _CASHTAG_RE.finditer(text):
        symbol = match.group(1).upper()
        if symbol in KNOWN_TICKERS:
            found.add(symbol)

    # 2) company names
    low = text.lower()
    for name, ticker in COMPANY_TO_TICKER.items():
        if re.search(rf"\b{re.escape(name)}\b", low):
            found.add(ticker)

    # 3) bare uppercase symbols that match a known ticker
    for match in _UPPER_RE.finditer(text):
        if match.group(1) in KNOWN_TICKERS:
            found.add(match.group(1))

    return sorted(found)
