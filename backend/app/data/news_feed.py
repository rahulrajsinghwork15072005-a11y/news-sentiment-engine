"""Real news headlines via RSS (stdlib only) with a synthetic fallback."""
from __future__ import annotations

import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

_TICKER_RE = re.compile(r"^[A-Za-z.\-]{1,10}$")

_SYNTH_TEMPLATES = [
    "{t} shares rise as quarterly profit beats estimates",
    "{t} slips after analysts flag weak guidance",
    "{t} jumps on upgraded price target and strong demand",
    "{t} falls amid broader market selloff and rate fears",
    "{t} gains after announcing a record buyback",
    "{t} drops following a downgrade on margin concerns",
]


@dataclass(frozen=True)
class Headline:
    title: str
    published: str | None
    source: str  # "yahoo-rss" | "synthetic"


def synthetic_headlines(ticker: str, limit: int = 10) -> list[Headline]:
    out = [Headline(tpl.format(t=ticker.upper()), None, "synthetic") for tpl in _SYNTH_TEMPLATES]
    return out[:limit]


def fetch_headlines(ticker: str, limit: int = 10, allow_network: bool = True) -> list[Headline]:
    """Fetch recent headlines for a ticker, falling back to synthetic ones."""
    if allow_network and _TICKER_RE.match(ticker):
        url = (
            "https://feeds.finance.yahoo.com/rss/2.0/headline"
            f"?s={ticker}&region=US&lang=en-US"
        )
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 (fixed host)
                root = ET.fromstring(response.read())
            items: list[Headline] = []
            for item in root.iter("item"):
                title = item.findtext("title")
                if title:
                    items.append(Headline(title, item.findtext("pubDate"), "yahoo-rss"))
                if len(items) >= limit:
                    break
            if items:
                return items
        except Exception:  # pragma: no cover - network/parse failures fall through
            pass
    return synthetic_headlines(ticker, limit)
