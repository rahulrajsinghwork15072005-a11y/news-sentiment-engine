"""Self-contained HTML event-study report (no external chart dependencies).

Renders a single-event study as a standalone HTML page: a headline stats table
plus an inline-SVG cumulative-abnormal-return (CAR) path. The output embeds no
scripts or remote assets, so it can be saved and opened anywhere.
"""
from __future__ import annotations

from html import escape

__all__ = ["event_study_html"]


def _svg_car_path(values: list[float], width: int = 540, height: int = 170, pad: int = 18) -> str:
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return '<svg viewBox="0 0 1 1" width="100%"></svg>'
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0

    def px(i: int) -> float:
        return pad + (width - 2 * pad) * (i / (n - 1) if n > 1 else 0.5)

    def py(v: float) -> float:
        return pad + (height - 2 * pad) * (1.0 - (v - lo) / rng)

    points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
    baseline = ""
    if lo <= 0.0 <= hi:
        y0 = py(0.0)
        baseline = (
            f'<line x1="{pad}" y1="{y0:.1f}" x2="{width - pad}" y2="{y0:.1f}" '
            f'stroke="#9ca3af" stroke-dasharray="4 3"/>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'style="background:#0d1117;border-radius:8px">'
        f"{baseline}"
        f'<polyline fill="none" stroke="#2ea043" stroke-width="2.2" points="{points}"/>'
        f"</svg>"
    )


def event_study_html(result: dict, title: str = "Event Study Report") -> str:
    """Render an event-study result dict to a standalone HTML string."""
    car = float(result.get("car", 0.0))
    t_stat = float(result.get("t_stat", 0.0))
    p_value = float(result.get("p_value", 1.0))
    alpha = float(result.get("alpha", 0.0))
    beta = float(result.get("beta", 0.0))
    car_path = result.get("car_path", [])
    event = result.get("event", {}) or {}

    sig = "yes" if p_value < 0.05 else "no"
    rows = [
        ("Ticker", escape(str(event.get("ticker", "-")))),
        ("Headline", escape(str(event.get("headline", "-")))),
        ("Sentiment", f"{float(event.get('sentiment', 0.0)):+.3f}"),
        ("CAR", f"{car * 100:+.2f}%"),
        ("t-stat", f"{t_stat:.2f}"),
        ("p-value", f"{p_value:.4f}"),
        ("Significant (5%)", sig),
        ("Market-model alpha", f"{alpha:.5f}"),
        ("Market-model beta", f"{beta:.3f}"),
    ]
    table = "\n".join(
        f'<tr><td class="k">{k}</td><td class="v">{v}</td></tr>' for k, v in rows
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{escape(title)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#0d1117; color:#e6edf3;
          margin:0; padding:24px; }}
  .card {{ max-width:640px; margin:0 auto; background:#161b22; border:1px solid #30363d;
           border-radius:12px; padding:20px 24px; }}
  h1 {{ font-size:1.25rem; margin:0 0 4px; }}
  p.sub {{ color:#8b949e; margin:0 0 16px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:14px; }}
  td {{ padding:6px 8px; border-bottom:1px solid #21262d; font-size:0.95rem; }}
  td.k {{ color:#8b949e; width:42%; }}
  td.v {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .note {{ color:#8b949e; font-size:0.8rem; margin-top:16px; }}
</style></head>
<body><div class="card">
  <h1>{escape(title)}</h1>
  <p class="sub">Market-model abnormal-return event study (research only).</p>
  {_svg_car_path(car_path)}
  <table>{table}</table>
  <p class="note">CAR is the cumulative abnormal return over the event window;
  significance is a two-sided t-test. Not investment advice.</p>
</div></body></html>"""
