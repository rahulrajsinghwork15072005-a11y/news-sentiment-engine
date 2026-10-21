# News Sentiment + Price Reaction Engine

A financial **event-study** platform: it scores news sentiment, extracts tickers,
and measures the market's reaction using **abnormal-return** analysis — CAR/CAAR
with significance tests, sentiment↔return correlation, and reaction **decay**.

> Research/analytics tool only. Not investment advice; produces no trade signals.

See [PLAN.md](PLAN.md) for the full design rationale and math, and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system architecture (the
event-study engine, the NLP layer, and the Patell/BMP standardized tests).

## What's implemented

| Layer | Status |
|-------|--------|
| Pure core engine (`app/core`) | returns, market-model expected return, AR/CAR/CAAR, significance tests (incl. **Patell & BMP** standardized tests), decay fit, correlation |
| NLP (`app/nlp`) | lexicon sentiment scorer (VADER-style), ticker extraction (cashtags + company map) |
| Synthetic data (`app/data`) | reproducible market + sentiment-driven news events |
| API (`app/main.py`) | FastAPI REST + OpenAPI docs |
| Dashboard (`app/static`) | Plotly dashboard served by FastAPI (no build step) |
| Tests (`tests/`) | 65 unit + API tests (incl. Patell/BMP standardized tests) |

Persistence is SQLite. Heavier integrations (FinBERT, PostgreSQL/ORM, spaCy NER,
scheduled RSS ingestion) are **roadmap** items in [PLAN.md](PLAN.md) — designed to
slot in behind the existing interfaces, but **not yet implemented**.

## Run it

From the repo root (`projects/`) a shared virtual environment lives in `.venv`.

```powershell
# 1. install (once) — uses the shared venv at projects/.venv
& ..\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. run the tests
& ..\.venv\Scripts\python.exe -m pytest -q

# 2b. throughput benchmark (event study + Patell/BMP)
& ..\.venv\Scripts\python.exe scripts\bench.py

# 3. start the API + dashboard
& ..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8001
```

Run the commands from `news-sentiment-engine/backend/`. Then open:

- Dashboard: http://127.0.0.1:8001/
- Swagger docs: http://127.0.0.1:8001/docs

## Key endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/sentiment/score` | score free text |
| POST | `/api/tickers/extract` | extract tickers from text |
| POST | `/api/study/run` | event study from price arrays |
| GET | `/api/demo/overview` | synthetic prices + events |
| GET | `/api/demo/event-study` | single-event AR/CAR |
| GET | `/api/demo/caar` | CAAR by sentiment sign |
| GET | `/api/demo/correlation` | sentiment vs forward CAR |
| GET | `/api/demo/decay` | reaction decay + half-life |
| GET | `/api/demo/caar-ci` | per-event CARs + bootstrap CI for the mean CAR |
| GET | `/api/demo/standardized-tests` | Patell + BMP standardized event-study tests |
| GET | `/api/demo/report` | standalone HTML event-study report |

## Live data (optional)

Real prices via yfinance and real headlines via RSS, each with a deterministic
offline fallback (append `&allow_network=false` to force synthetic):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/live/prices?ticker=AAPL` | real daily closes (or synthetic) |
| GET | `/api/live/news?ticker=AAPL` | real headlines, each scored |
| GET | `/api/live/study?ticker=AAPL&benchmark=SPY` | event study on the largest recent move |

## React frontend (Vite)

```powershell
cd frontend
npm install
npm run dev        # http://localhost:5181 (dev proxy /api -> :8001)
npm run build      # production build to dist/
```

## Docker

```powershell
docker compose up --build
# frontend: http://localhost:8081  |  API: http://localhost:8001
```

## Production & operations

- Observability: `GET /api/version`, `GET /api/ready`, `GET /metrics` (Prometheus text format).
- Middleware: CORS, GZip, request-id + security headers, structured request logs, optional rate limiting.
- Env config: `CORS_ORIGINS`, `LOG_LEVEL`, `RATE_LIMIT_PER_SEC`, `STUDIES_DB`.

## Persistence & non-parametric tests

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/studies` | run + persist an event study (SQLite) |
| GET | `/api/studies` | list persisted studies |
| GET | `/api/studies/{id}` | fetch a stored study |
| POST | `/api/significance` | Wilcoxon signed-rank + sign test on CARs |
| POST | `/api/significance/standardized` | Patell (1976) + BMP (1991) standardized tests |

## Known limitations & roadmap

**Limitations**
- Sentiment is a lexicon scorer (not a transformer); ticker mapping covers a
  curated company list; studies use daily event windows.
- Persistence is SQLite; live prices/headlines are optional with synthetic fallback.

**Roadmap**
- FinBERT (lexicon vs transformer vs ensemble) and spaCy NER for entity
  disambiguation.
- PostgreSQL/ORM, scheduled RSS ingestion with dedup. (Bootstrap CAAR confidence
  intervals and a standalone HTML event-study report are implemented:
  `GET /api/demo/caar-ci`, `GET /api/demo/report`.)
