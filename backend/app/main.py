"""FastAPI application: REST endpoints + static dashboard."""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import demo as demo_service
from . import live as live_service
from .config import settings
from .core.event_study import run_event_study
from .core.returns import simple_returns
from .core.significance import sign_test, wilcoxon_signed_rank
from .core.standardized_tests import bmp_test, patell_test
from .nlp.sentiment import score_text
from .nlp.tickers import extract_tickers
from .observability import (
    ObservabilityMiddleware,
    RateLimiter,
    RateLimitMiddleware,
    metrics,
    uptime_seconds,
)
from .persistence.store import StudyStore
from .schemas import (
    CarValuesRequest,
    EventStudyRequest,
    EventStudyResponse,
    SentimentResponse,
    StandardizedTestRequest,
    TextRequest,
    TickerResponse,
)

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="News Sentiment + Price Reaction Engine",
    version=settings.version,
    description="Event-study analytics linking news sentiment to abnormal returns.",
)

app.add_middleware(GZipMiddleware, minimum_size=512)
if settings.rate_limit_per_sec > 0:
    app.add_middleware(RateLimitMiddleware, limiter=RateLimiter(settings.rate_limit_per_sec))
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})


if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

study_store = StudyStore(settings.studies_db)


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="dashboard not found")
    return FileResponse(str(index))


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/version")
def version() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
    }


@app.get("/api/ready")
def ready() -> dict:
    return {"ready": True, "uptime_seconds": round(uptime_seconds(), 1)}


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@app.post("/api/sentiment/score", response_model=SentimentResponse)
def sentiment_score(req: TextRequest) -> SentimentResponse:
    result = score_text(req.text)
    return SentimentResponse(score=result.score, label=result.label, hits=result.hits)


@app.post("/api/tickers/extract", response_model=TickerResponse)
def tickers_extract(req: TextRequest) -> TickerResponse:
    return TickerResponse(tickers=extract_tickers(req.text))


@app.post("/api/study/run", response_model=EventStudyResponse)
def study_run(req: EventStudyRequest) -> EventStudyResponse:
    try:
        asset_ret = simple_returns(req.asset_prices)
        market_ret = simple_returns(req.market_prices)
        result = run_event_study(
            asset_ret,
            market_ret,
            event_idx=req.event_index,
            est_len=req.est_len,
            est_gap=req.est_gap,
            pre=req.pre,
            post=req.post,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return EventStudyResponse(**vars(result))


@app.post("/api/studies")
def create_study(req: EventStudyRequest) -> dict:
    try:
        asset_ret = simple_returns(req.asset_prices)
        market_ret = simple_returns(req.market_prices)
        result = run_event_study(
            asset_ret,
            market_ret,
            event_idx=req.event_index,
            est_len=req.est_len,
            est_gap=req.est_gap,
            pre=req.pre,
            post=req.post,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    payload = asdict(result)
    study_id = study_store.save(req.ticker or "UNKNOWN", payload)
    return {"id": study_id, **payload}


@app.get("/api/studies")
def list_studies(ticker: str | None = None, limit: int = 50) -> dict:
    return {"studies": study_store.list(ticker, limit)}


@app.get("/api/studies/{study_id}")
def get_study(study_id: str) -> dict:
    study = study_store.get(study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="study not found")
    return study


@app.post("/api/significance")
def significance(req: CarValuesRequest) -> dict:
    w_stat, w_p = wilcoxon_signed_rank(req.car_values)
    return {
        "wilcoxon": {"statistic": w_stat, "p_value": w_p},
        "sign_test": sign_test(req.car_values),
    }


@app.post("/api/significance/standardized")
def significance_standardized(req: StandardizedTestRequest) -> dict:
    """Patell (1976) and BMP (1991) standardized event-study tests."""
    try:
        patell = patell_test(req.cars, req.resid_stds, req.window_lens, req.est_lens)
        bmp = bmp_test(req.cars, req.resid_stds, req.window_lens)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"patell": asdict(patell), "bmp": asdict(bmp)}


@app.get("/api/demo/overview")
def demo_overview() -> dict:
    return demo_service.demo_overview()


@app.get("/api/demo/event-study")
def demo_event_study(event_number: int = 0, pre: int = 1, post: int = 5) -> dict:
    try:
        return demo_service.demo_event_study(event_number=event_number, pre=pre, post=post)
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/demo/caar")
def demo_caar(pre: int = 1, post: int = 10) -> dict:
    return demo_service.demo_caar(pre=pre, post=post)


@app.get("/api/demo/correlation")
def demo_correlation(post: int = 5) -> dict:
    return demo_service.demo_correlation(post=post)


@app.get("/api/demo/decay")
def demo_decay(post: int = 8) -> dict:
    return demo_service.demo_decay(post=post)


@app.get("/api/demo/caar-ci")
def demo_caar_ci(post: int = 5, n_boot: int = 2000) -> dict:
    """Per-event CARs and a percentile bootstrap CI for the mean CAR."""
    n_boot = max(100, min(n_boot, 20000))
    return demo_service.demo_caar_ci(post=post, n_boot=n_boot)


@app.get("/api/demo/standardized-tests")
def demo_standardized_tests(pre: int = 1, post: int = 5) -> dict:
    """Patell + BMP standardized tests aggregated over the demo events."""
    return demo_service.demo_standardized_tests(pre=pre, post=post)


@app.get("/api/demo/report", response_class=HTMLResponse)
def demo_report(event_number: int = 0, pre: int = 1, post: int = 5) -> HTMLResponse:
    """Standalone HTML event-study report for one synthetic event."""
    try:
        return HTMLResponse(demo_service.demo_report(event_number, pre=pre, post=post))
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/live/prices")
def live_prices(ticker: str = "AAPL", allow_network: bool = True) -> dict:
    return live_service.live_prices(ticker, allow_network=allow_network)


@app.get("/api/live/news")
def live_news(ticker: str = "AAPL", allow_network: bool = True) -> dict:
    return live_service.live_news(ticker, allow_network=allow_network)


@app.get("/api/live/study")
def live_study(ticker: str = "AAPL", benchmark: str = "SPY", allow_network: bool = True) -> dict:
    try:
        return live_service.live_study(ticker, benchmark, allow_network=allow_network)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
