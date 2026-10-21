"""Environment-driven application settings (no extra dependencies)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _csv(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    app_name: str = "news-sentiment-engine"
    version: str = "1.2.0"
    environment: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    # CORS origins for the React dev server / deployed frontend.
    cors_origins: list[str] = field(
        default_factory=lambda: _csv(
            "CORS_ORIGINS",
            ["http://localhost:5181", "http://localhost:5173", "http://localhost:8081"],
        )
    )
    # Requests/second/client; 0 disables rate limiting.
    rate_limit_per_sec: int = int(os.getenv("RATE_LIMIT_PER_SEC", "0"))
    # SQLite path for persisted studies (":memory:" keeps it ephemeral).
    studies_db: str = os.getenv("STUDIES_DB", ":memory:")


settings = Settings()
