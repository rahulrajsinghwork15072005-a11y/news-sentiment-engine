"""SQLite-backed store for persisted event studies (stdlib only)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid


class StudyStore:
    """Thread-safe store keyed by a generated study id.

    A single shared connection (``check_same_thread=False`` + a lock) keeps an
    in-memory database alive for the process; pass a file path to persist.
    """

    _COLUMNS = ["id", "ticker", "created_at", "car", "t_stat", "p_value"]

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS studies (
                    id TEXT PRIMARY KEY,
                    ticker TEXT,
                    created_at REAL,
                    car REAL,
                    t_stat REAL,
                    p_value REAL,
                    payload TEXT
                )
                """
            )
            self._conn.commit()

    def save(self, ticker: str, result: dict) -> str:
        study_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO studies VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    study_id,
                    ticker,
                    time.time(),
                    result.get("car"),
                    result.get("t_stat"),
                    result.get("p_value"),
                    json.dumps(result),
                ),
            )
            self._conn.commit()
        return study_id

    def list(self, ticker: str | None = None, limit: int = 50) -> list[dict]:
        query = f"SELECT {', '.join(self._COLUMNS)} FROM studies"
        args: list = []
        if ticker:
            query += " WHERE ticker = ?"
            args.append(ticker)
        query += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(query, args).fetchall()
        return [dict(zip(self._COLUMNS, row)) for row in rows]

    def get(self, study_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM studies WHERE id = ?", (study_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM studies").fetchone()[0])
