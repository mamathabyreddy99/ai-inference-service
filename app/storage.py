"""SQLite persistence for prediction history.

Two write paths on purpose, so the benchmark can show a real before/after:
  * naive_log: opens a new connection and commits for every request.
  * Store:     one shared connection, WAL mode, batched inserts.
"""
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    endpoint TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
INSERT = "INSERT INTO predictions (text, label, confidence, endpoint) VALUES (?, ?, ?, ?)"


def init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(SCHEMA)
        conn.commit()


def naive_log(path: str, text: str, pred: dict) -> None:
    with closing(sqlite3.connect(path, timeout=30)) as conn:
        conn.execute(INSERT, (text, pred["label"], pred["confidence"], "predict/naive"))
        conn.commit()


class Store:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()

    def add_many(self, rows: list[tuple[str, str, float]], endpoint: str) -> None:
        data = [(t, label, conf, endpoint) for t, label, conf in rows]
        with self._lock:
            self._conn.executemany(INSERT, data)
            self._conn.commit()

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, text, label, confidence, endpoint, created_at "
                "FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()
