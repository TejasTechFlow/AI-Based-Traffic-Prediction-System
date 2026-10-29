"""SQLite storage for prediction history (standard library only)."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    pred_date TEXT NOT NULL,
    hour INTEGER NOT NULL,
    day_name TEXT NOT NULL,
    holiday INTEGER NOT NULL,
    weather TEXT NOT NULL,
    temperature REAL NOT NULL,
    predicted_volume INTEGER NOT NULL,
    category TEXT NOT NULL,
    model TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
"""

EXPORT_COLUMNS = ["id", "created_at", "pred_date", "hour", "day_name", "holiday",
                  "weather", "temperature", "predicted_volume", "category", "model"]


def _connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(db_path)) as conn, conn:
        conn.executescript(SCHEMA)


def save_prediction(db_path: Path | str, result: dict) -> int:
    i = result["inputs"]
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute(
            "INSERT INTO predictions (created_at, pred_date, hour, day_name, holiday, weather, "
            "temperature, predicted_volume, category, model) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), i["date"], i["hour"], i["day_name"],
             int(i["holiday"]), i["weather"], i["temperature"], result["predicted_volume"],
             result["category"], result["model"]))
        return cur.lastrowid


def _where(q: str, category: str, start: str, end: str):
    clauses, params = [], []
    if q:
        like = f"%{q.strip()}%"
        clauses.append("(pred_date LIKE ? OR day_name LIKE ? OR weather LIKE ? OR category LIKE ? "
                       "OR model LIKE ? OR created_at LIKE ? OR CAST(predicted_volume AS TEXT) LIKE ?)")
        params += [like] * 7
    if category:
        clauses.append("category = ?")
        params.append(category)
    if start:
        clauses.append("date(created_at) >= date(?)")
        params.append(start)
    if end:
        clauses.append("date(created_at) <= date(?)")
        params.append(end)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def query_history(db_path, q="", category="", start="", end="", page=1, per_page=15):
    where, params = _where(q, category, start, end)
    with closing(_connect(db_path)) as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM predictions{where}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM predictions{where} ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [per_page, (max(page, 1) - 1) * per_page]).fetchall()
    return [dict(r) for r in rows], total


def all_rows(db_path, q="", category="", start="", end=""):
    where, params = _where(q, category, start, end)
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(f"SELECT {', '.join(EXPORT_COLUMNS)} FROM predictions{where} "
                            "ORDER BY id DESC", params).fetchall()
    return [dict(r) for r in rows]


def clear_history(db_path) -> int:
    with closing(_connect(db_path)) as conn, conn:
        return conn.execute("DELETE FROM predictions").rowcount
