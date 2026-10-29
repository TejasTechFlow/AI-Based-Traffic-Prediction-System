"""SQLite storage for prediction history (stdlib sqlite3, parameterised queries only)."""
import csv
import io
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

COLUMNS = ["id", "created_at", "date", "hour", "day_of_week", "holiday", "weather", "temperature_c",
           "predicted_volume", "category", "model"]


@contextmanager
def connect(db_path=None):
    conn = sqlite3.connect(db_path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=None):
    with connect(db_path) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, date TEXT NOT NULL,
            hour INTEGER NOT NULL, day_of_week TEXT, holiday INTEGER, weather TEXT, temperature_c REAL,
            predicted_volume INTEGER NOT NULL, category TEXT NOT NULL, model TEXT)""")


def add_prediction(result: dict, db_path=None) -> int:
    i = result["inputs"]
    with connect(db_path) as c:
        cur = c.execute(
            "INSERT INTO predictions (created_at,date,hour,day_of_week,holiday,weather,temperature_c,"
            "predicted_volume,category,model) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i["date"], i["hour"], i["day_of_week"],
             int(i["is_holiday"]), i["weather_main"], i["temperature_c"], result["predicted_volume"],
             result["category"], result["model"]))
        return cur.lastrowid


def query(q="", category="", start="", end="", limit=500, db_path=None) -> list:
    sql, args = "SELECT * FROM predictions WHERE 1=1", []
    if q:
        like = f"%{q}%"
        sql += (" AND (date LIKE ? OR weather LIKE ? OR day_of_week LIKE ? OR category LIKE ? "
                "OR CAST(predicted_volume AS TEXT) LIKE ? OR created_at LIKE ?)")
        args += [like] * 6
    if category:
        sql += " AND category = ?"; args.append(category)
    if start:
        sql += " AND date >= ?"; args.append(start)
    if end:
        sql += " AND date <= ?"; args.append(end)
    sql += " ORDER BY id DESC LIMIT ?"; args.append(int(limit))
    with connect(db_path) as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def clear(db_path=None) -> int:
    with connect(db_path) as c:
        return c.execute("DELETE FROM predictions").rowcount


def to_csv(rows: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Predicted at", "Travel date", "Hour", "Day", "Holiday", "Weather", "Temperature (C)",
                "Predicted volume (veh/h)", "Congestion category", "Model"])
    for r in rows:
        w.writerow([r["created_at"], r["date"], r["hour"], r["day_of_week"], "Yes" if r["holiday"] else "No",
                    r["weather"], r["temperature_c"], r["predicted_volume"], r["category"], r["model"]])
    return buf.getvalue()
