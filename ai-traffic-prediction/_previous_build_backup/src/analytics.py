"""Aggregations for the Dashboard and Analytics pages.

Everything here describes HISTORICAL records in the dataset (or, for the
synthetic sample, simulated records). Nothing is live traffic.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src import preprocess as pp

DAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_cache: dict = {}


class FilterError(ValueError):
    def __init__(self, errors: dict):
        super().__init__("; ".join(errors.values()))
        self.errors = errors


def get_clean(data_path: Path | str) -> pd.DataFrame:
    """Cleaned dataset, cached until the file changes on disk."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise pp.DataError("Dataset not found. Run `python src/generate_sample_data.py` or "
                           "upload a CSV on the Model Performance page.")
    mtime = data_path.stat().st_mtime
    hit = _cache.get(str(data_path))
    if hit and hit[0] == mtime:
        return hit[1]
    df = pp.clean(pp.load_raw(data_path))
    df["hour"] = df["date_time"].dt.hour
    df["dow"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    _cache[str(data_path)] = (mtime, df)
    return df


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def parse_filters(args, df: pd.DataFrame) -> dict:
    errors: dict = {}

    def _date(name):
        raw = (args.get(name) or "").strip()
        if not raw:
            return None
        try:
            return datetime.strptime(raw, "%Y-%m-%d")
        except ValueError:
            errors[name] = f"{name.capitalize()} date must look like YYYY-MM-DD."
            return None

    start, end = _date("start"), _date("end")
    if start and end and start > end:
        errors["start"] = "Start date must not be after the end date."

    def _hour(name, default):
        raw = str(args.get(name, "")).strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if not 0 <= val <= 23:
                raise ValueError
            return val
        except ValueError:
            errors[name] = "Hours must be whole numbers from 0 to 23."
            return default

    hour_from, hour_to = _hour("hour_from", 0), _hour("hour_to", 23)
    if hour_from > hour_to and "hour_from" not in errors and "hour_to" not in errors:
        errors["hour_from"] = "'From hour' must not be later than 'to hour'."

    day_type = (args.get("day_type") or "all").strip().lower()
    if day_type not in {"all", "weekday", "weekend"}:
        errors["day_type"] = "Day type must be all, weekday or weekend."

    if errors:
        raise FilterError(errors)
    return {"start": start, "end": end, "hour_from": hour_from, "hour_to": hour_to,
            "day_type": day_type}


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    mask = (df["hour"] >= f["hour_from"]) & (df["hour"] <= f["hour_to"])
    if f["start"] is not None:
        mask &= df["date_time"] >= pd.Timestamp(f["start"])
    if f["end"] is not None:
        mask &= df["date_time"] < pd.Timestamp(f["end"]) + pd.Timedelta(days=1)
    if f["day_type"] == "weekday":
        mask &= df["dow"] < 5
    elif f["day_type"] == "weekend":
        mask &= df["dow"] >= 5
    return df[mask]


def describe_filters(f: dict) -> dict:
    return {"start": f["start"].strftime("%Y-%m-%d") if f["start"] else None,
            "end": f["end"].strftime("%Y-%m-%d") if f["end"] else None,
            "hour_from": f["hour_from"], "hour_to": f["hour_to"], "day_type": f["day_type"]}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _list(series: pd.Series, index, digits=0) -> list:
    s = series.reindex(index)
    return [None if pd.isna(v) else (int(round(v)) if digits == 0 else round(float(v), digits))
            for v in s]


def category_counts(volumes: pd.Series, thresholds, labels) -> dict:
    idx = np.digitize(volumes.to_numpy(), thresholds)
    counts = np.bincount(idx, minlength=len(labels))
    return {label: int(c) for label, c in zip(labels, counts)}


def fallback_thresholds(volumes: pd.Series):
    return [float(np.quantile(volumes, q)) for q in (0.30, 0.65, 0.90)]


def _trend(df: pd.DataFrame):
    """Daily average, switching to weekly/monthly for long ranges."""
    if df.empty:
        return "daily", [], []
    span_days = (df["date_time"].max() - df["date_time"].min()).days + 1
    rule, name, fmt = (("D", "daily", "%Y-%m-%d") if span_days <= 120 else
                       ("W", "weekly", "%Y-%m-%d") if span_days <= 800 else
                       ("MS", "monthly", "%Y-%m"))
    s = df.set_index("date_time")["traffic_volume"].resample(rule).mean().dropna()
    return name, [t.strftime(fmt) for t in s.index], [int(round(v)) for v in s]


# --------------------------------------------------------------------------- #
# Payloads
# --------------------------------------------------------------------------- #
def analytics_payload(df: pd.DataFrame, thresholds=None, labels=("Low", "Moderate", "High", "Severe")) -> dict:
    if df.empty:
        return {"empty": True}
    vol = df["traffic_volume"]
    hourly = df.groupby("hour")["traffic_volume"].mean()
    wk = df[df["dow"] < 5].groupby("hour")["traffic_volume"].mean()
    we = df[df["dow"] >= 5].groupby("hour")["traffic_volume"].mean()
    top = hourly.sort_values(ascending=False).head(3)
    granularity, t_labels, t_values = _trend(df)
    dow = df.groupby("dow")["traffic_volume"].mean()
    month = df.groupby("month")["traffic_volume"].mean()
    wx = (df.groupby("weather_main")["traffic_volume"].agg(["mean", "count"])
            .query("count >= 20").sort_values("mean", ascending=False))
    cuts = thresholds or fallback_thresholds(vol)
    wk_avg = df.loc[df["dow"] < 5, "traffic_volume"].mean()
    we_avg = df.loc[df["dow"] >= 5, "traffic_volume"].mean()
    return {
        "empty": False,
        "kpis": {
            "records": int(len(df)),
            "average": int(round(vol.mean())),
            "maximum": int(vol.max()),
            "minimum": int(vol.min()),
            "peak_hour": int(hourly.idxmax()),
            "quietest_hour": int(hourly.idxmin()),
            "weekday_avg": None if pd.isna(wk_avg) else int(round(wk_avg)),
            "weekend_avg": None if pd.isna(we_avg) else int(round(we_avg)),
            "start": df["date_time"].min().strftime("%Y-%m-%d"),
            "end": df["date_time"].max().strftime("%Y-%m-%d"),
        },
        "peak_hours": [{"hour": int(h), "average": int(round(v))} for h, v in top.items()],
        "hourly": {"labels": [f"{h:02d}:00" for h in range(24)], "values": _list(hourly, range(24)),
                   "weekday": _list(wk, range(24)), "weekend": _list(we, range(24))},
        "trend": {"granularity": granularity, "labels": t_labels, "values": t_values},
        "day_of_week": {"labels": DAY_SHORT, "values": _list(dow, range(7))},
        "monthly": {"labels": MONTH_SHORT, "values": _list(month, range(1, 13))},
        "weekday_vs_weekend": {"labels": ["Weekday", "Weekend"],
                               "values": [int(round(wk_avg)) if not pd.isna(wk_avg) else None,
                                          int(round(we_avg)) if not pd.isna(we_avg) else None]},
        "weather": {"labels": wx.index.tolist(), "values": [int(round(v)) for v in wx["mean"]],
                    "counts": [int(c) for c in wx["count"]]},
        "congestion": category_counts(vol, cuts, labels),
        "thresholds": [float(c) for c in cuts],
    }


def dashboard_payload(df: pd.DataFrame, bundle: dict | None, metrics: dict | None) -> dict:
    thresholds = bundle["thresholds"] if bundle else None
    labels = bundle["labels"] if bundle else ["Low", "Moderate", "High", "Severe"]
    base = analytics_payload(df, thresholds, labels)

    # last 90 days of the dataset as a daily line
    last = df["date_time"].max()
    recent = df[df["date_time"] > last - timedelta(days=90)]
    daily = recent.set_index("date_time")["traffic_volume"].resample("D").mean().dropna()
    base["recent_daily"] = {"labels": [t.strftime("%Y-%m-%d") for t in daily.index],
                            "values": [int(round(v)) for v in daily]}

    latest = df.iloc[-1]
    cuts = thresholds or fallback_thresholds(df["traffic_volume"])
    base["latest_observation"] = {
        "timestamp": latest["date_time"].strftime("%Y-%m-%d %H:%M"),
        "volume": int(latest["traffic_volume"]),
        "category": labels[int(np.digitize([latest["traffic_volume"]], cuts)[0])],
        "weather": str(latest["weather_main"]),
    }
    if metrics:
        ts = metrics["test_series"]
        n = 7 * 24
        base["comparison"] = {"labels": ts["timestamps"][-n:], "actual": ts["actual"][-n:],
                              "predicted": ts["predicted"][-n:],
                              "range": [ts["timestamps"][-n:][0], ts["timestamps"][-1]]}
    return base
