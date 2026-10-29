"""Historical analytics over the cleaned dataset (all numbers are computed, filters apply to every chart)."""
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.preprocess import DAY_NAMES, DatasetError, clean_data, load_raw  # noqa: E402

_cache = {"mtime": None, "df": None}


def load_dataset() -> pd.DataFrame:
    """Cleaned dataset, cached until the CSV changes on disk."""
    if not os.path.exists(config.DATA_PATH):
        raise DatasetError("Dataset not found. Put a CSV at data/traffic.csv or run `python src/train_model.py`.")
    mtime = os.path.getmtime(config.DATA_PATH)
    if _cache["mtime"] != mtime:
        df = clean_data(load_raw(config.DATA_PATH))
        df["hour"] = df.date_time.dt.hour
        df["dow"] = df.date_time.dt.dayofweek
        df["date"] = df.date_time.dt.normalize()
        _cache.update(mtime=mtime, df=df)
    return _cache["df"]


def filter_options(df=None) -> dict:
    df = load_dataset() if df is None else df
    return {"date_min": df.date_time.min().strftime("%Y-%m-%d"), "date_max": df.date_time.max().strftime("%Y-%m-%d"),
            "weather": sorted(w for w in df.weather_main.unique() if w != "Unknown"),
            "has_weather": bool((df.weather_main != "Unknown").any())}


def apply_filters(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    if p.get("start"):
        df = df[df.date_time >= pd.Timestamp(p["start"])]
    if p.get("end"):
        df = df[df.date_time < pd.Timestamp(p["end"]) + pd.Timedelta(days=1)]
    hf, ht = p.get("hour_from"), p.get("hour_to")
    if hf not in (None, ""):
        df = df[df.hour >= int(hf)]
    if ht not in (None, ""):
        df = df[df.hour <= int(ht)]
    dt = p.get("day_type")
    if dt == "weekday":
        df = df[df.dow < 5]
    elif dt == "weekend":
        df = df[df.dow >= 5]
    elif dt == "holiday":
        df = df[df.is_holiday == 1]
    if p.get("weather"):
        df = df[df.weather_main == p["weather"]]
    return df


def _r(s, nd=0):
    return [None if pd.isna(v) else round(float(v), nd) for v in s]


def _congestion_counts(volumes: pd.Series, thresholds) -> dict:
    idx = np.searchsorted(np.asarray(thresholds), volumes.to_numpy(), side="right")
    counts = np.bincount(idx, minlength=4)
    return {"labels": config.CATEGORIES, "counts": counts.tolist(),
            "colors": [config.CATEGORY_COLORS[c] for c in config.CATEGORIES]}


def _thresholds(meta, df):
    if meta and "thresholds" in meta:
        return meta["thresholds"]["values"]
    return [float(np.percentile(df.traffic_volume, q)) for q in config.THRESHOLD_PERCENTILES]


def analytics(p: dict, meta: dict = None) -> dict:
    full = load_dataset()
    df = apply_filters(full, p)
    if df.empty:
        return {"empty": True, "message": "No records match the selected filters.", "rows": 0}
    v = "traffic_volume"
    hourly = df.groupby("hour")[v].mean().reindex(range(24))
    hour_p = df.groupby("hour")[v]
    dow = df.groupby("dow")[v].mean().reindex(range(7))
    wk = df[df.dow < 5].groupby("hour")[v].mean().reindex(range(24))
    we = df[df.dow >= 5].groupby("hour")[v].mean().reindex(range(24))
    monthly = df.set_index("date_time")[v].resample("MS").mean().dropna()
    span_days = (df.date.max() - df.date.min()).days + 1
    daily_rule, daily_name = ("D", "Daily average") if span_days <= 180 else ("W", "Weekly average")
    daily = df.set_index("date_time")[v].resample(daily_rule).mean().dropna()
    moy = df.groupby(df.date_time.dt.month)[v].mean().reindex(range(1, 13))
    top = hourly.dropna().sort_values(ascending=False).head(3)

    out = {
        "empty": False, "rows": int(len(df)),
        "kpis": {"avg": round(float(df[v].mean())), "max": int(df[v].max()), "min": int(df[v].min()),
                 "peak_hours": [f"{int(h):02d}:00" for h in top.index],
                 "peak_avg": round(float(top.iloc[0])),
                 "period": f"{df.date_time.min():%d %b %Y} - {df.date_time.max():%d %b %Y}"},
        "hourly": {"labels": [f"{h:02d}:00" for h in range(24)], "avg": _r(hourly),
                   "p25": _r(hour_p.quantile(.25).reindex(range(24))), "p75": _r(hour_p.quantile(.75).reindex(range(24)))},
        "dow": {"labels": [d[:3] for d in DAY_NAMES], "avg": _r(dow)},
        "weekday_weekend": {"labels": [f"{h:02d}:00" for h in range(24)], "weekday": _r(wk), "weekend": _r(we),
                            "weekday_avg": None if df[df.dow < 5].empty else round(float(df[df.dow < 5][v].mean())),
                            "weekend_avg": None if df[df.dow >= 5].empty else round(float(df[df.dow >= 5][v].mean()))},
        "monthly": {"labels": monthly.index.strftime("%b %Y").tolist(), "avg": _r(monthly)},
        "month_of_year": {"labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                          "avg": _r(moy)},
        "daily": {"name": daily_name, "labels": daily.index.strftime("%Y-%m-%d").tolist(), "avg": _r(daily)},
        "congestion": _congestion_counts(df[v], _thresholds(meta, full)),
    }
    w = df[df.weather_main != "Unknown"].groupby("weather_main")[v].agg(["mean", "count"])
    w = w[w["count"] >= 20].sort_values("mean", ascending=False)
    out["weather"] = {"labels": w.index.tolist(), "avg": _r(w["mean"]), "counts": w["count"].astype(int).tolist()}
    return out


def dashboard(meta: dict = None) -> dict:
    df = load_dataset()
    v = "traffic_volume"
    last = df.iloc[-1]
    last_week = df[df.date_time > df.date_time.max() - pd.Timedelta(days=7)]
    last30 = df[df.date_time > df.date_time.max() - pd.Timedelta(days=30)].set_index("date_time")[v].resample("D").mean().dropna()
    hourly = df.groupby("hour")[v].mean().reindex(range(24))
    dow = df.groupby("dow")[v].mean().reindex(range(7))
    top = hourly.sort_values(ascending=False).head(3)
    th = _thresholds(meta, df)
    from src.predict import classify
    last_cat = classify(float(last[v]), th)[0]
    return {
        "dataset_range": f"{df.date_time.min():%d %b %Y} - {df.date_time.max():%d %b %Y}",
        "latest_record": {"time": f"{last.date_time:%d %b %Y, %H:%M}", "volume": int(last[v]), "category": last_cat,
                          "color": config.CATEGORY_COLORS[last_cat], "weather": last.weather_main},
        "avg_volume": round(float(df[v].mean())),
        "peak_hours": [f"{int(h):02d}:00" for h in top.index], "peak_avg": round(float(top.iloc[0])),
        "line_last_week": {"labels": last_week.date_time.dt.strftime("%a %d %b %H:00").tolist(),
                           "values": last_week[v].astype(int).tolist()},
        "daily_30": {"labels": last30.index.strftime("%d %b").tolist(), "values": _r(last30)},
        "hourly": {"labels": [f"{h:02d}:00" for h in range(24)], "avg": _r(hourly)},
        "weekly": {"labels": [d[:3] for d in DAY_NAMES], "avg": _r(dow)},
        "congestion": _congestion_counts(df[v], th),
        "thresholds": [round(t) for t in th],
        "hist_vs_pred": (meta or {}).get("series", {}).get("last_week"),
    }


def report_csv(p: dict, meta: dict = None) -> str:
    a = analytics(p, meta)
    buf = io.StringIO()
    buf.write("Traffic analysis report (HISTORICAL data - not live traffic)\n")
    if meta:
        buf.write(f"Dataset:,{meta['dataset']['label']}\n")
    buf.write("Filters:," + "; ".join(f"{k}={v}" for k, v in p.items() if v) + "\n")
    if a.get("empty"):
        buf.write("No records match the selected filters.\n")
        return buf.getvalue()
    k = a["kpis"]
    buf.write(f"Records:,{a['rows']}\nPeriod:,{k['period']}\nAverage volume (veh/h):,{k['avg']}\n"
              f"Peak hours:,{' '.join(k['peak_hours'])}\n\n")
    buf.write("Hour,Average volume,P25,P75,Weekday avg,Weekend avg\n")
    for i, lab in enumerate(a["hourly"]["labels"]):
        buf.write(f"{lab},{a['hourly']['avg'][i]},{a['hourly']['p25'][i]},{a['hourly']['p75'][i]},"
                  f"{a['weekday_weekend']['weekday'][i]},{a['weekday_weekend']['weekend'][i]}\n")
    buf.write("\nDay of week,Average volume\n")
    for lab, val in zip(a["dow"]["labels"], a["dow"]["avg"]):
        buf.write(f"{lab},{val}\n")
    buf.write("\nMonth,Average volume\n")
    for lab, val in zip(a["monthly"]["labels"], a["monthly"]["avg"]):
        buf.write(f"{lab},{val}\n")
    buf.write("\nWeather,Average volume,Records\n")
    for lab, val, n in zip(a["weather"]["labels"], a["weather"]["avg"], a["weather"]["counts"]):
        buf.write(f"{lab},{val},{n}\n")
    return buf.getvalue()
