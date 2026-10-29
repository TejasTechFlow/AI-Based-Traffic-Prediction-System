"""Evaluation helpers: metrics, time-series cross-validation, importances, chart series."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def time_series_cv(pipeline, X, y, n_splits: int) -> dict:
    """Expanding-window CV on the TRAINING data only (each fold validates on later data)."""
    maes, rmses, r2s = [], [], []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
        model = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        m = regression_metrics(y.iloc[va], model.predict(X.iloc[va]))
        maes.append(m["mae"]); rmses.append(m["rmse"]); r2s.append(m["r2"])
    return {"cv_mae": float(np.mean(maes)), "cv_rmse": float(np.mean(rmses)),
            "cv_r2": float(np.mean(r2s)), "cv_rmse_std": float(np.std(rmses))}


def grouped_importance(pipeline, X_test, y_test, seed=42, max_rows=4000) -> list:
    """Permutation importance per ORIGINAL input feature (works for every model type)."""
    if len(X_test) > max_rows:
        idx = np.random.default_rng(seed).choice(len(X_test), max_rows, replace=False)
        X_test, y_test = X_test.iloc[idx], y_test.iloc[idx]
    groups = {
        "Hour of day": ["hour", "hour_sin", "hour_cos", "weekday_hour"],
        "Day of week": ["day_of_week", "is_weekend"],
        "Month / season": ["month_sin", "month_cos"],
        "Holiday": ["is_holiday"],
        "Temperature": ["temp_c"],
        "Rain": ["rain_1h"],
        "Snow": ["snow_1h"],
        "Cloud cover": ["clouds_all"],
        "Weather type": ["weather_main"],
    }
    base = pipeline.predict(X_test)
    base_rmse = float(np.sqrt(mean_squared_error(y_test, base)))
    rng = np.random.default_rng(seed)
    out = []
    for name, cols in groups.items():
        incr = []
        for _ in range(3):
            Xp = X_test.copy()
            perm = rng.permutation(len(Xp))
            # shuffle the whole group together so related columns stay consistent
            Xp[cols] = Xp[cols].to_numpy()[perm]
            incr.append(np.sqrt(mean_squared_error(y_test, pipeline.predict(Xp))) - base_rmse)
        out.append({"feature": name, "importance": float(max(np.mean(incr), 0.0))})
    total = sum(o["importance"] for o in out) or 1.0
    for o in out:
        o["share"] = o["importance"] / total
    return sorted(out, key=lambda d: d["importance"], reverse=True)


def prediction_series(timestamps, y_true, y_pred) -> dict:
    """Compact series for the performance charts."""
    df = pd.DataFrame({"t": pd.to_datetime(timestamps).to_numpy(),
                       "actual": np.asarray(y_true), "pred": np.asarray(y_pred)})
    last_week = df.tail(24 * 7)
    daily = df.set_index("t").resample("D").mean().dropna()
    scatter = df.sample(min(400, len(df)), random_state=0)
    return {
        "last_week": {
            "labels": last_week.t.dt.strftime("%a %d %b %H:00").tolist(),
            "actual": last_week.actual.round(0).tolist(),
            "predicted": last_week.pred.round(0).tolist(),
        },
        "daily": {
            "labels": daily.index.strftime("%Y-%m-%d").tolist(),
            "actual": daily.actual.round(0).tolist(),
            "predicted": daily.pred.round(0).tolist(),
        },
        "scatter": [{"x": float(a), "y": float(p)} for a, p in zip(scatter.actual.round(0), scatter.pred.round(0))],
    }
