"""Train, compare and save the traffic-volume model.

Usage:   python src/train_model.py [--data path/to/traffic.csv]

Workflow
  1. load CSV (a labelled synthetic sample is generated if none exists)
  2. validate + clean (duplicates, impossible values, holidays, Kelvin -> Celsius)
  3. engineer features (shared with prediction)
  4. chronological 80/20 split - the newest 20% is the untouched test set
  5. compare Linear Regression, Random Forest and Gradient Boosting with expanding-window
     time-series cross-validation on the TRAINING part; pick the lowest CV RMSE
  6. evaluate every model once on the test set (MAE, RMSE, R2) - reporting only, not used for selection
  7. save pipeline (joblib) + metadata (json) for the Flask app
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src.preprocess import (DatasetError, build_features, chronological_split, clean_data,  # noqa: E402
                            load_raw, make_preprocessor)


def dataset_info(path: str) -> dict:
    """Is the CSV our synthetic sample or user-provided/real data?  (decided by file hash)"""
    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    info = {}
    if os.path.exists(config.DATASET_INFO_PATH):
        with open(config.DATASET_INFO_PATH) as fh:
            info = json.load(fh)
    if info.get("synthetic") and info.get("sha256") == sha:
        return {"synthetic": True, "label": "Synthetic sample data (NOT real measurements)",
                "source": "src/generate_sample_data.py"}
    return {"synthetic": False, "label": "User-provided / real dataset", "source": os.path.basename(path)}


def candidate_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1,
                                               random_state=config.RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                                           random_state=config.RANDOM_STATE),
    }


def train(data_path: str = config.DATA_PATH, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()

    if not os.path.exists(data_path):
        if os.path.abspath(data_path) == os.path.abspath(config.DATA_PATH):
            from src.generate_sample_data import write
            log(f"[1/6] No dataset at {data_path} - generating SYNTHETIC sample data...")
            write(data_path)
        else:
            raise DatasetError(f"Dataset not found: {data_path}")
    info = dataset_info(data_path)
    log(f"[1/6] Dataset: {data_path}  ({info['label']})")

    raw = load_raw(data_path)
    df, report = clean_data(raw, return_report=True)
    log(f"[2/6] Cleaned: {report['rows_in']:,} -> {report['rows_out']:,} rows "
        f"({report['duplicates_removed']} duplicates removed, "
        f"{report.get('outliers_nulled', 0)} impossible readings blanked)")

    X = build_features(df)
    y = df[config.TARGET].astype(float)
    X_train, X_test, y_train, y_test = chronological_split(X, y, config.TEST_SIZE)
    t_train, t_test = df.date_time.iloc[:len(X_train)], df.date_time.iloc[len(X_train):]
    log(f"[3/6] Chronological split: train {len(X_train):,} rows ({t_train.min():%Y-%m-%d} -> {t_train.max():%Y-%m-%d}), "
        f"test {len(X_test):,} rows ({t_test.min():%Y-%m-%d} -> {t_test.max():%Y-%m-%d})")

    log("[4/6] Comparing models with time-series cross-validation (train data only)...")
    results, fitted = [], {}
    for name, est in candidate_models().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", est)])
        cv = ev.time_series_cv(pipe, X_train, y_train, config.CV_SPLITS)
        pipe.fit(X_train, y_train)
        test = ev.regression_metrics(y_test, pipe.predict(X_test))
        train_m = ev.regression_metrics(y_train, pipe.predict(X_train))
        fitted[name] = pipe
        results.append({"model": name, **cv,
                        "test_mae": test["mae"], "test_rmse": test["rmse"], "test_r2": test["r2"],
                        "train_rmse": train_m["rmse"], "train_r2": train_m["r2"]})
        log(f"      {name:<20} CV RMSE {cv['cv_rmse']:8.1f} | test MAE {test['mae']:7.1f}  "
            f"RMSE {test['rmse']:7.1f}  R2 {test['r2']:.3f}")

    best = min(results, key=lambda r: r["cv_rmse"])      # selection uses validation, NOT the test set
    best_name = best["model"]
    pipe = fitted[best_name]
    for r in results:
        r["selected"] = r["model"] == best_name
    log(f"[5/6] Selected model: {best_name} (lowest cross-validated RMSE)")

    y_pred = pipe.predict(X_test)
    importance = ev.grouped_importance(pipe, X_test, y_test)
    series = ev.prediction_series(t_test, y_test, y_pred)

    # congestion thresholds from TRAINING volumes only
    if config.ABSOLUTE_THRESHOLDS:
        thresholds = [float(t) for t in config.ABSOLUTE_THRESHOLDS]
        th_source = "fixed values from config.ABSOLUTE_THRESHOLDS"
    else:
        thresholds = [float(np.percentile(y_train, p)) for p in config.THRESHOLD_PERCENTILES]
        th_source = f"percentiles {config.THRESHOLD_PERCENTILES} of training volumes"
    capacity_ref = float(np.percentile(y_train, 99))

    weather_values = sorted(v for v in df.weather_main.unique() if v != "Unknown")
    metadata = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": best_name,
        "selection_rule": f"lowest mean RMSE over {config.CV_SPLITS}-fold expanding-window time-series CV on the training set",
        "dataset": {**info, "path": os.path.basename(data_path), "rows_raw": report["rows_in"],
                    "rows_clean": report["rows_out"], "date_min": str(df.date_time.min()),
                    "date_max": str(df.date_time.max()), "cleaning": report},
        "split": {"type": "chronological", "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                  "train_pct": round(100 * (1 - config.TEST_SIZE)), "test_pct": round(100 * config.TEST_SIZE),
                  "train_range": [str(t_train.min()), str(t_train.max())],
                  "test_range": [str(t_test.min()), str(t_test.max())]},
        "metrics": {k: best[f"test_{k}"] for k in ("mae", "rmse", "r2")},
        "train_metrics": {"rmse": best["train_rmse"], "r2": best["train_r2"]},
        "comparison": results,
        "feature_importance": importance,
        "series": series,
        "thresholds": {"values": thresholds, "source": th_source, "capacity_reference": capacity_ref,
                       "categories": config.CATEGORIES},
        "supported_inputs": report["supported_inputs"],
        "weather_options": weather_values,
        "temp_range_c": [float(df.temp_c.quantile(0.01)) if df.temp_c.notna().any() else -20.0,
                         float(df.temp_c.quantile(0.99)) if df.temp_c.notna().any() else 35.0],
        "median_temp_by_month": {int(m): float(v) for m, v in df.groupby(df.date_time.dt.month).temp_c.median().dropna().items()},
        "has_location": False,
        "note": "The dataset has no road/location column, so predictions are for the single monitored road segment.",
    }

    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    joblib.dump({"pipeline": pipe, "algorithm": best_name, "thresholds": thresholds,
                 "capacity_reference": capacity_ref, "version": 1}, config.MODEL_PATH)
    with open(config.METADATA_PATH, "w") as fh:
        json.dump(metadata, fh, indent=2)
    log(f"[6/6] Saved {config.MODEL_PATH}\n      Saved {config.METADATA_PATH}")
    m = metadata["metrics"]
    log(f"\nFinal model on held-out test data -> MAE {m['mae']:.1f} | RMSE {m['rmse']:.1f} | R2 {m['r2']:.3f}"
        f"   (done in {time.time() - t0:.0f}s)")
    return metadata


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the traffic prediction model")
    ap.add_argument("--data", default=config.DATA_PATH, help="CSV path (default: data/traffic.csv)")
    args = ap.parse_args()
    try:
        train(args.data)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
