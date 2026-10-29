"""Tests for preprocessing, model pipeline and prediction.  Run:  python -m pytest -q"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.predict import InputValidationError, TrafficPredictor, classify  # noqa: E402
from src.preprocess import (DatasetError, FEATURE_COLUMNS, build_features, chronological_split,  # noqa: E402
                            clean_data, inputs_to_frame, make_preprocessor, validate_dataframe)


def raw_frame(n_days=90):
    idx = pd.date_range("2016-01-01", periods=24 * n_days, freq="h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "holiday": "None", "temp": 280 + rng.normal(0, 3, len(idx)), "rain_1h": 0.0, "snow_1h": 0.0,
        "clouds_all": 40, "weather_main": "Clear", "weather_description": "sky is clear",
        "date_time": idx.strftime("%Y-%m-%d %H:%M:%S"),
        "traffic_volume": (3000 + 2000 * np.sin(idx.hour / 24 * 2 * np.pi) + rng.normal(0, 100, len(idx))).astype(int),
    })
    df.loc[0, "holiday"] = "New Year's Day"
    return df


# ------------------------------------------------------------------ preprocessing
def test_validate_flags_missing_columns():
    assert any("Missing required" in p for p in validate_dataframe(pd.DataFrame({"a": [1]})))


def test_clean_removes_duplicates_and_outliers():
    df = raw_frame()
    df = pd.concat([df, df.iloc[:50]])                       # 50 duplicates
    df.loc[10, "temp"] = 0.0                                 # 0 Kelvin sensor error
    df.loc[11, "rain_1h"] = 9831.3                           # absurd rainfall
    out, rep = clean_data(df, return_report=True)
    assert rep["duplicates_removed"] == 50
    assert out.date_time.is_unique and out.date_time.is_monotonic_increasing
    assert pd.isna(out.loc[10, "temp_c"]) and pd.isna(out.loc[11, "rain_1h"])
    assert out.temp_c.dropna().between(-40, 40).all()       # Kelvin converted to Celsius


def test_holiday_flag_spreads_over_whole_day():
    out = clean_data(raw_frame())
    first_day = out[out.date_time.dt.date == pd.Timestamp("2016-01-01").date()]
    assert (first_day.is_holiday == 1).all()
    assert (out[out.date_time.dt.date > pd.Timestamp("2016-01-01").date()].is_holiday == 0).all()


def test_build_features_columns_and_cyclical_hours():
    X = build_features(clean_data(raw_frame()))
    assert list(X.columns) == FEATURE_COLUMNS
    assert np.allclose(X.hour_sin ** 2 + X.hour_cos ** 2, 1)


def test_chronological_split_has_no_overlap():
    df = clean_data(raw_frame())
    X, y = build_features(df), df.traffic_volume
    Xtr, Xte, ytr, yte = chronological_split(X, y, 0.2)
    assert df.date_time.iloc[len(Xtr) - 1] < df.date_time.iloc[len(Xtr)]
    assert len(Xtr) + len(Xte) == len(X) and abs(len(Xte) / len(X) - 0.2) < 0.01


def test_preprocessor_fitted_on_train_only_handles_unseen_category():
    df = clean_data(raw_frame())
    X, y = build_features(df), df.traffic_volume
    Xtr, Xte, _, _ = chronological_split(X, y, 0.2)
    prep = make_preprocessor().fit(Xtr)
    Xte = Xte.copy(); Xte["weather_main"] = "Tornado"        # never seen in training
    assert prep.transform(Xte).shape[0] == len(Xte)


def test_single_row_features_match_training_features():
    df = clean_data(raw_frame())
    row = df.iloc[100]
    batch = build_features(df.iloc[[100]])
    single = inputs_to_frame(str(row.date_time.date()), row.date_time.hour, bool(row.is_holiday), row.weather_main,
                             row.temp_c, row.rain_1h, row.snow_1h, row.clouds_all)
    pd.testing.assert_frame_equal(batch.reset_index(drop=True), single.reset_index(drop=True), check_dtype=False)


# ------------------------------------------------------------------ congestion
def test_classify_thresholds():
    th = [1000, 3000, 5000]
    assert classify(10, th)[0] == "Low" and classify(1000, th)[0] == "Moderate"
    assert classify(4999, th)[0] == "High" and classify(9000, th)[0] == "Severe"


# ------------------------------------------------------------------ prediction (needs trained model)
needs_model = pytest.mark.skipif(not os.path.exists(config.MODEL_PATH), reason="run `python src/train_model.py` first")
GOOD = {"date": "2026-11-03", "hour": 8, "holiday": False, "weather_main": "Clear", "temperature_c": 10}


@needs_model
def test_prediction_is_model_output_and_varies_with_inputs():
    p = TrafficPredictor()
    rush = p.predict(GOOD)
    night = p.predict({**GOOD, "hour": 3})
    assert rush["predicted_volume"] > night["predicted_volume"] * 2          # rush hour >> 3 a.m.
    assert rush["category"] in config.CATEGORIES and 0 <= rush["intensity_pct"] <= 100
    assert len(rush["profile"]["volumes"]) == 24
    assert rush["profile"]["volumes"][8] == rush["predicted_volume"]          # same model, same number
    assert rush["kind"] == "forecast"


@needs_model
def test_prediction_is_deterministic():
    p = TrafficPredictor()
    assert p.predict(GOOD)["predicted_volume"] == p.predict(GOOD)["predicted_volume"]


@needs_model
def test_holiday_reduces_weekday_traffic():
    p = TrafficPredictor()
    assert p.predict({**GOOD, "holiday": True})["predicted_volume"] < p.predict(GOOD)["predicted_volume"]


@needs_model
@pytest.mark.parametrize("bad", [
    {**GOOD, "hour": 24}, {**GOOD, "hour": "x"}, {**GOOD, "date": "not-a-date"}, {**GOOD, "weather_main": "Lava"},
    {**GOOD, "temperature_c": 500}, {**GOOD, "day_of_week": "Friday"},        # 2026-11-03 is a Tuesday
])
def test_invalid_inputs_are_rejected(bad):
    with pytest.raises(InputValidationError):
        TrafficPredictor().predict(bad)


def test_missing_model_gives_clear_error(tmp_path):
    from src.predict import ModelNotReadyError
    with pytest.raises(ModelNotReadyError):
        TrafficPredictor(str(tmp_path / "none.joblib"), str(tmp_path / "none.json")).predict(GOOD)


@needs_model
def test_flask_api_roundtrip(tmp_path, monkeypatch):
    import app as app_module
    from src import database as db
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "t.db"))
    db.init_db()
    c = app_module.app.test_client()
    r = c.post("/api/predict", json=GOOD)
    assert r.status_code == 200 and r.get_json()["predicted_volume"] > 0
    assert c.post("/api/predict", json={**GOOD, "hour": 99}).status_code == 422
    assert c.get("/api/history/export.csv").data.decode().startswith("Predicted at,")
