"""Load the saved pipeline and turn user inputs into real model predictions.

Nothing here returns a hard-coded value: every number comes from
`pipeline.predict` on features built by the same code used in training.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import evaluate as ev
from src import preprocess as pp

CONGESTION_NOTE = ("Predicted traffic VOLUME (vehicles per hour) is not the same as road "
                   "congestion. Categories compare the prediction with volumes seen in the "
                   "historical data; true congestion also depends on road capacity, "
                   "incidents and speed, which this dataset does not contain.")


class ModelNotReadyError(RuntimeError):
    """The trained model file is missing or unreadable."""


class InputValidationError(ValueError):
    """User input failed validation; `.errors` maps field -> message."""

    def __init__(self, errors: dict):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


_cache: dict = {}


def load_bundle(model_path: Path | str) -> dict:
    """Load the joblib bundle (cached; reloads automatically after retraining)."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelNotReadyError("No trained model found. Run `python src/train_model.py` "
                                 "(or use Retrain on the Model Performance page).")
    mtime = model_path.stat().st_mtime
    hit = _cache.get(str(model_path))
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        bundle = joblib.load(model_path)
    except Exception as exc:
        raise ModelNotReadyError(f"The saved model could not be loaded ({exc}). "
                                 "Please retrain it.") from exc
    _cache[str(model_path)] = (mtime, bundle)
    return bundle


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on", "holiday"}


def validate_inputs(form: dict, bundle: dict) -> dict:
    """Return cleaned inputs or raise InputValidationError with per-field messages."""
    errors: dict = {}
    cleaned: dict = {}

    # date
    try:
        d = datetime.strptime(str(form.get("date", "")).strip(), "%Y-%m-%d").date()
        if not (date(2000, 1, 1) <= d <= date(2100, 12, 31)):
            raise ValueError
        cleaned["date"] = d
    except ValueError:
        errors["date"] = "Enter a valid date (YYYY-MM-DD) between 2000 and 2100."

    # hour
    try:
        hour = int(str(form.get("hour", "")).strip())
        if not 0 <= hour <= 23:
            raise ValueError
        cleaned["hour"] = hour
    except ValueError:
        errors["hour"] = "Hour must be a whole number from 0 to 23."

    # day of week (must agree with the date)
    raw_dow = str(form.get("day_of_week", "")).strip()
    if "date" in cleaned:
        actual = cleaned["date"].weekday()
        cleaned["day_of_week"] = actual
        if raw_dow != "":
            try:
                given = int(raw_dow)
                if given != actual:
                    errors["day_of_week"] = (f"{cleaned['date']} is a {pp.DAY_NAMES[actual]}, "
                                             f"not {pp.DAY_NAMES[given] if 0 <= given <= 6 else 'that day'}.")
            except ValueError:
                errors["day_of_week"] = "Day of week must be a number from 0 (Mon) to 6 (Sun)."

    # holiday
    cleaned["is_holiday"] = _truthy(form.get("holiday", "0"))

    # weather
    weather = str(form.get("weather", "")).strip()
    if weather not in bundle["weather_options"]:
        errors["weather"] = "Choose one of: " + ", ".join(bundle["weather_options"]) + "."
    else:
        cleaned["weather"] = weather

    # temperature (Celsius)
    lo, hi = bundle["temp_range_c"]
    try:
        temp = float(str(form.get("temperature", "")).strip())
        if not np.isfinite(temp) or not (lo - 10 <= temp <= hi + 10):
            errors["temperature"] = (f"Temperature must be between {lo - 10:.0f} and "
                                     f"{hi + 10:.0f} °C (the range seen in the training data ±10).")
        else:
            cleaned["temperature"] = temp
    except ValueError:
        errors["temperature"] = "Enter the temperature in °C as a number."

    if errors:
        raise InputValidationError(errors)
    return cleaned


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def _predict_rows(bundle: dict, rows: list[pd.DataFrame]) -> np.ndarray:
    X = pd.concat(rows, ignore_index=True)[bundle["feature_columns"]]
    return np.clip(bundle["pipeline"].predict(X), 0, None)


def categorize(bundle: dict, volume: float) -> str:
    return ev.classify_volume(volume, bundle["thresholds"], bundle["labels"])


def intensity_percent(bundle: dict, volume: float) -> int:
    """0-100 gauge value relative to the busiest typical hours in training data."""
    return int(round(min(volume / bundle["capacity_reference"], 1.0) * 100))


def predict_day_profile(bundle: dict, cleaned: dict) -> list[float]:
    """Predict all 24 hours of the chosen day under the same other conditions."""
    rows = [pp.make_feature_row(cleaned["date"], h, cleaned["is_holiday"],
                                cleaned["weather"], cleaned["temperature"]) for h in range(24)]
    return [float(v) for v in _predict_rows(bundle, rows)]


def _recommendation(bundle: dict, category: str, hour: int, volume: float,
                    profile: list[float]) -> str:
    peak_hour = int(np.argmax(profile))
    if category == "Low":
        return "Traffic is expected to be light at this hour - a good time to travel."
    if category == "Moderate":
        return ("Moderate traffic is expected. Allow a little extra time, especially if "
                f"you are travelling near the day's busiest hour ({peak_hour:02d}:00).")
    # High / Severe: look for a clearly quieter hour within +/-3 hours
    window = [h for h in range(max(0, hour - 3), min(23, hour + 3) + 1) if h != hour]
    best = min(window, key=lambda h: profile[h]) if window else hour
    saving = (1 - profile[best] / volume) * 100 if volume else 0
    if best != hour and saving >= 10:
        return (f"Heavy traffic is predicted. Travelling around {best:02d}:00 is predicted to "
                f"carry about {saving:.0f}% less traffic. The day's busiest hour is "
                f"{peak_hour:02d}:00.")
    return ("Heavy traffic is predicted around this time. No clearly quieter hour exists within "
            f"3 hours, so consider another day or allow extra time (busiest hour: {peak_hour:02d}:00).")


def predict_traffic(form: dict, model_path: Path | str) -> dict:
    """Validate the form, run the saved pipeline and build the response payload."""
    bundle = load_bundle(model_path)
    cleaned = validate_inputs(form, bundle)

    profile = predict_day_profile(bundle, cleaned)
    volume = profile[cleaned["hour"]]          # the 24-hour profile includes the requested hour
    category = categorize(bundle, volume)
    labels = bundle["labels"]
    cuts = bundle["thresholds"]
    return {
        "predicted_volume": int(round(volume)),
        "category": category,
        "intensity_percent": intensity_percent(bundle, volume),
        "recommendation": _recommendation(bundle, category, cleaned["hour"], volume, profile),
        "day_profile": [int(round(v)) for v in profile],
        "peak_hour": int(np.argmax(profile)),
        "quietest_hour": int(np.argmin(profile)),
        "inputs": {
            "date": cleaned["date"].isoformat(),
            "hour": cleaned["hour"],
            "day_of_week": cleaned["day_of_week"],
            "day_name": pp.DAY_NAMES[cleaned["day_of_week"]],
            "holiday": cleaned["is_holiday"],
            "weather": cleaned["weather"],
            "temperature": cleaned["temperature"],
        },
        "thresholds": [{"label": labels[0], "range": f"< {cuts[0]:.0f}"},
                       {"label": labels[1], "range": f"{cuts[0]:.0f} - {cuts[1]:.0f}"},
                       {"label": labels[2], "range": f"{cuts[1]:.0f} - {cuts[2]:.0f}"},
                       {"label": labels[3], "range": f">= {cuts[2]:.0f}"}],
        "model": bundle["model_name"],
        "data_source": bundle["data_source"].get("label", ""),
        "note": CONGESTION_NOTE,
        "kind": "forecast",   # a model forecast, never a live observation
    }


def typical_conditions(bundle: dict, when: datetime) -> dict:
    """Typical weather for a month, from training data (for 'typical day' forecasts)."""
    clim = bundle["climatology"].get(str(when.month)) or bundle["climatology"].get(when.month)
    if clim is None:
        clim = {"weather_main": bundle["weather_options"][0], "temp_c": 10.0}
    return clim
