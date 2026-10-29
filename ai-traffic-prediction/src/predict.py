"""Load the saved pipeline and turn user inputs into REAL model predictions.

Every number returned here comes from ``pipeline.predict`` applied to features built by the same
``build_features`` code that was used in training.  Nothing is hard-coded.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.preprocess import DAY_NAMES, inputs_to_frame  # noqa: E402


class ModelNotReadyError(RuntimeError):
    """Trained model (or its metadata) is missing or unreadable."""


class InputValidationError(ValueError):
    """User input failed validation; ``errors`` maps field -> message."""

    def __init__(self, errors: dict):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


def classify(volume: float, thresholds) -> tuple:
    """Return (category name, index 0-3) for a volume using ascending thresholds."""
    idx = int(np.searchsorted(np.asarray(thresholds), volume, side="right"))
    return config.CATEGORIES[idx], idx


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on", "y")


def _num(payload, key, errors, lo, hi, required=False, label=None):
    label = label or key
    raw = payload.get(key)
    if raw in (None, ""):
        if required:
            errors[key] = f"{label} is required."
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        errors[key] = f"{label} must be a number."
        return None
    if not np.isfinite(val) or val < lo or val > hi:
        errors[key] = f"{label} must be between {lo:g} and {hi:g}."
        return None
    return val


class TrafficPredictor:
    def __init__(self, model_path=config.MODEL_PATH, meta_path=config.METADATA_PATH):
        self.model_path, self.meta_path = model_path, meta_path
        self._bundle = self._meta = None
        self._mtime = None

    # ------------------------------------------------------------------ loading
    def _stamp(self):
        try:
            return (os.path.getmtime(self.model_path), os.path.getmtime(self.meta_path))
        except OSError:
            return None

    def load(self):
        stamp = self._stamp()
        if stamp is None:
            raise ModelNotReadyError("The trained model was not found. Run `python src/train_model.py` first.")
        if stamp != self._mtime:        # (re)load automatically after retraining
            try:
                self._bundle = joblib.load(self.model_path)
                with open(self.meta_path) as fh:
                    self._meta = json.load(fh)
            except Exception as exc:
                raise ModelNotReadyError(f"The model files could not be loaded ({exc}). Retrain the model.") from exc
            self._mtime = stamp
        return self._bundle, self._meta

    @property
    def ready(self) -> bool:
        try:
            self.load()
            return True
        except ModelNotReadyError:
            return False

    @property
    def metadata(self) -> dict:
        return self.load()[1]

    # ------------------------------------------------------------------ validation
    def validate(self, payload: dict) -> dict:
        meta = self.metadata
        sup = meta["supported_inputs"]
        errors = {}

        date = None
        raw_date = (payload.get("date") or "").strip() if isinstance(payload.get("date"), str) else payload.get("date")
        if not raw_date:
            errors["date"] = "Date is required."
        else:
            try:
                date = pd.Timestamp(raw_date).normalize()
                if not (pd.Timestamp("2000-01-01") <= date <= pd.Timestamp("2100-12-31")):
                    errors["date"] = "Date must be between 2000 and 2100."
            except Exception:
                errors["date"] = "Date is not valid (use YYYY-MM-DD)."

        hour = _num(payload, "hour", errors, 0, 23, required=True, label="Hour")
        if hour is not None and int(hour) != hour:
            errors["hour"] = "Hour must be a whole number from 0 to 23."

        dow_in = payload.get("day_of_week")
        if dow_in not in (None, "") and date is not None and "date" not in errors:
            names = [d.lower() for d in DAY_NAMES]
            val = str(dow_in).strip().lower()
            given = names.index(val) if val in names else (int(val) if val.isdigit() and 0 <= int(val) <= 6 else None)
            if given is None:
                errors["day_of_week"] = "Day of week is not valid."
            elif given != date.dayofweek:
                errors["day_of_week"] = f"{date.date()} is a {DAY_NAMES[date.dayofweek]}, not {DAY_NAMES[given]}."

        weather = None
        if sup["weather"]:
            weather = (payload.get("weather_main") or "").strip()
            if not weather:
                errors["weather_main"] = "Weather condition is required."
            elif weather not in meta["weather_options"]:
                errors["weather_main"] = f"Unsupported weather condition. Choose one of: {', '.join(meta['weather_options'])}."
        temp = None
        if sup["temperature"]:
            temp = _num(payload, "temperature_c", errors, -60, 60, required=True, label="Temperature (°C)")
        rain = _num(payload, "rain_1h", errors, 0, 100, label="Rain (mm/h)") if sup["precipitation"] else None
        snow = _num(payload, "snow_1h", errors, 0, 100, label="Snow (mm/h)") if sup["precipitation"] else None
        clouds = _num(payload, "clouds_all", errors, 0, 100, label="Cloud cover (%)") if sup["clouds"] else None

        if errors:
            raise InputValidationError(errors)

        assumed = []
        # fill the optional weather details with simple, documented defaults when not supplied
        if sup["precipitation"]:
            if rain is None:
                rain = 1.0 if weather in ("Rain", "Drizzle", "Thunderstorm") else 0.0
                assumed.append(f"rain = {rain:g} mm/h (default for '{weather}')")
            if snow is None:
                snow = 0.3 if weather == "Snow" else 0.0
                assumed.append(f"snow = {snow:g} mm/h (default for '{weather}')")
        if sup["clouds"] and clouds is None:
            clouds = {"Clear": 10}.get(weather, 75 if weather == "Clouds" else 90)
            assumed.append(f"cloud cover = {clouds:g}% (default for '{weather}')")

        return {"date": date.strftime("%Y-%m-%d"), "hour": int(hour), "is_holiday": _to_bool(payload.get("holiday")),
                "weather_main": weather or "Unknown", "temperature_c": temp, "rain_1h": rain, "snow_1h": snow,
                "clouds_all": clouds, "assumed": assumed, "day_of_week": DAY_NAMES[date.dayofweek]}

    # ------------------------------------------------------------------ prediction
    def _predict_volume(self, pipeline, c: dict, hour: int) -> float:
        X = inputs_to_frame(c["date"], hour, c["is_holiday"], c["weather_main"], c["temperature_c"],
                            c["rain_1h"], c["snow_1h"], c["clouds_all"])
        return float(max(pipeline.predict(X)[0], 0.0))

    def predict(self, payload: dict) -> dict:
        bundle, meta = self.load()
        c = self.validate(payload)
        pipe, th = bundle["pipeline"], bundle["thresholds"]

        volume = self._predict_volume(pipe, c, c["hour"])
        category, level = classify(volume, th)
        profile = [self._predict_volume(pipe, c, h) for h in range(24)]
        prof_cat = [classify(v, th)[0] for v in profile]
        peak_h = int(np.argmax(profile))
        capacity = bundle["capacity_reference"]

        return {
            "predicted_volume": int(round(volume)),
            "category": category, "category_level": level, "color": config.CATEGORY_COLORS[category],
            "intensity_pct": int(round(min(volume / capacity, 1.0) * 100)),
            "thresholds": [round(t) for t in th],
            "profile": {"hours": list(range(24)), "volumes": [int(round(v)) for v in profile], "categories": prof_cat},
            "peak_hour": peak_h, "peak_volume": int(round(profile[peak_h])),
            "recommendation": self._recommend(c["hour"], profile, category, th),
            "inputs": c, "model": bundle["algorithm"], "assumptions": c["assumed"],
            "outside_training_range": not (pd.Timestamp(meta["dataset"]["date_min"]).year - 1 <= int(c["date"][:4]) <= pd.Timestamp(meta["dataset"]["date_max"]).year + 5),
            "disclaimer": config.VOLUME_DISCLAIMER,
            "kind": "forecast",
        }

    def predict_at(self, date, hour, weather="Clear", temperature_c=None, holiday=False) -> dict:
        """Convenience for internal use (dashboard 'next hour' estimate)."""
        meta = self.metadata
        if temperature_c is None:
            temperature_c = meta["median_temp_by_month"].get(str(pd.Timestamp(date).month), 10.0)
        weather = weather if weather in meta["weather_options"] else (meta["weather_options"] or ["Unknown"])[0]
        return self.predict({"date": date, "hour": hour, "holiday": holiday, "weather_main": weather,
                             "temperature_c": temperature_c})

    @staticmethod
    def _recommend(hour, profile, category, th) -> str:
        v = profile[hour]
        window = [h for h in range(max(0, hour - 3), min(23, hour + 3) + 1) if h != hour]
        best = min(window, key=lambda h: profile[h]) if window else hour
        fmt = lambda h: f"{h:02d}:00"
        if category in ("High", "Severe"):
            if profile[best] < v * 0.9:
                drop = round(100 * (1 - profile[best] / v))
                return (f"Expect heavy traffic around {fmt(hour)}. Travelling near {fmt(best)} is forecast to carry "
                        f"about {drop}% less traffic - consider shifting your trip if you can.")
            return (f"Expect heavy traffic around {fmt(hour)}, and nearby hours are similar. "
                    f"Allow extra travel time or consider public transport.")
        if category == "Moderate":
            return (f"Traffic should be moderate around {fmt(hour)}. Leaving a little earlier than usual is sensible; "
                    f"the quietest nearby hour is {fmt(best)}.")
        return f"Traffic should be light around {fmt(hour)} - a good time to travel."
