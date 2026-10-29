"""Data loading, validation, cleaning and feature engineering.

The SAME ``build_features`` function is used for training and for single-row prediction,
so the model always sees identically-prepared inputs.  Everything that *learns* from data
(imputation medians, scaling, one-hot categories) lives inside the sklearn ``ColumnTransformer``
and is therefore fitted on training data only (no leakage).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

REQUIRED_COLUMNS = ["date_time", "traffic_volume"]
OPTIONAL_COLUMNS = ["holiday", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main"]

NUMERIC_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "is_holiday",
                    "temp_c", "rain_1h", "snow_1h", "clouds_all"]
CATEGORICAL_FEATURES = ["hour", "day_of_week", "weekday_hour", "weather_main"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class DatasetError(ValueError):
    """Raised when a dataset is missing or not compatible."""


# --------------------------------------------------------------------------- loading
def load_raw(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise DatasetError(f"Dataset not found at {path}.")
    try:
        return pd.read_csv(path)
    except Exception as exc:  # malformed CSV
        raise DatasetError(f"Could not read CSV: {exc}") from exc


def validate_dataframe(df: pd.DataFrame) -> list:
    """Return a list of human-readable problems (empty list == compatible)."""
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"Missing required column(s): {', '.join(missing)}")
        return problems
    dt = pd.to_datetime(df["date_time"], errors="coerce")
    if dt.isna().mean() > 0.05:
        problems.append("More than 5% of 'date_time' values could not be parsed as dates.")
    vol = pd.to_numeric(df["traffic_volume"], errors="coerce")
    if vol.isna().mean() > 0.05:
        problems.append("More than 5% of 'traffic_volume' values are not numeric.")
    if (vol < 0).any():
        problems.append("'traffic_volume' contains negative values.")
    if not problems and dt.notna().sum() < 500:
        problems.append("At least 500 valid rows are needed to train a model.")
    if not problems and dt.dropna().dt.normalize().nunique() < 60:
        problems.append("The data should cover at least 60 distinct days.")
    return problems


# --------------------------------------------------------------------------- cleaning
def clean_data(df: pd.DataFrame, return_report: bool = False):
    """Parse dates, remove duplicates / impossible values and tidy the schema."""
    problems = validate_dataframe(df)
    if problems:
        raise DatasetError(" ".join(problems))

    report = {"rows_in": int(len(df))}
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["traffic_volume"] = pd.to_numeric(df["traffic_volume"], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=["date_time", "traffic_volume"])
    report["dropped_unparseable"] = int(n0 - len(df))

    # duplicates: exact duplicate rows, then repeated timestamps (keep first record)
    n0 = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="date_time", keep="first")
    report["duplicates_removed"] = int(n0 - len(df))

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col != "weather_main" and col != "holiday" else None

    # temperature: Kelvin (as in the Metro dataset) or Celsius -> always store Celsius
    temp = pd.to_numeric(df["temp"], errors="coerce")
    if temp.notna().any() and temp.median() > 200:
        temp = temp - 273.15
        temp[temp < -60] = np.nan           # 0 K readings are sensor errors
    else:
        temp[(temp < -60) | (temp > 60)] = np.nan
    df["temp_c"] = temp

    rain = pd.to_numeric(df["rain_1h"], errors="coerce")
    report["outliers_nulled"] = int((rain > 100).sum() + (temp.isna() & df["temp"].notna()).sum())
    rain[(rain < 0) | (rain > 100)] = np.nan   # >100 mm in one hour is a data error
    df["rain_1h"] = rain
    snow = pd.to_numeric(df["snow_1h"], errors="coerce")
    snow[(snow < 0) | (snow > 100)] = np.nan
    df["snow_1h"] = snow
    clouds = pd.to_numeric(df["clouds_all"], errors="coerce")
    clouds[(clouds < 0) | (clouds > 100)] = np.nan
    df["clouds_all"] = clouds

    df["weather_main"] = df["weather_main"].fillna("Unknown").astype(str)

    # holiday: the real dataset names the holiday on the first record of the day only.
    # Propagate it to every record of that date and reduce it to a 0/1 flag.
    hol = df["holiday"].fillna("None").astype(str).str.strip()
    flag = ~hol.str.lower().isin(["none", "", "nan", "0", "false"])
    day = df["date_time"].dt.normalize()
    df["is_holiday"] = flag.groupby(day).transform("max").astype(int)

    df = df.sort_values("date_time").reset_index(drop=True)
    report["rows_out"] = int(len(df))
    report["supported_inputs"] = {
        "weather": bool((df["weather_main"] != "Unknown").any()),
        "temperature": bool(df["temp_c"].notna().any()),
        "holiday": bool(df["is_holiday"].any()),
        "precipitation": bool(df["rain_1h"].notna().any() or df["snow_1h"].notna().any()),
        "clouds": bool(df["clouds_all"].notna().any()),
    }
    if return_report:
        return df, report
    return df


# --------------------------------------------------------------------------- features
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model inputs from a cleaned frame (needs date_time, is_holiday, temp_c, ...)."""
    dt = pd.to_datetime(df["date_time"])
    X = pd.DataFrame(index=df.index)
    hour = dt.dt.hour
    month = dt.dt.month
    dow = dt.dt.dayofweek
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    X["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    X["is_weekend"] = (dow >= 5).astype(int)
    X["is_holiday"] = df["is_holiday"].astype(int).to_numpy()
    X["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
    X["rain_1h"] = pd.to_numeric(df["rain_1h"], errors="coerce")
    X["snow_1h"] = pd.to_numeric(df["snow_1h"], errors="coerce")
    X["clouds_all"] = pd.to_numeric(df["clouds_all"], errors="coerce")
    X["hour"] = hour.astype(str)
    X["day_of_week"] = dow.astype(str)
    # weekday/weekend x hour interaction: lets even linear models learn two different daily profiles
    X["weekday_hour"] = (X["is_weekend"].astype(str) + "_" + hour.astype(str))
    X["weather_main"] = df["weather_main"].fillna("Unknown").astype(str)
    return X[FEATURE_COLUMNS]


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def chronological_split(X: pd.DataFrame, y: pd.Series, test_size: float):
    """Oldest (1-test_size) share for training, newest share for testing (no shuffling)."""
    cut = int(len(X) * (1 - test_size))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def inputs_to_frame(date: str, hour: int, is_holiday: bool, weather_main: str, temp_c,
                    rain_1h=None, snow_1h=None, clouds_all=None) -> pd.DataFrame:
    """Turn one set of user inputs into the model's feature frame."""
    row = pd.DataFrame([{
        "date_time": pd.Timestamp(date) + pd.Timedelta(hours=int(hour)),
        "is_holiday": int(bool(is_holiday)),
        "temp_c": temp_c,
        "rain_1h": rain_1h,
        "snow_1h": snow_1h,
        "clouds_all": clouds_all,
        "weather_main": weather_main,
    }])
    return build_features(row)
