"""Generate a clearly labelled SYNTHETIC sample dataset.

The real "Metro Interstate Traffic Volume" dataset (UCI ML Repository, id 492) could not be
downloaded in every environment, so this script produces a stand-in with the *same schema*:

    holiday, temp (Kelvin), rain_1h, snow_1h, clouds_all, weather_main,
    weather_description, date_time, traffic_volume

The numbers are simulated (rush-hour peaks, weekend dips, holiday dips, weather effects,
seasonality, noise) - they are NOT real measurements.  A sidecar file ``dataset_info.json``
records that the file is synthetic (with a SHA-256 of the CSV so that replacing the CSV with
real data is detected automatically).

Replace ``data/traffic.csv`` with the real dataset and re-run ``python src/train_model.py``.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

WEEKDAY = np.array([400, 300, 250, 300, 800, 2800, 5000, 6000, 5300, 4700, 4500, 4800,
                    5100, 5000, 5300, 5900, 6400, 5900, 4500, 3400, 2800, 2300, 1600, 900], float)
WEEKEND = np.array([1100, 700, 500, 400, 450, 800, 1400, 2400, 3400, 4000, 4400, 4600,
                    4700, 4700, 4700, 4600, 4400, 4200, 3700, 3300, 2900, 2500, 2000, 1500], float)

WEATHER_DESC = {
    "Clear": ["sky is clear"],
    "Clouds": ["scattered clouds", "broken clouds", "overcast clouds", "few clouds"],
    "Rain": ["light rain", "moderate rain", "heavy intensity rain"],
    "Drizzle": ["light intensity drizzle", "drizzle"],
    "Snow": ["light snow", "snow", "heavy snow"],
    "Mist": ["mist"],
    "Fog": ["fog"],
    "Haze": ["haze"],
    "Thunderstorm": ["thunderstorm", "proximity thunderstorm"],
}
WEATHER_FACTOR = {"Clear": 1.0, "Clouds": 0.99, "Rain": 0.93, "Drizzle": 0.95, "Snow": 0.86,
                  "Mist": 0.96, "Fog": 0.95, "Haze": 0.98, "Thunderstorm": 0.84}


def generate(start="2015-01-01", end="2018-09-30 23:00", seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h")
    df = pd.DataFrame({"date_time": idx})
    doy = df.date_time.dt.dayofyear.to_numpy()
    hour = df.date_time.dt.hour.to_numpy()
    dow = df.date_time.dt.dayofweek.to_numpy()

    # temperature (deg C): seasonal + diurnal + daily noise
    day_noise = pd.Series(rng.normal(0, 4, len(idx))).groupby(df.date_time.dt.date.values).transform("mean").to_numpy()
    temp_c = 7 - 17 * np.cos(2 * np.pi * (doy - 20) / 365.25) + 5 * np.sin(2 * np.pi * (hour - 9) / 24) + day_noise * 2

    # weather: one state per day (with a little hourly jitter), constrained by temperature
    days = df.date_time.dt.normalize()
    day_codes, uniq = pd.factorize(days)
    day_temp = pd.Series(temp_c).groupby(day_codes).mean().to_numpy()
    wm_day = []
    for t in day_temp:
        if t < 0:
            p = {"Clear": .28, "Clouds": .30, "Snow": .27, "Mist": .06, "Fog": .04, "Haze": .05}
        else:
            p = {"Clear": .30, "Clouds": .33, "Rain": .14, "Drizzle": .05, "Mist": .06,
                 "Fog": .03, "Haze": .06, "Thunderstorm": .03}
        keys = list(p)
        pr = np.array(list(p.values())); pr = pr / pr.sum()
        wm_day.append(rng.choice(keys, p=pr))
    weather_main = np.array(wm_day)[day_codes]
    weather_desc = np.array([rng.choice(WEATHER_DESC[w]) for w in weather_main])

    rain = np.where(np.isin(weather_main, ["Rain", "Drizzle", "Thunderstorm"]),
                    rng.gamma(1.5, 0.8, len(idx)), 0.0).round(2)
    snow = np.where(weather_main == "Snow", rng.gamma(1.2, 0.05, len(idx)), 0.0).round(2)
    clouds = np.select(
        [weather_main == "Clear", weather_main == "Clouds"],
        [rng.integers(0, 20, len(idx)), rng.integers(40, 100, len(idx))],
        default=rng.integers(75, 100, len(idx)))

    # traffic volume
    base = np.where(dow >= 5, WEEKEND[hour], WEEKDAY[hour])
    base = base * np.where(dow == 6, 0.93, 1.0)
    season = 1.0 + 0.06 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    growth = 1.0 + 0.02 * ((df.date_time.dt.year.to_numpy() - 2015) + doy / 365.25)
    hol_dates = USFederalHolidayCalendar().holidays(start=idx.min(), end=idx.max())
    holiday_names = USFederalHolidayCalendar().holidays(start=idx.min(), end=idx.max(), return_name=True)
    is_hol = days.isin(hol_dates).to_numpy()
    wf = np.array([WEATHER_FACTOR[w] for w in weather_main])
    vol = base * season * growth * np.where(is_hol, 0.62, 1.0) * wf
    vol = vol * np.exp(rng.normal(0, 0.07, len(idx)))
    vol = np.clip(vol, 0, None).round().astype(int)

    # holiday label only on the first record of the day (mirrors the real dataset)
    hol_label = np.full(len(idx), "None", dtype=object)
    name_map = {d: n for d, n in holiday_names.items()}
    first_of_day = (hour == 0) & is_hol
    hol_label[first_of_day] = [name_map[d] for d in days[first_of_day]]

    out = pd.DataFrame({
        "holiday": hol_label,
        "temp": (temp_c + 273.15).round(2),
        "rain_1h": rain,
        "snow_1h": snow,
        "clouds_all": clouds.astype(int),
        "weather_main": weather_main,
        "weather_description": weather_desc,
        "date_time": df.date_time.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "traffic_volume": vol,
    })
    # sprinkle a few realistic data problems so the cleaning step has work to do
    bad = rng.choice(len(out), 12, replace=False)
    out.loc[bad[:6], "temp"] = 0.0                 # impossible 0 K readings (exist in the real data)
    out.loc[bad[6:9], "rain_1h"] = 9831.3          # absurd rainfall (exists in the real data)
    out = pd.concat([out, out.sample(40, random_state=1)]).sort_values("date_time").reset_index(drop=True)  # duplicates
    return out


def write(path=config.DATA_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = generate()
    df.to_csv(path, index=False)
    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    info = {
        "synthetic": True,
        "sha256": sha,
        "description": "SYNTHETIC sample data with the Metro Interstate Traffic Volume schema. "
                       "Not real measurements. Replace data/traffic.csv with the real dataset for real results.",
        "rows": int(len(df)),
    }
    with open(config.DATASET_INFO_PATH, "w") as fh:
        json.dump(info, fh, indent=2)
    return path


if __name__ == "__main__":
    p = write()
    print(f"Wrote synthetic sample dataset to {p}")
