# AI-Based Traffic Prediction System

A Flask + scikit-learn web app that forecasts hourly traffic **volume**, labels it Low / Moderate / High / Severe, and shows historical analytics and model-evaluation results in a light-theme dashboard. Built for a college-level AI & Data Science demonstration.

> **Forecasts and historical data only - there is no live traffic feed.** Traffic *volume* (vehicles/hour) is not the same as *congestion*; the categories are relative bands from the dataset's own volume distribution.

## ⚠️ About the dataset (read this first)

The app is designed for the **Metro Interstate Traffic Volume** dataset (UCI Machine Learning Repository, id 492). It could not be downloaded in the environment this project was built in, so `data/traffic.csv` currently contains **synthetic sample data** that follows the same schema (`holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, weather_description, date_time, traffic_volume`). It is simulated - **not real measurements** - and the UI says so everywhere (header badge, dashboard, performance page).

**To use the real data:**
1. Download `Metro_Interstate_Traffic_Volume.csv` from <https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume>.
2. Save it as `data/traffic.csv` (replace the sample file).
3. Run `python src/train_model.py`. The app detects (by file hash) that the data is no longer the sample, drops the "synthetic" labels, and all metrics are recomputed.

Any CSV with at least `date_time` and `traffic_volume` also works (upload it on the *Model performance* page); weather/holiday inputs appear only if those columns exist.

## Quick start

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/train_model.py         # trains, compares models, saves models/traffic_model.joblib (~1 min)
python app.py                     # prints:  http://127.0.0.1:5000
python -m pytest -q               # optional: 19 tests
```

If `data/traffic.csv` is missing, `train_model.py` generates the labelled synthetic sample automatically. Change host/port with `HOST` / `PORT` environment variables.

The pages load Bootstrap, Bootstrap Icons, Chart.js and the Inter font from CDNs (pinned versions), so an internet connection is needed in the browser. For fully offline use, download those files into `static/vendor/` and point the `<link>`/`<script>` tags in `templates/base.html` at them.

## Pages

| Page | What it does |
|---|---|
| **Home** | Introduction, feature cards, how-it-works, illustration |
| **Dashboard** | Latest *recorded* (historical) value, next-hour *forecast* + congestion level, average volume, peak hours; line, bar, doughnut, historical-vs-predicted, daily and weekly charts |
| **Predict traffic** | Date, hour, day of week (auto from date), holiday, weather, temperature, optional rain/snow/clouds → predicted volume, congestion category, intensity meter, recommendation, 24-hour forecast chart. Loading state + client and server validation. "Sample mode" buttons fill realistic scenarios (they still go through the model). |
| **Analytics** | Filters for date range, hours, weekday/weekend/holiday, weather - every chart and KPI updates: hourly (with percentile band), weekday vs weekend, daily/weekly trend, day of week, monthly, seasonality, weather vs traffic, congestion mix. CSV report download. |
| **Model performance** | Selected algorithm, dataset size, split, MAE/RMSE/R², algorithm comparison table, actual-vs-predicted (week / daily / scatter), feature importance, thresholds, upload-and-retrain |
| **History** | Every prediction saved in SQLite (`history.db`); search, category/date filters, CSV export, clear |

## Machine-learning workflow (`src/`)

| File | Role |
|---|---|
| `generate_sample_data.py` | Labelled synthetic dataset in the Metro schema (only used when no CSV exists) |
| `preprocess.py` | Validation, cleaning, feature engineering, sklearn preprocessing, chronological split |
| `train_model.py` | Compare models, select, evaluate, save |
| `evaluate.py` | MAE/RMSE/R², time-series CV, permutation importance, chart series |
| `predict.py` | Input validation, loading the saved pipeline, prediction, congestion class, recommendation |
| `analytics.py` | Filterable historical aggregations for the dashboard/analytics pages |
| `database.py` | SQLite history |

1. **Cleaning** - parse dates; drop duplicate rows and repeated timestamps; blank impossible values (0 K temperatures, rain > 100 mm/h); Kelvin → °C; holiday names → a 0/1 flag applied to *every* hour of that date.
2. **Features** - hour (cyclical sin/cos + categorical), weekday, weekday×hour interaction, month (cyclical), weekend, holiday, temperature, rain, snow, cloud cover, weather type. The same `build_features()` runs in training and prediction.
3. **No leakage** - imputation medians, scaling and one-hot categories live inside the sklearn pipeline and are fitted on the **training** split only; no target-derived features; the congestion thresholds come from training volumes only.
4. **Split** - chronological: oldest 80 % train, newest 20 % test (never shuffled).
5. **Model selection** - Linear Regression, Random Forest and (Hist) Gradient Boosting are compared with 5-fold expanding-window time-series cross-validation **on the training data**. The lowest mean CV RMSE wins; the test set is not used for selection.
6. **Saved** with Joblib (`models/traffic_model.joblib`: pipeline + thresholds) plus `models/model_metadata.json` (all metrics/chart data the UI shows). The saved model is trained on the training split only, so the reported test metrics describe exactly the model that serves predictions.

### Results from the run in this project (synthetic sample data)

Training 40,300 rows (2013-01 → 2017-08), test 10,076 rows (2017-08 → 2018-09):

| Model | CV RMSE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Linear Regression | 450.5 | 336.7 | 466.7 | 0.952 |
| Random Forest | 360.6 | 274.7 | 376.2 | 0.969 |
| **Gradient Boosting (selected)** | 349.5 | 264.5 | 361.3 | 0.971 |

Permutation importance of the selected model: hour of day 67 %, day of week 23 %, holiday 6.5 %, weather and season ≈ 3 % combined.

> These numbers show the model recovers the *simulated* patterns. The synthetic weather effect is small, so weather importance is low here; with the real dataset expect different figures. Do not quote them as real-world accuracy.

Example predictions (real output of `src/predict.py`, dataset thresholds: Low < 2,674 ≤ Moderate < 4,767 ≤ High < 6,048 ≤ Severe veh/h):

| Scenario | Predicted volume (veh/h) | Category |
|---|---|---|
| Tuesday 08:00, clear, 8 °C | 5,844 | High |
| Tuesday 17:00, rain, 6 °C | 6,102 | Severe |
| Sunday 03:00, clear, 4 °C | 555 | Low |
| Monday 08:00, snow, −5 °C | 4,379 | Moderate |
| Tuesday 12:00, public holiday | 3,785 | Moderate |

(Your numbers will differ slightly if you retrain.)

## Congestion categories

Defined in `config.py`: `THRESHOLD_PERCENTILES = (40, 70, 90)` of training volumes, or set `ABSOLUTE_THRESHOLDS = (2000, 4000, 5500)` for fixed values. Colours: green Low, yellow Moderate, orange High, red Severe. The intensity meter compares the forecast with the dataset's 99th-percentile volume.

## Honest limitations

- **No location input.** The dataset has a single road segment, so the location field is shown disabled with an explanation rather than faking per-road predictions.
- Optional weather details left blank use simple defaults based on the weather type (shown in the result as "Defaults used"). Day of week is derived from the date; a mismatching value sent to the API is rejected.
- The model has no year/trend term, so dates after the training period get the *typical* pattern for that hour/weekday/season/weather, not growth or one-off events (roadworks, accidents, special events).
- Holiday status is user-supplied (no holiday calendar is bundled).
- Retraining runs inside the web request (about a minute); fine for a demo, use a job queue in production.

## Security notes

Uploads: `.csv` only, size-limited (25 MB), stored under a server-generated name, parsed only as data with pandas and never executed; the retrain endpoint accepts only a 32-hex upload id. SQL uses parameterised queries. Flask runs without debug mode by default (`FLASK_DEBUG=1` to enable).

## Project structure

```text
ai-traffic-prediction/
├── app.py · config.py · requirements.txt · README.md
├── data/            traffic.csv, dataset_info.json (marks the sample as synthetic)
├── models/          traffic_model.joblib, model_metadata.json
├── src/             preprocess.py, train_model.py, evaluate.py, predict.py, analytics.py, database.py, generate_sample_data.py
├── templates/       base, index, dashboard, prediction, analytics, performance, history, error
├── static/          css/style.css, js/*.js, images/
├── tests/           test_prediction.py
├── uploads/         validated uploads (runtime)
└── _previous_build_backup/   files from an earlier partial build, kept for reference
```

## Troubleshooting

- **"No trained model found" banner** → run `python src/train_model.py`.
- **Charts empty / unstyled** → the browser could not reach the CDNs; connect to the internet or vendor the libraries locally.
- **Port 5000 busy (macOS AirPlay)** → `PORT=5001 python app.py`.
