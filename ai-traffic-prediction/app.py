"""Flask application: AI-Based Traffic Prediction System.

Run:  python app.py      (after `python src/train_model.py`)
"""
import os
import re
import shutil
import threading
import uuid
from datetime import date, datetime, timedelta

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

import config
from src import analytics as an
from src import database as db
from src.predict import InputValidationError, ModelNotReadyError, TrafficPredictor
from src.preprocess import DatasetError, load_raw, validate_dataframe

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False

predictor = TrafficPredictor()
_train_lock = threading.Lock()
db.init_db()
os.makedirs(config.UPLOAD_DIR, exist_ok=True)

UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")


# ----------------------------------------------------------------------------- helpers
def page_context(**extra):
    ctx = {"model_ready": predictor.ready, "disclaimer": config.VOLUME_DISCLAIMER, "meta": None,
           "dataset_label": None, "synthetic": False, "categories": config.CATEGORIES,
           "colors": config.CATEGORY_COLORS}
    if ctx["model_ready"]:
        meta = predictor.metadata
        ctx.update(meta=meta, dataset_label=meta["dataset"]["label"], synthetic=meta["dataset"]["synthetic"])
    ctx.update(extra)
    return ctx


def api_error(message, status=400, **extra):
    return jsonify({"error": message, **extra}), status


def meta_or_none():
    return predictor.metadata if predictor.ready else None


def filters_from_request():
    keys = ("start", "end", "hour_from", "hour_to", "day_type", "weather")
    p = {k: (request.args.get(k) or "").strip() for k in keys}
    for k in ("start", "end"):
        if p[k]:
            try:
                pd.Timestamp(p[k])
            except Exception:
                raise ValueError(f"'{k}' is not a valid date.")
    for k in ("hour_from", "hour_to"):
        if p[k]:
            if not p[k].isdigit() or not 0 <= int(p[k]) <= 23:
                raise ValueError("Hours must be whole numbers between 0 and 23.")
    if p["hour_from"] and p["hour_to"] and int(p["hour_from"]) > int(p["hour_to"]):
        raise ValueError("'From hour' must not be later than 'To hour'.")
    if p["day_type"] not in ("", "all", "weekday", "weekend", "holiday"):
        raise ValueError("Unknown day type.")
    return p


def next_weekday(weekday: int) -> str:
    d = date.today() + timedelta(days=1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d.isoformat()


# ----------------------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html", active="home", **page_context())


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html", active="dashboard", **page_context())


@app.route("/predict")
def prediction_page():
    ctx = page_context()
    return render_template("prediction.html", active="predict", today=date.today().isoformat(),
                           day_names=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                           **ctx)


@app.route("/analytics")
def analytics_page():
    opts = None
    try:
        opts = an.filter_options()
    except DatasetError:
        pass
    return render_template("analytics.html", active="analytics", opts=opts, **page_context())


@app.route("/performance")
def performance_page():
    return render_template("performance.html", active="performance", max_mb=config.MAX_UPLOAD_MB, **page_context())


@app.route("/history")
def history_page():
    return render_template("history.html", active="history", **page_context())


# ----------------------------------------------------------------------------- APIs
@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return api_error("Send the inputs as a JSON object.")
    try:
        result = predictor.predict(payload)
    except InputValidationError as exc:
        return api_error("Please correct the highlighted fields.", 422, fields=exc.errors)
    except ModelNotReadyError as exc:
        return api_error(str(exc), 503)
    result["history_id"] = db.add_prediction(result)
    return jsonify(result)


@app.route("/api/sample-scenarios")
def api_samples():
    if not predictor.ready:
        return api_error("Model not trained yet.", 503)
    meta = predictor.metadata
    w = meta["weather_options"]
    pick = lambda *names: next((n for n in names if n in w), w[0] if w else "Clear")
    mm = {int(k): v for k, v in meta["median_temp_by_month"].items()}
    t = lambda d: round(mm.get(pd.Timestamp(d).month, 10.0), 1)
    tue, sun, mon, fri = next_weekday(1), next_weekday(6), next_weekday(0), next_weekday(4)
    return jsonify([
        {"name": "Weekday morning rush", "date": tue, "hour": 7, "holiday": False, "weather_main": pick("Clear", "Clouds"), "temperature_c": t(tue)},
        {"name": "Friday evening rush (rain)", "date": fri, "hour": 16, "holiday": False, "weather_main": pick("Rain", "Drizzle", "Clouds"), "temperature_c": t(fri)},
        {"name": "Sunday late night", "date": sun, "hour": 23, "holiday": False, "weather_main": pick("Clear", "Clouds"), "temperature_c": t(sun)},
        {"name": "Snowy Monday morning", "date": mon, "hour": 8, "holiday": False, "weather_main": pick("Snow", "Clouds"), "temperature_c": -5.0},
        {"name": "Public holiday midday", "date": tue, "hour": 12, "holiday": True, "weather_main": pick("Clear", "Clouds"), "temperature_c": t(tue)},
    ])


@app.route("/api/dashboard")
def api_dashboard():
    try:
        data = an.dashboard(meta_or_none())
    except DatasetError as exc:
        return api_error(str(exc), 503)
    data["next_hour"] = None
    if predictor.ready:
        t = datetime.now() + timedelta(hours=1)
        try:
            r = predictor.predict_at(t.strftime("%Y-%m-%d"), t.hour)
            data["next_hour"] = {"label": t.strftime("%a %d %b, %H:00"), "volume": r["predicted_volume"],
                                 "category": r["category"], "color": r["color"], "intensity_pct": r["intensity_pct"],
                                 "assumption": f"Assumes {r['inputs']['weather_main'].lower()} weather, a typical "
                                               f"{r['inputs']['temperature_c']:.0f} °C for the month and no public holiday."}
        except Exception:
            pass
    data["model"] = {"algorithm": predictor.metadata["algorithm"], "r2": predictor.metadata["metrics"]["r2"]} if predictor.ready else None
    return jsonify(data)


@app.route("/api/analytics")
def api_analytics():
    try:
        return jsonify(an.analytics(filters_from_request(), meta_or_none()))
    except ValueError as exc:
        return api_error(str(exc), 422)
    except DatasetError as exc:
        return api_error(str(exc), 503)


@app.route("/api/report.csv")
def api_report():
    try:
        text = an.report_csv(filters_from_request(), meta_or_none())
    except ValueError as exc:
        return api_error(str(exc), 422)
    except DatasetError as exc:
        return api_error(str(exc), 503)
    return Response(text, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=traffic_analysis_report.csv"})


@app.route("/api/performance")
def api_performance():
    if not predictor.ready:
        return api_error("The trained model was not found. Run `python src/train_model.py` first.", 503)
    return jsonify(predictor.metadata)


@app.route("/api/history")
def api_history():
    rows = db.query(request.args.get("q", "").strip(), request.args.get("category", "").strip(),
                    request.args.get("start", "").strip(), request.args.get("end", "").strip())
    return jsonify({"rows": rows, "count": len(rows)})


@app.route("/api/history/export.csv")
def api_history_export():
    rows = db.query(request.args.get("q", "").strip(), request.args.get("category", "").strip(),
                    request.args.get("start", "").strip(), request.args.get("end", "").strip(), limit=100000)
    return Response(db.to_csv(rows), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=prediction_history.csv"})


@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    return jsonify({"deleted": db.clear()})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Validate an uploaded CSV. The file is only ever parsed as data (pandas) - never executed."""
    f = request.files.get("file")
    if f is None or not f.filename:
        return api_error("Choose a CSV file to upload.")
    if not f.filename.lower().endswith(".csv"):
        return api_error("Only .csv files are accepted.")
    upload_id = uuid.uuid4().hex                      # server-chosen name: user filename is never used on disk
    path = os.path.join(config.UPLOAD_DIR, f"{upload_id}.csv")
    f.save(path)
    try:
        df = load_raw(path)
        problems = validate_dataframe(df)
    except DatasetError as exc:
        os.remove(path)
        return api_error(str(exc), 422)
    except Exception:
        os.remove(path)
        return api_error("That file could not be read as a CSV.", 422)
    if problems:
        os.remove(path)
        return api_error("This CSV is not compatible.", 422, problems=problems)
    dt = pd.to_datetime(df["date_time"], errors="coerce")
    optional = [c for c in ("holiday", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main") if c in df.columns]
    return jsonify({"upload_id": upload_id, "rows": int(len(df)), "date_min": str(dt.min().date()),
                    "date_max": str(dt.max().date()), "optional_columns": optional,
                    "message": "File is valid. Click 'Retrain with this file' to build a new model."})


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    from src.train_model import train
    body = request.get_json(silent=True) or {}
    upload_id = body.get("upload_id")
    if upload_id is not None and not UPLOAD_ID.match(str(upload_id)):
        return api_error("Invalid upload reference.")
    if not _train_lock.acquire(blocking=False):
        return api_error("A training run is already in progress.", 409)
    backup = config.DATA_PATH + ".bak"
    swapped = False
    try:
        if upload_id:
            src = os.path.join(config.UPLOAD_DIR, f"{upload_id}.csv")
            if not os.path.exists(src):
                return api_error("Uploaded file not found. Upload it again.", 404)
            if os.path.exists(config.DATA_PATH):
                shutil.copy2(config.DATA_PATH, backup)
            shutil.copy2(src, config.DATA_PATH)
            swapped = True
        meta = train(config.DATA_PATH, verbose=False)
        return jsonify({"ok": True, "algorithm": meta["algorithm"], "metrics": meta["metrics"]})
    except DatasetError as exc:
        if swapped and os.path.exists(backup):
            shutil.copy2(backup, config.DATA_PATH)
        return api_error(str(exc), 422)
    except Exception as exc:
        if swapped and os.path.exists(backup):
            shutil.copy2(backup, config.DATA_PATH)
        return api_error(f"Training failed: {exc}", 500)
    finally:
        _train_lock.release()


# ----------------------------------------------------------------------------- errors
@app.errorhandler(413)
def too_large(_):
    return api_error(f"File too large (limit {config.MAX_UPLOAD_MB} MB).", 413)


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return api_error("Not found.", 404)
    return render_template("error.html", active="", code=404, message="That page does not exist.", **page_context()), 404


@app.errorhandler(500)
def server_error(_):
    if request.path.startswith("/api/"):
        return api_error("Unexpected server error.", 500)
    return render_template("error.html", active="", code=500, message="Something went wrong on the server.", **page_context()), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_ready": predictor.ready})


if __name__ == "__main__":
    if not predictor.ready:
        print("WARNING: no trained model found - run `python src/train_model.py` first.")
    print(f"\n  AI Traffic Prediction System running at  http://{config.HOST}:{config.PORT}\n")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
