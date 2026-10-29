"""Central configuration for the AI-Based Traffic Prediction System."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Paths ---------------------------------------------------------------
DATA_PATH = os.path.join(BASE_DIR, "data", "traffic.csv")
DATASET_INFO_PATH = os.path.join(BASE_DIR, "data", "dataset_info.json")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
DB_PATH = os.path.join(BASE_DIR, "history.db")

# ---- Web server ----------------------------------------------------------
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 5000))
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
MAX_UPLOAD_MB = 25

# ---- ML settings ---------------------------------------------------------
TARGET = "traffic_volume"
TEST_SIZE = 0.20            # chronological split: last 20% of the timeline is the test set
CV_SPLITS = 5               # TimeSeriesSplit folds used for model selection (train data only)
RANDOM_STATE = 42

# ---- Congestion categories ----------------------------------------------
# Thresholds are the given percentiles of the *training* traffic volumes
# (vehicles/hour).  To use fixed numbers instead, set ABSOLUTE_THRESHOLDS to
# e.g. (2000, 4000, 5500) -> Low < 2000 <= Moderate < 4000 <= High < 5500 <= Severe.
THRESHOLD_PERCENTILES = (40, 70, 90)   # Low | Moderate | High | Severe
ABSOLUTE_THRESHOLDS = None

CATEGORIES = ["Low", "Moderate", "High", "Severe"]
CATEGORY_COLORS = {
    "Low": "#16a34a",        # green
    "Moderate": "#eab308",   # yellow
    "High": "#f97316",       # orange
    "Severe": "#dc2626",     # red
}

VOLUME_DISCLAIMER = (
    "Traffic volume (vehicles per hour) is not the same as road congestion. "
    "Congestion categories here are relative bands derived from this dataset's own volume "
    "distribution. Reliable congestion estimates need road capacity or labelled congestion data."
)
