# config/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Configuration for Night Bus Service Reliability Prediction System
# ST5011CEM – Big Data Programming Project
# ─────────────────────────────────────────────────────────────────────────────
import os

# ── Spark ──────────────────────────────────────────────────────────────────
SPARK_CONFIG = {
    "app_name": "NightBusReliabilityPlatform",
    "master": "local[4]",           # 4 cores → satisfies ≥4 partition requirement
    "executor_memory": "2g",
    "driver_memory": "2g",
    "sql_shuffle_partitions": "8",
    "default_parallelism": "8",
    "log_level": "WARN",
}

# ── Database ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(BASE_DIR, "database", "night_bus.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"   # no hard-coded credentials

# ── Data ───────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "data")
SYNTHETIC_RECORDS = 120_000          # target ≥ 100 000 records
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_PARTITIONS = 8

# ── Metric thresholds (from brief) ─────────────────────────────────────────
SERVICE_RELIABILITY_THRESHOLD = 0.85   # ≥ 85 % on-time
HEADWAY_SD_MAX_RATIO = 0.20            # SD ≤ 20 % of scheduled headway
TRAVEL_TIME_CV_MAX = 0.15             # CV ≤ 15 %
COMPLETION_RATE_MIN = 0.95            # ≥ 95 %
ON_TIME_URBAN_MINUTES = 2             # ± 2 min urban tolerance

# ── ML ─────────────────────────────────────────────────────────────────────
LABEL_COL = "is_delayed"
FEATURES_COL = "features"
NUM_FOLDS = 5
