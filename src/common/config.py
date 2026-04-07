from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METRICS_DIR = DATA_DIR / "metrics"

PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"

DEFAULT_RANDOM_SEED = 42

for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, METRICS_DIR, PLOTS_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)