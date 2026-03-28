from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODELS_DIR = OUTPUTS_DIR / "models"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "mcd_udg_challenges"

SUPPORTED_CHALLENGES = {"cancer", "nlp", "thesis"}

DEFAULT_RANDOM_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5