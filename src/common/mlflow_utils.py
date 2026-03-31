import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.common.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_params(params: dict) -> None:
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: dict) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)


def log_artifact(file_path: str) -> None:
    mlflow.log_artifact(file_path)


def log_model(model, X_sample, run_name: str = "model") -> None:
    signature = infer_signature(X_sample, model.predict(X_sample))
    mlflow.sklearn.log_model(model, name=run_name, signature=signature)