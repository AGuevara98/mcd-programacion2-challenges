import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.common.io_utils import load_csv, save_json
from src.common.validation import validate_file_exists
from src.common.preprocessing import (
    split_features_target,
    train_test_split_data,
    scale_numeric_features,
)
from src.common.evaluation import evaluate_classifier, cross_validate_model
from src.common.config import METRICS_DIR

def clean_cancer_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df["diagnosis"].dtype == object:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df


def run_cancer_pipeline(path: str) -> dict:
    validate_file_exists(path)
    df = load_csv(path)
    df = clean_cancer_dataset(df)

    X, y = split_features_target(df, "diagnosis")
    X_train, X_test, y_train, y_test = train_test_split_data(X,y)
    X_train, X_test, _ = scale_numeric_features(X_train,X_test)

    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier()
    }

    results = {}

    for name, model in models.items():
        with mlflow.start_run(nested=True, run_name=f"cancer_{name}"):
            model.fit(X_train, y_train)
            metrics = evaluate_classifier(model, X_test, y_test)
            metrics.update(cross_validate_model(model, X_train, y_train))
            results[name] = metrics

            mlflow.log_params({"model": name})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.sklearn.log_model(model, artifact_path=f"cancer_{name}")

    save_json(results, METRICS_DIR/"cancer_metrics.json")
    return results