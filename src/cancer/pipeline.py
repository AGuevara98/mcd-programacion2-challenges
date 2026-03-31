import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.common.io_utils import load_csv, save_json
from src.common.validation import (
    validate_file_exists,
    validate_dataframe_not_empty,
    validate_target_column,
)
from src.common.preprocessing import (
    basic_cleaning,
    summarize_dataframe,
    split_features_target,
    train_test_split_data,
    scale_numeric_features,
)
from src.common.evaluation import evaluate_classifier, cross_validate_model
from src.common.plots import plot_confusion_matrix, plot_roc_curve
from src.common.config import METRICS_DIR, PLOTS_DIR
from src.common.mlflow_utils import log_params, log_metrics, log_artifact, log_model

import mlflow


def clean_cancer_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "diagnosis" not in df.columns:
        raise ValueError("The dataset must contain a 'diagnosis' column.")

    if df["diagnosis"].dtype == object:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    if df["diagnosis"].isna().any():
        raise ValueError(
            "The 'diagnosis' column contains unmapped or null values after encoding."
        )

    return df


def run_single_model(
    model,
    model_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    X_full,
    df,
    target_column: str,
    summary: dict,
) -> dict:
    model.fit(X_train, y_train)

    metrics = evaluate_classifier(model, X_test, y_test)
    cv_metrics = cross_validate_model(model, X_train, y_train)
    metrics.update(cv_metrics)

    cm_path = str(PLOTS_DIR / f"{model_name}_confusion_matrix.png")
    roc_path = str(PLOTS_DIR / f"{model_name}_roc_curve.png")
    metrics_path = str(METRICS_DIR / f"{model_name}_metrics.json")

    plot_confusion_matrix(model, X_test, y_test, cm_path)
    plot_roc_curve(model, X_test, y_test, roc_path)

    save_json(
        {
            "model": model_name,
            "summary": summary,
            "metrics": metrics,
        },
        metrics_path,
    )

    log_params(
        {
            "challenge": "cancer",
            "model": model_name,
            "target_column": target_column,
            "n_features": len(X_full.columns),
            "n_rows": len(df),
        }
    )
    log_metrics(metrics)
    log_artifact(cm_path)

    if "roc_auc" in metrics:
        log_artifact(roc_path)

    log_artifact(metrics_path)
    log_model(model, X_train, run_name=f"{model_name}_model")

    return metrics


def run_cancer_pipeline(data_path: str) -> dict:
    validate_file_exists(data_path)

    df = load_csv(data_path)
    validate_dataframe_not_empty(df)

    df = clean_cancer_dataset(df)
    df = basic_cleaning(df)

    target_column = "diagnosis"
    validate_target_column(df, target_column)

    summary = summarize_dataframe(df)

    X, y = split_features_target(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    X_train, X_test, scaler = scale_numeric_features(X_train, X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    all_results = {}

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"cancer_{model_name}", nested=True):
            all_results[model_name] = run_single_model(
                model=model,
                model_name=model_name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                X_full=X,
                df=df,
                target_column=target_column,
                summary=summary,
            )

    best_model_name = max(
        all_results,
        key=lambda name: all_results[name].get("f1_score", float("-inf"))
    )

    comparison_path = str(METRICS_DIR / "cancer_model_comparison.json")
    save_json(
        {
            "summary": summary,
            "results": all_results,
            "best_model_by_f1": best_model_name,
        },
        comparison_path,
    )

    return {
        "summary": summary,
        "results": all_results,
        "best_model_by_f1": best_model_name,
    }