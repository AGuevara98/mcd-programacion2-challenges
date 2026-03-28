from sklearn.linear_model import LogisticRegression

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


def run_cancer_pipeline(data_path: str) -> dict:
    validate_file_exists(data_path)

    df = load_csv(data_path)
    validate_dataframe_not_empty(df)

    df = basic_cleaning(df)

    # Adjust this target name depending on final dataset
    target_column = "diagnosis"
    validate_target_column(df, target_column)

    summary = summarize_dataframe(df)

    X, y = split_features_target(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    X_train, X_test, scaler = scale_numeric_features(X_train, X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    metrics = evaluate_classifier(model, X_test, y_test)
    cv_metrics = cross_validate_model(model, X_train, y_train)
    metrics.update(cv_metrics)

    cm_path = str(PLOTS_DIR / "cancer_confusion_matrix.png")
    roc_path = str(PLOTS_DIR / "cancer_roc_curve.png")
    metrics_path = str(METRICS_DIR / "cancer_metrics.json")

    plot_confusion_matrix(model, X_test, y_test, cm_path)
    plot_roc_curve(model, X_test, y_test, roc_path)
    save_json({"summary": summary, "metrics": metrics}, metrics_path)

    log_params({
        "challenge": "cancer",
        "model": "LogisticRegression",
        "target_column": target_column,
    })
    log_metrics(metrics)
    log_artifact(cm_path)
    if "roc_auc" in metrics:
        log_artifact(roc_path)
    log_artifact(metrics_path)
    log_model(model, X_train, run_name="cancer_model")

    return {"summary": summary, "metrics": metrics}