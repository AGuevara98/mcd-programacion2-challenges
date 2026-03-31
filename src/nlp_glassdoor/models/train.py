from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.common.io_utils import load_csv, save_json


def train_baseline(df: pd.DataFrame, random_state: int = 42):
    if "clean_text" not in df.columns:
        raise ValueError("Expected 'clean_text' column.")
    if "label" not in df.columns:
        raise ValueError("Expected 'label' column.")

    X = df["clean_text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=5000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))

    artifacts = {
        "metrics": metrics,
        "classification_report": report,
        "labels": sorted(y.unique()),
        "confusion_matrix": cm.tolist(),
    }

    return pipeline, artifacts, y_test, y_pred


def save_confusion_matrix(y_true, y_pred, labels, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        ax=ax,
        colorbar=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(input_path: str, metrics_output: str, cm_output: str) -> None:
    df = load_csv(input_path)
    print(f"Input shape: {df.shape}")

    _, artifacts, y_test, y_pred = train_baseline(df)

    save_json(artifacts, metrics_output)
    save_confusion_matrix(y_test, y_pred, artifacts["labels"], cm_output)

    print("Metrics:")
    print(json.dumps(artifacts["metrics"], indent=2, ensure_ascii=False))
    print(f"Saved metrics to {metrics_output}")
    print(f"Saved confusion matrix to {cm_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline TF-IDF + Logistic Regression model.")
    parser.add_argument("--input_path", required=True, help="Path to cleaned dataset CSV.")
    parser.add_argument("--metrics_output", required=True, help="Path to save metrics JSON.")
    parser.add_argument("--cm_output", required=True, help="Path to save confusion matrix PNG.")
    args = parser.parse_args()

    main(
        input_path=args.input_path,
        metrics_output=args.metrics_output,
        cm_output=args.cm_output,
    )