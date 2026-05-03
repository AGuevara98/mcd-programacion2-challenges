from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score


def evaluate_classifier(
    model: Any, X: pd.DataFrame, y: pd.Series
) -> Dict[str, Any]:
    y_pred = model.predict(X)
    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }
    if hasattr(model, "predict_proba"):
        metrics["roc_auc"] = roc_auc_score(y, model.predict_proba(X)[:, 1])
    return metrics


def cross_validate_model(
    model: Any, X: pd.DataFrame, y: pd.Series
) -> Dict[str, float]:
    scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    return {"cv_f1_mean": float(scores.mean()), "cv_f1_std": float(scores.std())}