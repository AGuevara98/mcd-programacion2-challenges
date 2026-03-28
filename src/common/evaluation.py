from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score


def evaluate_classifier(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="binary", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="binary", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))

    return metrics


def cross_validate_model(model, X_train, y_train, cv: int = 5) -> dict:
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    return {
        "cv_f1_mean": float(scores.mean()),
        "cv_f1_std": float(scores.std()),
    }