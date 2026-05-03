from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


def _import_lightgbm():
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        raise ImportError(
            "lightgbm is required for thesis modeling. Install from requirements.txt"
        ) from exc
    return LGBMClassifier


def split_training_data(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    cols = list(feature_cols)
    for c in cols + [target_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[target_col].astype(int)

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)


def train_models(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    LGBMClassifier = _import_lightgbm()

    models = {
        "lightgbm": LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def cross_validate_f1(model, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    return {
        "cv_f1_mean": float(scores.mean()),
        "cv_f1_std": float(scores.std()),
    }


def _shap_values_for_binary(model, X_sample: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_sample)
    if isinstance(values, list):
        if len(values) == 1:
            return values[0]
        return values[1]
    if hasattr(values, "values"):
        arr = values.values
        if arr.ndim == 3:
            return arr[:, :, 1]
        return arr
    return values


def create_shap_summary_plot(
    model,
    X_train: pd.DataFrame,
    output_path: str,
    max_rows: int = 1000,
) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sample = X_train.sample(min(max_rows, len(X_train)), random_state=42)

    shap_values = _shap_values_for_binary(model, sample)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    return output_path


def create_roc_plot(model, X_test: pd.DataFrame, y_test: pd.Series, output_path: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def feature_importance_df(model, X_train: pd.DataFrame) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    imp = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": np.asarray(model.feature_importances_, dtype=float),
        }
    ).sort_values("importance", ascending=False)
    return imp
