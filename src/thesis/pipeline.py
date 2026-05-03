from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import mlflow
import geopandas as gpd
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.common.config import METRICS_DIR, PLOTS_DIR
from src.common.evaluation import evaluate_classifier
from src.common.io_utils import save_dataframe_csv, save_json
from src.common.mlflow_utils import log_artifact, log_metrics, log_model, log_params
from src.common.validation import validate_target_column
from src.thesis.critic import apply_weighted_score, build_critic_summary
from src.thesis.data_access import get_postgis_engine, load_thesis_input, save_dataframe_to_features
from src.thesis.modeling import (
    create_roc_plot,
    create_shap_summary_plot,
    cross_validate_f1,
    feature_importance_df,
    split_training_data,
    train_models,
)
from src.thesis.network_synthesis import run_steiner_from_postgis


def _resolve_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    preferred = [
        "node_connectivity",
        "place_population",
        "place_employment",
        "real_estate_proxy",
        "vitality_retail",
        "vitality_food",
        "vitality_social",
    ]

    existing_preferred = [c for c in preferred if c in df.columns]
    if len(existing_preferred) >= 4:
        return existing_preferred

    excluded = {
        target_col,
        "ageb_id",
        "station_id",
        "nearest_node",
        "geometry",
        "label",
        "balanced_label",
    }
    numeric_cols = [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(numeric_cols) < 4:
        raise ValueError(
            "Need at least 4 numeric NP-RV features. Provide preferred columns or numeric feature columns."
        )
    return numeric_cols


def _confusion_to_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_thesis_pipeline(path: str | None = None, target_column: str = "target") -> Dict[str, Dict[str, float]]:
    engine = get_postgis_engine()
    df = load_thesis_input(path, engine=engine)

    if df.empty:
        raise ValueError("Thesis dataset is empty")

    feature_cols = _resolve_feature_columns(df, target_column)

    # Compute CRITIC weights once; conditionally use labels if target is absent.
    critic = build_critic_summary(df, feature_cols)
    critic_meta = {}
    if target_column not in df.columns:
        df = critic["labeled_df"]
        target_column = "target"
        critic_meta = {
            "critic_threshold": float(critic["threshold"]),
            "critic_positive_rate": float(critic["positive_rate"]),
        }

    validate_target_column(df, target_column)

    # City-wide suitability score used later by network synthesis.
    critic_weights = critic["weights"]
    df = apply_weighted_score(df, critic_weights, feature_cols, score_col="suitability_score")

    # Keep a consistent label column name for downstream geospatial steps.
    if target_column != "target":
        df["target"] = df[target_column]
        target_column = "target"

    X_train, X_test, y_train, y_test = split_training_data(df, feature_cols, target_column)
    models = train_models(X_train, y_train)

    thesis_plot_dir = PLOTS_DIR / "thesis"
    thesis_plot_dir.mkdir(parents=True, exist_ok=True)

    metrics_output = {}

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"thesis_{model_name}", nested=True):
            model_metrics = evaluate_classifier(model, X_test, y_test)
            cv_metrics = cross_validate_f1(model, X_train, y_train)
            metrics = {**model_metrics, **cv_metrics}

            cm = confusion_matrix(y_test, model.predict(X_test))
            cm_metrics = _confusion_to_metrics(cm)
            metrics.update(cm_metrics)

            roc_path = str(thesis_plot_dir / f"{model_name}_roc.png")
            shap_path = str(thesis_plot_dir / f"{model_name}_shap_summary.png")
            fi_path = str(thesis_plot_dir / f"{model_name}_feature_importance.csv")

            create_roc_plot(model, X_test, y_test, roc_path)
            create_shap_summary_plot(model, X_train, shap_path)
            fi_df = feature_importance_df(model, X_train)
            save_dataframe_csv(fi_df, fi_path)

            params = {
                "model_name": model_name,
                "target_column": target_column,
                "feature_count": len(feature_cols),
            }
            params.update({f"critic_w_{k}": float(v) for k, v in critic_weights.items()})
            params.update(critic_meta)

            log_params(params)
            log_metrics(metrics)
            log_artifact(roc_path)
            log_artifact(shap_path)
            log_artifact(fi_path)
            log_model(model, X_train.head(5), run_name=f"{model_name}_model")

            metrics_output[model_name] = {
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in metrics.items()
            }

    # Persist scored AGEB suitability output for phase-3 network synthesis.
    scored_out = METRICS_DIR / "thesis_ageb_scored.csv"
    save_dataframe_csv(df, str(scored_out))

    # Save analysis tables to features schema in PostGIS.
    save_dataframe_to_features(df, "thesis_ageb_scored", engine=engine)

    steiner_status = "skipped"
    if isinstance(df, gpd.GeoDataFrame) or ("geom" in df.columns):
        try:
            run_steiner_from_postgis(
                top_k=12,
                suitability_table="features.thesis_ageb_scored",
                boundary_table="base.ageb",
                output_table="thesis_steiner_corridors",
            )
            steiner_status = "completed"
        except Exception as exc:
            steiner_status = f"failed: {exc}"

    payload = {
        "features_used": feature_cols,
        "critic_weights": {k: float(v) for k, v in critic_weights.items()},
        "models": metrics_output,
        "scored_ageb_path": str(scored_out),
        "postgis_table": "features.thesis_ageb_scored",
        "steiner_status": steiner_status,
    }

    save_json(payload, METRICS_DIR / "thesis_metrics.json")
    return payload
