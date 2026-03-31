from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import classification_report

from src.common.io_utils import load_csv, save_dataframe_csv, save_json
from src.nlp_glassdoor.sentiment.vader_model import vader_predict
from src.nlp_glassdoor.sentiment.pysentimiento_model import pysentimiento_predict


def run_vader(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pred_sentiment"] = df["text"].apply(vader_predict)
    return df


def run_pysentimiento(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pred_sentiment"] = df["text"].apply(pysentimiento_predict)
    return df


def evaluate(df: pd.DataFrame):
    y_true = df["label"]
    y_pred = df["pred_sentiment"]

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return report


def main(input_path: str, model: str, output_path: str, metrics_path: str):
    df = load_csv(input_path)

    if model == "vader":
        df_out = run_vader(df)
    elif model == "pysentimiento":
        df_out = run_pysentimiento(df)
    else:
        raise ValueError("Model must be 'vader' or 'pysentimiento'")

    metrics = evaluate(df_out)

    save_dataframe_csv(df_out, output_path)
    save_json(metrics, metrics_path)

    print("Saved predictions to:", output_path)
    print("Saved metrics to:", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--metrics_path", required=True)
    args = parser.parse_args()

    main(
        input_path=args.input_path,
        model=args.model,
        output_path=args.output_path,
        metrics_path=args.metrics_path,
    )