from __future__ import annotations

import argparse

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

from src.common.io_utils import load_csv, save_dataframe_csv

DetectorFactory.seed = 42


def detect_language(text: str) -> str:
    if pd.isna(text):
        return "unknown"

    text = str(text).strip()
    if not text:
        return "unknown"

    try:
        lang = detect(text)
        if lang in {"en", "es"}:
            return lang
        return "other"
    except LangDetectException:
        return "unknown"


def split_languages(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "text" not in df.columns:
        raise ValueError("Input dataframe must contain a 'text' column.")

    print(f"Input shape: {df.shape}")

    out = df.copy()
    out["language"] = out["text"].apply(detect_language)

    print("Language distribution:")
    print(out["language"].value_counts(dropna=False))

    df_en = out[out["language"] == "en"].copy().reset_index(drop=True)
    df_es = out[out["language"] == "es"].copy().reset_index(drop=True)

    print(f"English dataset shape: {df_en.shape}")
    print(f"Spanish dataset shape: {df_es.shape}")

    return out, df_en, df_es


def main(input_path: str, output_all_path: str, output_en_path: str, output_es_path: str) -> None:
    df = load_csv(input_path)
    all_df, df_en, df_es = split_languages(df)

    save_dataframe_csv(all_df, output_all_path)
    save_dataframe_csv(df_en, output_en_path)
    save_dataframe_csv(df_es, output_es_path)

    print(f"Saved full dataset with language column to {output_all_path}")
    print(f"Saved English dataset to {output_en_path}")
    print(f"Saved Spanish dataset to {output_es_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect language and split Glassdoor text rows into English and Spanish datasets."
    )
    parser.add_argument("--input_path", required=True, help="Path to prepared text-row dataset CSV.")
    parser.add_argument("--output_all_path", required=True, help="Path to save full dataset with language column.")
    parser.add_argument("--output_en_path", required=True, help="Path to save English-only dataset.")
    parser.add_argument("--output_es_path", required=True, help="Path to save Spanish-only dataset.")
    args = parser.parse_args()

    main(
        input_path=args.input_path,
        output_all_path=args.output_all_path,
        output_en_path=args.output_en_path,
        output_es_path=args.output_es_path,
    )