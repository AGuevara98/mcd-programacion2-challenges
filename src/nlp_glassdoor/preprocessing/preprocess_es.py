from __future__ import annotations

import argparse

import pandas as pd
import spacy

from src.common.io_utils import load_csv, save_dataframe_csv
from src.nlp_glassdoor.preprocessing.cleaning import basic_clean_text

NLP = spacy.load("es_core_news_sm", disable=["ner", "parser"])


def preprocess_spanish_text(text: str) -> str:
    text = basic_clean_text(text)
    if not text:
        return ""

    doc = NLP(text)
    tokens = [
        token.lemma_.lower().strip()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.lemma_.strip()
    ]
    return " ".join(tokens).strip()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError("Input dataframe must contain a 'text' column.")

    out = df.copy()
    out["clean_text"] = out["text"].astype(str).apply(preprocess_spanish_text)
    out["clean_text_length"] = out["clean_text"].str.len()
    out = out[out["clean_text"] != ""].reset_index(drop=True)

    print(f"Spanish cleaned shape: {out.shape}")
    return out


def main(input_path: str, output_path: str) -> None:
    df = load_csv(input_path)
    out = preprocess_dataframe(df)
    save_dataframe_csv(out, output_path)
    print(f"Saved cleaned Spanish dataset to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Spanish Glassdoor text.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    main(args.input_path, args.output_path)