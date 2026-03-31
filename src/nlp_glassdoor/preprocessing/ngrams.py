from __future__ import annotations

import argparse

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.common.io_utils import load_csv, save_json


def top_ngrams(texts, n=1, top_k=20):
    texts = [str(t).strip() for t in texts if str(t).strip()]
    if not texts:
        return []

    vec = CountVectorizer(ngram_range=(n, n))
    X = vec.fit_transform(texts)
    freqs = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()

    pairs = [(str(term), int(freq)) for term, freq in zip(vocab, freqs)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


def compute_ngrams(df: pd.DataFrame, label: str, n_values=(1, 2), top_k=20):
    subset = df[df["label"] == label]
    texts = subset["clean_text"].dropna().tolist()

    results = {}
    for n in n_values:
        results[f"{n}-gram"] = top_ngrams(texts, n=n, top_k=top_k)

    return results


def main(input_path: str, output_path: str):
    df = load_csv(input_path)

    if "clean_text" not in df.columns:
        raise ValueError("Expected 'clean_text' column")
    if "label" not in df.columns:
        raise ValueError("Expected 'label' column")

    print(f"Input shape: {df.shape}")

    results = {
        "positive": compute_ngrams(df, "positive"),
        "negative": compute_ngrams(df, "negative"),
    }

    save_json(results, output_path)

    print("Top n-grams (positive):")
    for k, v in results["positive"].items():
        print(k, v[:5])

    print("Top n-grams (negative):")
    for k, v in results["negative"].items():
        print(k, v[:5])

    print(f"Saved n-grams to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    main(args.input_path, args.output_path)