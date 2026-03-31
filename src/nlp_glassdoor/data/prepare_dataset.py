from __future__ import annotations

import argparse

import pandas as pd

from src.common.io_utils import load_csv, save_dataframe_csv


def safe_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def build_section_dataframe(
    df: pd.DataFrame,
    text_column: str,
    source_section: str,
    label: str,
) -> pd.DataFrame:
    keep_cols = [
        col
        for col in ["company", "review_title", "rating", "review_date", "source_url", text_column]
        if col in df.columns
    ]

    section_df = df[keep_cols].copy()
    section_df = section_df.rename(columns={text_column: "text"})
    section_df["text"] = section_df["text"].apply(safe_text)
    section_df = section_df[section_df["text"] != ""].copy()

    section_df["source_section"] = source_section
    section_df["label"] = label

    return section_df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"company", "pros", "cons"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print(f"Validated input shape: {df.shape}")

    pros_df = build_section_dataframe(
        df=df,
        text_column="pros",
        source_section="pros",
        label="positive",
    )

    cons_df = build_section_dataframe(
        df=df,
        text_column="cons",
        source_section="cons",
        label="negative",
    )

    text_df = pd.concat([pros_df, cons_df], ignore_index=True)

    text_df["text_length"] = text_df["text"].str.len()
    text_df = text_df[text_df["text_length"] > 0].copy()

    dedupe_cols = [col for col in ["company", "review_title", "source_section", "text"] if col in text_df.columns]
    text_df = text_df.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)

    ordered_cols = [
        col
        for col in [
            "company",
            "review_title",
            "source_section",
            "label",
            "text",
            "text_length",
            "rating",
            "review_date",
            "source_url",
        ]
        if col in text_df.columns
    ]
    text_df = text_df[ordered_cols]

    print(f"Prepared text dataset shape: {text_df.shape}")
    print("Label distribution:")
    print(text_df["label"].value_counts(dropna=False))
    print("Source section distribution:")
    print(text_df["source_section"].value_counts(dropna=False))

    return text_df


def main(input_path: str, output_path: str) -> None:
    df = load_csv(input_path)
    prepared_df = prepare_dataset(df)
    save_dataframe_csv(prepared_df, output_path)
    print(f"Saved prepared dataset to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert validated Glassdoor reviews into row-level NLP dataset."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to validated reviews CSV.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save the prepared text-row dataset CSV.",
    )
    args = parser.parse_args()

    main(args.input_path, args.output_path)