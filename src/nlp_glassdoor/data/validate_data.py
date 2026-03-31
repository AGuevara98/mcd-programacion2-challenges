import pandas as pd
from src.common.io_utils import load_csv, save_dataframe_csv


def safe_text(x):
    return str(x).strip() if x is not None else ""


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Initial shape: {df.shape}")

    # Normalize text columns
    for col in ["pros", "cons", "review_title"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_text)

    # Remove rows with no usable text
    df = df[(df["pros"] != "") | (df["cons"] != "")]
    print(f"After removing empty pros/cons: {df.shape}")

    # Drop duplicates
    df = df.drop_duplicates(subset=["company", "review_title", "pros", "cons"])
    print(f"After deduplication: {df.shape}")

    # Optional: clean rating
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    return df.reset_index(drop=True)


def main(input_path: str, output_path: str):
    df = load_csv(input_path)
    df_clean = validate_dataframe(df)
    save_dataframe_csv(df_clean, output_path)
    print(f"Saved cleaned data to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    main(args.input_path, args.output_path)