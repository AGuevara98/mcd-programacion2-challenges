from pathlib import Path

def validate_file_exists(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")

def validate_dataframe_not_empty(df):
    if df.empty:
        raise ValueError("DataFrame is empty")

def validate_target_column(df, column: str):
    if column not in df.columns:
        raise ValueError(f"Missing target column: {column}")

def validate_challenge_name(name: str):
    if name not in {"cancer", "nlp", "thesis"}:
        raise ValueError("Invalid challenge name")