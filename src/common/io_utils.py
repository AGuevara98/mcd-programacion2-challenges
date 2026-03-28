from pathlib import Path
import json
import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def save_dataframe_csv(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_json(data: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)