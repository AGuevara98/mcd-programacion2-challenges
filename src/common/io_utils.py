from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_dataframe_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)