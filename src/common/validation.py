from pathlib import Path
import pandas as pd

from src.common.config import SUPPORTED_CHALLENGES
from src.common.exceptions import PipelineInputError, DataValidationError


def validate_challenge_name(challenge_name: str) -> None:
    if challenge_name not in SUPPORTED_CHALLENGES:
        raise PipelineInputError(
            f"Unsupported challenge '{challenge_name}'. "
            f"Supported values: {sorted(SUPPORTED_CHALLENGES)}"
        )


def validate_file_exists(file_path: str) -> None:
    if not Path(file_path).exists():
        raise PipelineInputError(f"File does not exist: {file_path}")


def validate_dataframe_not_empty(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise DataValidationError("The input DataFrame is empty.")


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    if target_column not in df.columns:
        raise DataValidationError(f"Target column '{target_column}' not found in DataFrame.")
    if df[target_column].isna().all():
        raise DataValidationError(f"Target column '{target_column}' contains only null values.")