import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.common.config import DEFAULT_RANDOM_SEED, DEFAULT_TEST_SIZE


def summarize_dataframe(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "null_counts": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()
    return cleaned


def split_features_target(df: pd.DataFrame, target_column: str):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def train_test_split_data(X, y, test_size: float = DEFAULT_TEST_SIZE):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=DEFAULT_RANDOM_SEED,
        stratify=y if len(set(y)) > 1 else None
    )


def scale_numeric_features(X_train, X_test):
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if numeric_cols:
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train_scaled, X_test_scaled, scaler