import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def summarize_dataframe(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }

def split_features_target(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def train_test_split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_numeric_features(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler