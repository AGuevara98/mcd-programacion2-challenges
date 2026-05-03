from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.common.config import DEFAULT_RANDOM_SEED

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

try:
    from pysentimiento import create_analyzer
except ImportError:
    create_analyzer = None


def train_segment_classifier(
    df: pd.DataFrame,
    text_col: str = "lemma_text",
    label_col: str = "segment_label",
):
    usable = df[[text_col, label_col]].dropna().copy()
    usable = usable[usable[text_col].str.strip() != ""]

    if usable.empty:
        raise ValueError("No usable rows available for classifier training.")

    if usable[label_col].nunique() < 2:
        raise ValueError("Classifier requires at least two classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        usable[text_col],
        usable[label_col],
        test_size=0.2,
        random_state=DEFAULT_RANDOM_SEED,
        stratify=usable[label_col],
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=DEFAULT_RANDOM_SEED)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["pros", "cons"]).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    eval_df = pd.DataFrame(
        {
            "text": X_test.values,
            "y_true": y_test.values,
            "y_pred": y_pred,
        }
    )

    return pipeline, metrics, eval_df


def run_vader_sentiment(df: pd.DataFrame, text_col: str = "raw_text") -> pd.DataFrame:
    out = df.copy()

    if SentimentIntensityAnalyzer is None:
        out["vader_compound"] = None
        out["vader_label"] = "not_installed"
        return out

    analyzer = SentimentIntensityAnalyzer()

    def _score_row(row) -> pd.Series:
        if row.get("language", "unknown") != "en":
            return pd.Series({"vader_compound": None, "vader_label": "not_applicable"})
        score = analyzer.polarity_scores(row[text_col])["compound"]
        if score >= 0.05:
            label = "positive"
        elif score <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return pd.Series({"vader_compound": score, "vader_label": label})

    out[["vader_compound", "vader_label"]] = out.apply(_score_row, axis=1)
    return out


def run_pysentimiento(df: pd.DataFrame, text_col: str = "raw_text") -> pd.DataFrame:
    out = df.copy()

    if create_analyzer is None:
        out["pysentimiento_label"] = "not_installed"
        out["pysentimiento_score"] = None
        return out

    analyzers: dict = {}

    def _score_row(row) -> pd.Series:
        lang = row.get("language", "unknown")
        if lang not in {"en", "es"}:
            return pd.Series({"pysentimiento_label": "not_applicable", "pysentimiento_score": None})
        if lang not in analyzers:
            analyzers[lang] = create_analyzer(task="sentiment", lang=lang)
        pred = analyzers[lang].predict(row[text_col])
        return pd.Series({
            "pysentimiento_label": pred.output,
            "pysentimiento_score": float(max(pred.probas.values())),
        })

    out[["pysentimiento_label", "pysentimiento_score"]] = out.apply(_score_row, axis=1)
    return out