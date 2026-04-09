from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.common.config import METRICS_DIR, PLOTS_DIR, PROCESSED_DATA_DIR
from src.common.io_utils import save_dataframe_csv, save_json
from src.common.mlflow_utils import log_artifact, log_metrics, log_model, log_params
from src.common.validation import validate_file_exists
from src.nlp_glassdoor.model import (
    run_pysentimiento,
    run_vader_sentiment,
    train_segment_classifier,
)
from src.nlp_glassdoor.preprocessing import (
    build_ngram_summary,
    expand_reviews_to_segments,
    preprocess_segment_dataset,
    token_frequency,
)


def _plot_bar(items, title: str, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    labels = [x[0] for x in items]
    values = [x[1] for x in items]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], values[::-1])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_sentiment_counts(df: pd.DataFrame, col: str, title: str, output_path: str):
    counts = df[col].fillna("null").value_counts()
    items = list(zip(counts.index.astype(str).tolist(), counts.values.tolist()))
    _plot_bar(items, title, output_path)


def run_nlp_pipeline(path: str):
    validate_file_exists(path)
    raw_df = pd.read_csv(path)

    if raw_df.empty:
        raise ValueError("The scraped CSV is empty.")

    if "review_title" not in raw_df.columns and "title" in raw_df.columns:
        raw_df = raw_df.rename(columns={"title": "review_title"})

    required = {"company", "review_title", "pros", "cons"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in NLP CSV: {sorted(missing)}")

    segments_path = str(PROCESSED_DATA_DIR / "nlp_segments.csv")
    processed_path = str(PROCESSED_DATA_DIR / "nlp_processed_segments.csv")
    eval_path = str(PROCESSED_DATA_DIR / "nlp_classifier_eval.csv")
    metrics_path = str(METRICS_DIR / "nlp_metrics.json")

    nlp_plot_dir = PLOTS_DIR / "nlp"
    nlp_plot_dir.mkdir(parents=True, exist_ok=True)

    segment_df = expand_reviews_to_segments(raw_df)
    save_dataframe_csv(segment_df, segments_path)

    processed_df = preprocess_segment_dataset(segment_df)
    save_dataframe_csv(processed_df, processed_path)

    classifier_model, classifier_metrics, eval_df = train_segment_classifier(processed_df)
    save_dataframe_csv(eval_df, eval_path)

    vader_df = run_vader_sentiment(processed_df)
    pysent_df = run_pysentimiento(processed_df)

    ngram_summary = build_ngram_summary(processed_df)
    lang_counts = processed_df["language"].value_counts(dropna=False).to_dict()
    label_counts = processed_df["segment_label"].value_counts(dropna=False).to_dict()

    en_tokens = token_frequency(
        processed_df.loc[processed_df["language"] == "en", "lemma_text"].tolist()
    )
    es_tokens = token_frequency(
        processed_df.loc[processed_df["language"] == "es", "lemma_text"].tolist()
    )

    en_plot = str(nlp_plot_dir / "nlp_en_top_tokens.png")
    es_plot = str(nlp_plot_dir / "nlp_es_top_tokens.png")
    vader_plot = str(nlp_plot_dir / "nlp_vader_counts.png")
    pysent_plot = str(nlp_plot_dir / "nlp_pysentimiento_counts.png")

    if en_tokens:
        _plot_bar(en_tokens, "Top English Tokens", en_plot)
    if es_tokens:
        _plot_bar(es_tokens, "Top Spanish Tokens", es_plot)

    _plot_sentiment_counts(vader_df, "vader_label", "VADER Sentiment Counts", vader_plot)
    _plot_sentiment_counts(
        pysent_df,
        "pysentimiento_label",
        "pysentimiento Sentiment Counts",
        pysent_plot,
    )

    summary = {
        "n_scraped_reviews": int(len(raw_df)),
        "n_text_segments": int(len(segment_df)),
        "n_processed_segments": int(len(processed_df)),
        "language_counts": {str(k): int(v) for k, v in lang_counts.items()},
        "segment_label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "ngram_summary": ngram_summary,
    }

    final_metrics = {
        **classifier_metrics,
        "n_processed_segments": int(len(processed_df)),
        "n_languages_detected": int(processed_df["language"].nunique()),
    }

    save_json(
        {
            "summary": summary,
            "classifier_metrics": classifier_metrics,
        },
        metrics_path,
    )

    log_params(
        {
            "challenge": "nlp",
            "model": "tfidf_logistic_regression",
            "sentiment_models": "vader,pysentimiento",
            "n_processed_segments": len(processed_df),
        }
    )
    log_metrics(final_metrics)
    log_artifact(segments_path)
    log_artifact(processed_path)
    log_artifact(eval_path)
    log_artifact(metrics_path)
    log_artifact(vader_plot)
    log_artifact(pysent_plot)

    if Path(en_plot).exists():
        log_artifact(en_plot)
    if Path(es_plot).exists():
        log_artifact(es_plot)

    log_model(
        classifier_model,
        processed_df["lemma_text"].head(20),
        run_name="nlp_segment_classifier",
    )

    return {
        "summary": summary,
        "classifier_metrics": classifier_metrics,
    }