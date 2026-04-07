from __future__ import annotations

import re
from collections import Counter

import pandas as pd
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import CountVectorizer

DetectorFactory.seed = 42

try:
    import spacy
except ImportError:
    spacy = None


_SPACY_MODELS: dict[str, object] = {}


def _load_spacy_model(lang: str):
    if spacy is None:
        return None

    model_name = {
        "en": "en_core_web_sm",
        "es": "es_core_news_sm",
    }.get(lang)

    if model_name is None:
        return None

    if model_name not in _SPACY_MODELS:
        _SPACY_MODELS[model_name] = spacy.load(model_name)

    return _SPACY_MODELS[model_name]


def detect_language(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "unknown"

    try:
        lang = detect(text)
        return lang if lang in {"en", "es"} else "unknown"
    except Exception:
        return "unknown"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-záéíóúñü\s]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_text(text: str, lang: str) -> str:
    if not text:
        return ""

    nlp = _load_spacy_model(lang)
    if nlp is None:
        return text

    doc = nlp(text)
    lemmas = [
        token.lemma_.lower().strip()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return " ".join(tok for tok in lemmas if tok)


def expand_reviews_to_segments(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        base = {
            "company": row.get("company"),
            "review_title": row.get("review_title"),
            "rating": row.get("rating"),
            "review_date": row.get("review_date"),
            "source_url": row.get("source_url"),
            "page_number": row.get("page_number"),
        }

        pros = str(row.get("pros", "") or "").strip()
        cons = str(row.get("cons", "") or "").strip()

        if pros:
            rows.append({**base, "segment_label": "pros", "raw_text": pros})
        if cons:
            rows.append({**base, "segment_label": "cons", "raw_text": cons})

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No pros/cons text found after expansion.")

    return out


def preprocess_segment_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["language"] = out["raw_text"].apply(detect_language)
    out["clean_text"] = out["raw_text"].apply(clean_text)
    out["lemma_text"] = out.apply(
        lambda row: lemmatize_text(row["clean_text"], row["language"]),
        axis=1,
    )
    out["text_length"] = out["lemma_text"].str.split().apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    return out


def compute_top_ngrams(texts: list[str], ngram_range=(2, 2), top_k: int = 15):
    usable = [t for t in texts if isinstance(t, str) and t.strip()]
    if not usable:
        return []

    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=1)
    matrix = vectorizer.fit_transform(usable)
    counts = matrix.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()

    pairs = sorted(zip(vocab, counts), key=lambda x: x[1], reverse=True)
    return [(term, int(count)) for term, count in pairs[:top_k]]


def build_ngram_summary(df: pd.DataFrame) -> dict:
    summary = {}

    for lang in ["en", "es"]:
        subset = df[df["language"] == lang]
        summary[lang] = {
            "top_bigrams": compute_top_ngrams(subset["lemma_text"].tolist(), (2, 2)),
            "top_trigrams": compute_top_ngrams(subset["lemma_text"].tolist(), (3, 3)),
        }

    return summary


def token_frequency(texts: list[str], top_k: int = 20):
    counter = Counter()

    for text in texts:
        if isinstance(text, str) and text.strip():
            counter.update(text.split())

    return counter.most_common(top_k)