from __future__ import annotations

import re
import unicodedata


URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_WORD_RE = re.compile(r"[^a-záéíóúñü\s]")
MULTISPACE_RE = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def basic_clean_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = URL_RE.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = NON_WORD_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text