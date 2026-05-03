from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _as_indicator_frame(df: pd.DataFrame, indicator_cols: Iterable[str]) -> pd.DataFrame:
    cols = list(indicator_cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing indicator columns for CRITIC: {missing}")

    x = df[cols].copy()
    if x.isna().all(axis=0).any():
        bad = x.columns[x.isna().all(axis=0)].tolist()
        raise ValueError(f"Indicators with all null values are not valid for CRITIC: {bad}")

    return x.apply(pd.to_numeric, errors="coerce").fillna(x.median(numeric_only=True))


def compute_critic_weights(df: pd.DataFrame, indicator_cols: Iterable[str]) -> Dict[str, float]:
    """Compute objective CRITIC weights.

    CRITIC weight is proportional to:
      contrast_j * conflict_j
    where:
      contrast_j = stddev of criterion j
      conflict_j = sum_k(1 - corr(j, k))
    """
    x = _as_indicator_frame(df, indicator_cols)

    std = x.std(ddof=0)
    corr = x.corr().fillna(0.0)

    conflict = pd.Series(0.0, index=x.columns)
    for c in x.columns:
        conflict[c] = (1.0 - corr.loc[c]).sum()

    critic_scores = (std * conflict).clip(lower=0.0)
    total = float(critic_scores.sum())

    if total <= 0:
        n = len(critic_scores)
        return {k: 1.0 / n for k in critic_scores.index}

    weights = critic_scores / total
    return weights.to_dict()


def apply_weighted_score(
    df: pd.DataFrame,
    weights: Dict[str, float],
    indicator_cols: Iterable[str],
    score_col: str = "nprv_score",
) -> pd.DataFrame:
    """Apply weighted NP-RV score using min-max normalized indicators."""
    cols = list(indicator_cols)
    x = _as_indicator_frame(df, cols)

    min_vals = x.min()
    max_vals = x.max()
    denom = (max_vals - min_vals).replace(0, np.nan)
    x_norm = ((x - min_vals) / denom).fillna(0.0)

    w = pd.Series(weights).reindex(cols).fillna(0.0)
    out = df.copy()
    out[score_col] = x_norm.mul(w, axis=1).sum(axis=1)
    return out


def label_balanced_unbalanced(
    df: pd.DataFrame,
    score_col: str = "nprv_score",
    threshold: float | None = None,
    label_col: str = "target",
) -> Tuple[pd.DataFrame, float]:
    """Label rows as balanced (1) vs unbalanced (0) by score threshold."""
    if score_col not in df.columns:
        raise ValueError(f"Missing score column: {score_col}")

    t = float(df[score_col].median()) if threshold is None else float(threshold)
    out = df.copy()
    out[label_col] = (out[score_col] >= t).astype(int)
    return out, t


def build_critic_summary(
    station_df: pd.DataFrame,
    indicator_cols: List[str],
    threshold: float | None = None,
) -> Dict[str, object]:
    """Compute weights, scores, and labels for station audit training data."""
    weights = compute_critic_weights(station_df, indicator_cols)
    scored = apply_weighted_score(station_df, weights, indicator_cols)
    labeled, used_threshold = label_balanced_unbalanced(scored, threshold=threshold)

    return {
        "weights": weights,
        "threshold": used_threshold,
        "labeled_df": labeled,
        "positive_rate": float(labeled["target"].mean()),
    }
