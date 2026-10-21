"""Correlation between sentiment and forward returns."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def sentiment_return_correlation(sentiment: ArrayLike, returns: ArrayLike) -> dict[str, float]:
    """Pearson and Spearman correlation between sentiment and forward returns."""
    s = np.asarray(sentiment, dtype=float)
    r = np.asarray(returns, dtype=float)
    if s.shape != r.shape or s.ndim != 1:
        raise ValueError("sentiment and returns must be 1-D arrays of equal length")
    if s.size < 3:
        raise ValueError("need at least 3 observations for correlation")

    pearson_r, pearson_p = stats.pearsonr(s, r)
    spearman_r, spearman_p = stats.spearmanr(s, r)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n": int(s.size),
    }
