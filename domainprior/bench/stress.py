"""Stress transforms on real test data (Q2): MNAR missingness, imbalance, shift."""
from __future__ import annotations
import numpy as np
import pandas as pd


def inject_mnar(X, risk_col=None, rate=0.25, seed=0):
    """Missingness depending on a risk-related column (default: first numeric)."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    if not num:
        return X
    ref = risk_col if risk_col in X.columns else num[0]
    z = (X[ref].fillna(X[ref].median()).to_numpy(dtype=float))
    z = (z - z.mean()) / (z.std() + 1e-9)
    p = 1 / (1 + np.exp(-(np.log(rate / (1 - rate + 1e-9)) + z)))
    targets = [c for c in num if c != ref][: max(1, len(num) // 3)]
    for c in targets:
        mask = rng.random(len(X)) < p * 1.5
        X.loc[mask, c] = np.nan
    return X


def downsample_positives(X, y, pos_rate=0.01, seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series(np.asarray(y))
    pos = np.where(y.to_numpy() == 1)[0]; neg = np.where(y.to_numpy() == 0)[0]
    n_pos = max(1, int(len(neg) * pos_rate / (1 - pos_rate)))
    keep_pos = rng.choice(pos, size=min(len(pos), n_pos), replace=False)
    keep = np.sort(np.concatenate([neg, keep_pos]))
    return X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


def covariate_shift_split(X, y, col=None, train_q=0.7):
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    col = col if col in X.columns else (num[0] if num else X.columns[0])
    cut = X[col].fillna(X[col].median() if col in X else 0).quantile(train_q)
    tr_mask = (X[col].fillna(cut - 1) <= cut).to_numpy() if col in X.columns else np.ones(len(X), bool)
    return (X[tr_mask], y[tr_mask], X[~tr_mask], y[~tr_mask])
