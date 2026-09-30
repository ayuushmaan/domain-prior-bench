"""Metrics: log-loss primary; AUC/Gini, PR-AUC, Brier, ECE-15, KS + time/mem."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    log_loss, roc_auc_score, average_precision_score, brier_score_loss,
)


def ece_score(y_true, p_prob, n_bins=15):
    y_true = np.asarray(y_true); p = np.asarray(p_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    ece, n = 0.0, len(y_true)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


def ks_stat(y_true, p_prob):
    from scipy.stats import ks_2samp
    y_true = np.asarray(y_true); p = np.asarray(p_prob)
    p1, p0 = p[y_true == 1], p[y_true == 0]
    if len(p1) == 0 or len(p0) == 0:
        return float("nan")
    return float(ks_2samp(p1, p0).statistic)


def compute_metrics(y_true, p_prob, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p_prob, dtype=float), 1e-7, 1 - 1e-7)
    out = {}
    out["log_loss"] = float(log_loss(y_true, p, labels=[0, 1]))
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, p))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, p))
    except ValueError:
        out["pr_auc"] = float("nan")
    out["brier"] = float(brier_score_loss(y_true, p))
    out["ece"] = float(ece_score(y_true, p, n_bins))
    out["ks"] = float(ks_stat(y_true, p))
    auc = out["roc_auc"]
    out["gini"] = float(2 * auc - 1) if np.isfinite(auc) else float("nan")
    return out
