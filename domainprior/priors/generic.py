"""Lightweight generic prior placeholder (Weeks 2-3).

Stands in for nanoTabPFN dumps / priorforge / TabICL prior until one is
wired in. Samples dense Gaussian features with a random-MLP label function
so the alpha-mixture and dump pipeline run end-to-end. NOT a substitute
for a proper SCM prior in final results — flag in report.
"""
from __future__ import annotations
import numpy as np
from .base import Task


def sample_generic_task(rng, n_rows=(128, 2048), n_feat=(6, 48),
                        pos_rate=(0.05, 0.5), n_layers=(0, 2)) -> Task:
    n = int(rng.integers(n_rows[0], n_rows[1] + 1))
    d = int(rng.integers(n_feat[0], n_feat[1] + 1))
    X = rng.standard_normal((n, d))
    h = X[:, :max(1, d // 2)]
    for _ in range(int(rng.integers(n_layers[0], n_layers[1] + 1))):
        W = rng.normal(0, 1, (h.shape[1], max(2, h.shape[1] // 2)))
        h = np.tanh(h @ W + rng.normal(0, 0.5, W.shape[1]))
    logit = h @ rng.normal(0, 1, h.shape[1])
    target = float(rng.uniform(*pos_rate))
    lo, hi = -30.0, 30.0
    for _ in range(50):
        b = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit - b)))).mean() > target:
            lo = b
        else:
            hi = b
    y = (rng.random(n) < 1 / (1 + np.exp(-(logit - b)))).astype(np.int64)
    n_tr = int(n * rng.uniform(0.5, 0.8))
    order = rng.permutation(n)
    tr, te = order[:n_tr], order[n_tr:]
    if y[tr].min() == y[tr].max():
        return sample_generic_task(rng, n_rows, n_feat, pos_rate, n_layers)
    return Task(X_train=X[tr], y_train=y[tr], X_test=X[te], y_test=y[te],
                is_cat=np.zeros(d, bool), families=["noise"] * d,
                meta={"pos_rate_train": float(y[tr].mean()), "missing_rate": 0.0,
                      "oot": False, "drifted": False, "generic": True})
