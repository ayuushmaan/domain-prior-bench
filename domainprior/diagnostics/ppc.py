"""Prior predictive check: same learner on synthetic tasks vs real dev splits.

Small gap => prior difficulty resembles real data. Big gap => prior too
easy (or too hard) in ways real data isn't.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from domainprior.models.wrappers import _prep


def _auc(Xtr, ytr, Xte, yte, seed=0):
    ytr = np.asarray(ytr).astype(int)
    yte = np.asarray(yte).astype(int)
    if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
        return float("nan")  # degenerate split: no ranking defined
    n_min = min(np.bincount(ytr))
    Xt, Xe = _prep(pd.DataFrame(np.asarray(Xtr)), pd.DataFrame(np.asarray(Xte)))
    clf = HistGradientBoostingClassifier(
        max_iter=200, early_stopping=(n_min >= 10),  # tiny minority: no val split
        validation_fraction=0.2, random_state=seed)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xt, ytr)
        return float(roc_auc_score(yte, clf.predict_proba(Xe)[:, 1]))
    except Exception:
        return float("nan")


def ppc(synth_tasks, dev_frames: dict, n_repeats=3, seed=0) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sa_all = [_auc(t.X_train, t.y_train, t.X_test, t.y_test, seed) for t in synth_tasks]
    sa = [a for a in sa_all if np.isfinite(a)]
    ra = {}
    for name, (X, y) in dev_frames.items():
        y = np.asarray(y.astype(int) if hasattr(y, "astype") else y)
        sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=0.3, random_state=seed)
        Xa = np.asarray(X)
        aucs = []
        for tr, te in sss.split(Xa, y):
            try:
                aucs.append(_auc(Xa[tr], y[tr], Xa[te], y[te], seed))
            except Exception:
                continue
        ra[name] = [round(float(a), 3) for a in aucs]
    sa = [round(float(a), 3) for a in sa]
    gap = float(np.mean(sa) - np.mean([a for v in ra.values() for a in v])) if (sa and ra) else float("nan")
    return {"synth_auc_mean": round(float(np.mean(sa)), 3) if sa else None,
            "synth_n_valid": len(sa), "synth_n_degenerate": len(sa_all) - len(sa),
            "real_auc_mean": round(float(np.mean([a for v in ra.values() for a in v])), 3) if ra else None,
            "gap_synth_minus_real": round(gap, 3), "real_per_ds": ra}
