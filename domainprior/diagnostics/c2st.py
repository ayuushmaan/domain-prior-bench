"""C2ST: can a classifier tell synthetic from real tables on meta-features?

AUC ~0.5 = prior covers real tables. Report domain AND generic priors.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def c2st(synth_mf: pd.DataFrame, real_mf: pd.DataFrame, cv=5) -> dict:
    X = pd.concat([synth_mf, real_mf], ignore_index=True).fillna(0).to_numpy(dtype=float)
    y = np.array([0] * len(synth_mf) + [1] * len(real_mf))
    cv = min(cv, len(synth_mf), len(real_mf), 5)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, p))
    return {"c2st_auc": round(auc, 3), "n_synth": len(synth_mf), "n_real": len(real_mf),
            "verdict": "covers" if auc < 0.65 else ("marginal" if auc < 0.8 else "gap")}
